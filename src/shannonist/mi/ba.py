"""Barber-Agakov mutual-information estimation."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
import math
from typing import Any

import torch
from tensordict import TensorClass, TensorDict
from torch import Tensor, nn

from shannonist.framework import ObjectiveOutput, TrainableEstimator
from shannonist.mi.types import MIBatch, MIEstimate, PairwiseMIBatch
from shannonist.models.flow import FlowDensityEstimator, Invertible, InvertibleMLP
from shannonist.models.mlp import MultiMLP


class Proposal(nn.Module, ABC):
    """Interface for conditional proposal distributions.

    Parameters
    ----------
    dim : int
        Feature dimension of both conditions and density values.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    @abstractmethod
    def forward(self, condition: Tensor) -> TensorDict:
        """Compute distribution parameters conditioned on an input.

        Parameters
        ----------
        condition : Tensor
            Values conditioning the proposal distribution.

        Returns
        -------
        TensorDict
            Proposal-specific distribution parameters.
        """
        ...

    def prob(
        self,
        x: Tensor,
        params: TensorDict | None = None,
    ) -> Tensor:
        """Evaluate the proposal density at arbitrary values.

        This convenience method exponentiates :meth:`log_prob`. Objectives
        should use :meth:`log_prob` directly to avoid numerical underflow.

        Parameters
        ----------
        x : Tensor
            Values at which to evaluate the density.
        params : TensorDict, optional
            Proposal-specific parameters returned by :meth:`forward`. If
            omitted, they are computed by passing ``x`` to :meth:`forward`.

        Returns
        -------
        Tensor
            Probability density for each value in ``x``.
        """
        return self.log_prob(x, params).exp()

    @abstractmethod
    def log_prob(
        self,
        x: Tensor,
        params: TensorDict | None = None,
    ) -> Tensor:
        """Evaluate the proposal log-density at arbitrary values.

        Parameters
        ----------
        x : Tensor
            Values at which to evaluate the log-density.
        params : TensorDict, optional
            Proposal-specific parameters returned by :meth:`forward`. If
            omitted, they are computed by passing ``x`` to :meth:`forward`.

        Returns
        -------
        Tensor
            Log-probability density for each value in ``x``.
        """
        ...


class EntropyEstimator(nn.Module, ABC):
    """Interface for estimating entropy from latent observations.

    Parameters
    ----------
    dim : int
        Feature dimension of the latent observations.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Estimate entropy for latent observations.

        Parameters
        ----------
        x : Tensor
            Latent observations with shape ``(*, dim)``.

        Returns
        -------
        Tensor
            Entropy estimate with shape ``(*)``.
        """
        ...

    def compute_objectives(
        self,
        predictions: Tensor,
    ) -> ObjectiveOutput | None:
        """Return an optional objective for fitting the entropy estimator.

        Parameters
        ----------
        predictions : Tensor
            Entropy values returned by :meth:`forward`.

        Returns
        -------
        ObjectiveOutput or None
            Estimator-specific training objective, or ``None`` when the
            estimator requires no gradient-based fitting.
        """
        del predictions
        return None


class StandardNormalEntropyEstimator(EntropyEstimator):
    """Return the entropy of a standard normal distribution.

    This deliberately simple default assumes every latent feature is an
    independent standard normal random variable. It does not inspect the
    empirical distribution of its input.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Return standard-normal entropy for every latent observation.

        Parameters
        ----------
        x : Tensor
            Latent observations with shape ``(*, dim)``.

        Returns
        -------
        Tensor
            Constant entropy values with shape ``(*)``.

        Raises
        ------
        ValueError
            If ``x`` does not have the configured feature dimension.
        """
        if x.ndim == 0 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have trailing dimension {self.dim}")
        entropy = self.dim / 2 * (1 + math.log(2 * math.pi))
        return x.new_full(x.shape[:-1], entropy)


class GaussianEntropyEstimator(EntropyEstimator):
    """Online entropy estimator for a diagonal Gaussian fit.

    The estimator maintains the sample count, mean, and sum of squared
    deviations as buffers. In training mode, every forward pass updates these
    statistics from the observed samples. In evaluation mode, the accumulated
    fit is used without modification. The fitted variance is the maximum-
    likelihood population variance.

    Parameters
    ----------
    dim : int
        Feature dimension of the latent observations.
    min_variance : float, default=1e-8
        Floor applied to fitted variances before computing entropy.
    """

    count: Tensor
    mean: Tensor
    m2: Tensor

    def __init__(self, dim: int, min_variance: float = 1e-8) -> None:
        super().__init__(dim)
        if min_variance <= 0:
            raise ValueError("min_variance must be positive")
        self.min_variance = min_variance
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("m2", torch.zeros(dim))

    @property
    def variance(self) -> Tensor:
        """Return the fitted diagonal population variance."""
        denominator = self.count.clamp_min(1).to(self.m2.dtype)
        return self.m2 / denominator

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Update the diagonal Gaussian fit from a batch of samples.

        Parameters
        ----------
        x : Tensor
            Latent observations with shape ``(*, dim)``.

        Raises
        ------
        ValueError
            If ``x`` has an incompatible shape or contains no samples.
        """
        self._validate_input(x)
        samples = x.detach().reshape(-1, self.dim).to(self.mean)
        batch_count = samples.shape[0]
        if batch_count == 0:
            raise ValueError("x must contain at least one sample")

        batch_mean = samples.mean(dim=0)
        batch_m2 = (samples - batch_mean).square().sum(dim=0)
        previous_count = self.count.to(self.mean.dtype)
        total_count = previous_count + batch_count
        delta = batch_mean - self.mean
        updated_mean = self.mean + delta * (batch_count / total_count)
        updated_m2 = (
            self.m2
            + batch_m2
            + delta.square() * previous_count * batch_count / total_count
        )

        self.mean.copy_(updated_mean)
        self.m2.copy_(updated_m2)
        self.count.add_(batch_count)

    def reset(self) -> None:
        """Discard all accumulated samples and fitted statistics."""
        self.count.zero_()
        self.mean.zero_()
        self.m2.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Update the fit when training and return its entropy.

        Parameters
        ----------
        x : Tensor
            Latent observations with shape ``(*, dim)``.

        Returns
        -------
        Tensor
            Fitted diagonal-Gaussian entropy with shape ``(*)``.

        Raises
        ------
        RuntimeError
            If entropy is requested in evaluation mode before any samples
            have been observed.
        """
        self._validate_input(x)
        if self.training:
            self.update(x)
        if self.count.item() == 0:
            raise RuntimeError("no samples have been observed")

        variance = self.variance.clamp_min(self.min_variance)
        entropy = 0.5 * torch.log(2 * torch.pi * math.e * variance).sum()
        return entropy.to(x).expand(x.shape[:-1])

    def _validate_input(self, x: Tensor) -> None:
        """Validate a latent observation tensor."""
        if x.ndim == 0 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have trailing dimension {self.dim}")


class FlowEntropyEstimator(EntropyEstimator):
    """Estimate entropy with a trainable normalizing-flow density model.

    Per-sample entropy contributions are the negative log-density under the
    wrapped flow. The optional training objective is their mean, corresponding
    to maximum-likelihood fitting of the flow to observed samples.

    Parameters
    ----------
    density_estimator : FlowDensityEstimator
        Normalizing-flow density model. Its transform must expose a positive
        integer ``dim`` attribute.
    """

    def __init__(self, density_estimator: FlowDensityEstimator) -> None:
        if not isinstance(density_estimator, FlowDensityEstimator):
            raise TypeError(
                "density_estimator must be a FlowDensityEstimator"
            )
        dim = getattr(density_estimator.transform, "dim", None)
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                "density estimator transform must expose a positive integer dim"
            )
        super().__init__(dim)
        self.density_estimator = density_estimator

    def forward(self, x: Tensor) -> Tensor:
        """Return negative flow log-density for each observation.

        Parameters
        ----------
        x : Tensor
            Latent observations with shape ``(*, dim)``.

        Returns
        -------
        Tensor
            Per-observation entropy contributions with shape ``(*)``.

        Raises
        ------
        ValueError
            If ``x`` has an incompatible feature dimension.
        """
        if x.ndim == 0 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have trailing dimension {self.dim}")
        return -self.density_estimator.log_prob(x.detach())

    def compute_objectives(self, predictions: Tensor) -> ObjectiveOutput:
        """Compute the mean negative log-likelihood of observed samples.

        Parameters
        ----------
        predictions : Tensor
            Per-observation negative log-densities from :meth:`forward`.

        Returns
        -------
        ObjectiveOutput
            Mean flow negative log-likelihood and diagnostic values.
        """
        loss = predictions.mean()
        metrics = TensorDict(
            {"negative_log_likelihood": predictions},
            batch_size=[],
        )
        return ObjectiveOutput(
            loss=loss,
            estimate=predictions.mean(),
            metrics=metrics,
            batch_size=[],
        )


class FlowProposal(Proposal):
    r"""Flow proposal for the pairwise conditional density ``q(x | y)``.

    Given paired variables ``x`` and ``y`` and a flow density ``p_flow``, this
    proposal uses the location-family model

    .. math::

        q(x \mid y) = p_{\mathrm{flow}}(x - \mu_\theta(y)).

    The location model defaults to a linear projection and the wrapped flow
    learns the distribution of pairwise residuals. There is no additional
    conditioning variable.

    Parameters
    ----------
    density_estimator : FlowDensityEstimator
        Normalizing-flow density model for residuals. Its transform must expose
        a positive integer ``dim`` attribute.
    location_model : nn.Module, optional
        Module mapping ``y`` to the proposal location. Defaults to a linear
        projection from ``dim`` to ``dim``.
    """

    def __init__(
        self,
        density_estimator: FlowDensityEstimator,
        location_model: nn.Module | None = None,
    ) -> None:
        if not isinstance(density_estimator, FlowDensityEstimator):
            raise TypeError(
                "density_estimator must be a FlowDensityEstimator"
            )
        dim = getattr(density_estimator.transform, "dim", None)
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                "density estimator transform must expose a positive integer dim"
            )
        super().__init__(dim)
        self.density_estimator = density_estimator
        if location_model is None:
            location_model = nn.Linear(dim, dim)
            nn.init.xavier_uniform_(location_model.weight)
            nn.init.zeros_(location_model.bias)
        if not isinstance(location_model, nn.Module):
            raise TypeError("location_model must be an nn.Module")
        self.location_model = location_model

    def forward(self, y: Tensor) -> TensorDict:
        """Return parameters for the proposal ``q(x | y)``.

        Parameters
        ----------
        y : Tensor
            Values of the conditioning random variable with shape
            ``(*, dim)``.

        Returns
        -------
        TensorDict
            Parameters containing ``location`` with shape ``(*, dim)``.

        Raises
        ------
        ValueError
            If ``y`` has an incompatible feature dimension.
        """
        self._validate_input(y, "y")
        location = self.location_model(y)
        self._validate_input(location, "location")
        return TensorDict(
            {"location": location},
            batch_size=y.shape[:-1],
        )

    def log_prob(
        self,
        x: Tensor,
        params: TensorDict | None = None,
    ) -> Tensor:
        """Evaluate the conditional flow log-density.

        Parameters
        ----------
        x : Tensor
            Values with shape ``(*, dim)``.
        params : TensorDict, optional
            Parameters containing ``location`` returned by :meth:`forward`. If
            omitted, parameters are computed from ``x``, yielding a zero
            residual only when the location model maps ``x`` to ``x``.

        Returns
        -------
        Tensor
            Conditional log-density with shape ``(*)``.
        """
        self._validate_input(x, "x")
        if params is None:
            params = self(x)
        location = params["location"]
        self._validate_input(location, "location")
        return self.density_estimator.log_prob(x - location)

    def _validate_input(self, x: Tensor, name: str) -> None:
        """Validate a proposal value tensor."""
        if x.ndim == 0 or x.shape[-1] != self.dim:
            raise ValueError(f"{name} must have trailing dimension {self.dim}")


class GaussianProposal(Proposal):
    r"""Diagonal Gaussian proposal.

    The proposal is parameterized as

    .. math::

        q(y \mid x) = \mathcal{N}(\mu_\theta(x),
        \operatorname{diag}(\exp(\operatorname{logvar}_\theta(x)))).

    Both parameters are computed by linear projections.  The mean projection
    uses Xavier-uniform weights and a zero bias, giving a zero expected mean
    for zero-mean inputs.  The log-variance projection starts at zero, so the
    initial standard deviation is exactly one.

    Parameters
    ----------
    dim : int
        Number of features in both the conditioning input and Gaussian value.
    min_logvar : float, default=-20.0
        Minimum predicted log-variance.
    max_logvar : float, default=20.0
        Maximum predicted log-variance.
    """

    def __init__(
        self,
        dim: int,
        min_logvar: float = -20.0,
        max_logvar: float = 20.0,
    ) -> None:
        super().__init__(dim)
        if min_logvar >= max_logvar:
            raise ValueError("min_logvar must be less than max_logvar")
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.mu = nn.Linear(dim, dim)
        self.logvar = nn.Linear(dim, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the proposal near a standard normal distribution."""
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, condition: Tensor) -> TensorDict:
        """Compute Gaussian parameters for a conditioning value.

        Parameters
        ----------
        condition : Tensor
            Conditioning values with shape ``(*, dim)``.

        Returns
        -------
        TensorDict
            Parameters ``mu`` and ``logvar``, each with shape ``(*, dim)``.
        """
        mu = self.mu(condition)
        logvar = self.logvar(condition).clamp(
            min=self.min_logvar,
            max=self.max_logvar,
        )
        batch_size = condition.shape[:-1]
        return TensorDict(
            {"mu": mu, "logvar": logvar},
            batch_size=batch_size,
        )

    def log_prob(
        self,
        x: Tensor,
        params: TensorDict | None = None,
    ) -> Tensor:
        """Evaluate the diagonal Gaussian log-density at arbitrary values.

        Parameters
        ----------
        x : Tensor
            Values with trailing dimension ``dim`` at which to evaluate the
            density.
        params : TensorDict, optional
            Gaussian parameters containing ``mu`` and ``logvar``. If omitted,
            the parameters are computed from ``x``.

        Returns
        -------
        Tensor
            Joint log-density across the final feature dimension.

        Raises
        ------
        ValueError
            If ``x`` does not have the proposal's output feature dimension.
        KeyError
            If ``params`` does not contain ``mu`` and ``logvar``.
        """
        if x.ndim == 0 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have trailing dimension {self.dim}")

        if params is None:
            params = self(x)
        assert params is not None
        mu = params["mu"]
        logvar = params["logvar"]
        distribution = torch.distributions.Independent(
            torch.distributions.Normal(mu, torch.exp(0.5 * logvar)),
            1,
        )
        return distribution.log_prob(x)


def make_entropy_estimator(
    name: str,
    dim: int,
    opts: Mapping[str, Any] | None = None,
) -> EntropyEstimator:
    """Construct an entropy estimator from a short name.

    Parameters
    ----------
    name : str
        One of ``"standard_normal"``, ``"gaussian"``, or ``"flow"``.
    dim : int
        Feature dimension of latent observations.
    opts : Mapping[str, Any], optional
        Constructor options. For ``"flow"``, ``transform``, ``prior``, and
        ``hidden_dims`` are recognized. The default transform is an
        :class:`InvertibleMLP` with one hidden layer of width ``dim``.

    Returns
    -------
    EntropyEstimator
        Constructed entropy estimator.

    Raises
    ------
    ValueError
        If ``name`` is unknown.
    """
    options = dict(opts or {})
    normalized_name = name.lower().replace("-", "_")
    if normalized_name in {"standard_normal", "normal", "standard"}:
        return StandardNormalEntropyEstimator(dim=dim, **options)
    if normalized_name == "gaussian":
        return GaussianEntropyEstimator(dim=dim, **options)
    if normalized_name == "flow":
        density = _make_flow_density_estimator(dim, options)
        return FlowEntropyEstimator(density)
    raise ValueError(f"unknown entropy estimator: {name}")


def make_proposal(
    name: str,
    dim: int,
    opts: Mapping[str, Any] | None = None,
) -> Proposal:
    """Construct a conditional proposal from a short name.

    Parameters
    ----------
    name : str
        One of ``"gaussian"`` or ``"flow"``.
    dim : int
        Feature dimension of proposal values.
    opts : Mapping[str, Any], optional
        Constructor options. For ``"flow"``, ``transform``, ``prior``, and
        ``hidden_dims`` are recognized. The default transform is an
        :class:`InvertibleMLP` with one hidden layer of width ``dim``.

    Returns
    -------
    Proposal
        Constructed conditional proposal.

    Raises
    ------
    ValueError
        If ``name`` is unknown.
    """
    options = dict(opts or {})
    normalized_name = name.lower().replace("-", "_")
    if normalized_name == "gaussian":
        return GaussianProposal(dim=dim, **options)
    if normalized_name == "flow":
        location_model = options.pop("location_model", None)
        density = _make_flow_density_estimator(dim, options)
        return FlowProposal(density, location_model=location_model)
    raise ValueError(f"unknown proposal: {name}")


def _make_flow_density_estimator(
    dim: int,
    options: dict[str, Any],
) -> FlowDensityEstimator:
    """Build a flow density with sklearn-style defaults."""
    density = options.pop("density_estimator", None)
    if density is not None:
        if options:
            unexpected = ", ".join(sorted(options))
            raise ValueError(
                "density_estimator cannot be combined with options: "
                f"{unexpected}"
            )
        if not isinstance(density, FlowDensityEstimator):
            raise TypeError("density_estimator must be a FlowDensityEstimator")
        return density

    transform = options.pop("transform", None)
    prior = options.pop("prior", None)
    hidden_dims = options.pop("hidden_dims", (dim,))
    if options:
        unexpected = ", ".join(sorted(options))
        raise ValueError(f"unknown flow options: {unexpected}")
    if transform is None:
        transform = InvertibleMLP(
            input_dim=dim,
            hidden_dims=hidden_dims,
            output_dim=dim,
        )
    if not isinstance(transform, Invertible):
        raise TypeError("transform must be an Invertible")
    return FlowDensityEstimator(transform=transform, prior=prior)


class JointBAOutput(TensorClass):
    """Predictions and proposal parameters produced by :class:`JointBA`.

    Parameters
    ----------
    hx : Tensor
        Encoded values of ``x``.
    hy : Tensor
        Encoded values of ``y``.
    conditional_log_prob : Tensor
        Log-density of encoded ``x`` under the proposal conditioned on
        encoded ``y``.
    entropy : Tensor
        Entropy estimate for encoded ``x``.
    conditional_params : TensorDict
        Parameters of the conditional proposal.
    """

    hx: Tensor
    hy: Tensor
    conditional_log_prob: Tensor
    entropy: Tensor
    conditional_params: TensorDict


def joint_ba_loss(
    predictions: JointBAOutput,
) -> tuple[Tensor, TensorDict]:
    r"""Compute the negative Barber-Agakov lower bound.

    The per-example bound is

    .. math::

        H(x) + \log q_{\mathrm{cond}}(x \mid y).

    Parameters
    ----------
    predictions : JointBAOutput
        Output returned by :meth:`JointBA.compute_forward`.

    Returns
    -------
    tuple[Tensor, TensorDict]
        Negative mean lower bound and per-example diagnostics.

    Raises
    ------
    ValueError
        If the conditional log-density and entropy shapes differ or the
        log-density contains non-finite values.
    """
    conditional_log_prob = predictions.conditional_log_prob
    entropy_vec = predictions.entropy.detach()
    if conditional_log_prob.shape != entropy_vec.shape:
        raise ValueError(
            "conditional log-probability and entropy must have identical shapes"
        )
    if not torch.all(torch.isfinite(conditional_log_prob)):
        raise ValueError("conditional log-probability must be finite")

    estimate_vec = entropy_vec + conditional_log_prob
    loss_vec = -estimate_vec
    loss = loss_vec.mean()
    details = TensorDict(
        {
            "loss_vec": loss_vec,
            "estimate_vec": estimate_vec,
            "entropy_vec": entropy_vec,
            "conditional_log_prob": conditional_log_prob,
        },
        batch_size=[],
    )
    return loss, details


class JointBA(
    nn.Module,
    TrainableEstimator[MIBatch, JointBAOutput, MIEstimate],
):
    """Joint Barber-Agakov mutual-information estimator.

    Parameters
    ----------
    dim : int
        Feature dimension of each input observation.
    enc_dim : int
        Feature dimension produced by the encoder.
    conditional_proposal : Proposal or str, default="gaussian"
        Conditional proposal instance or factory name.
    entropy_estimator : EntropyEstimator or str, default="gaussian"
        Entropy estimator instance or factory name.
    encoder : nn.Module, optional
        Module mapping ``dim`` input features to ``enc_dim`` encoded features.
        Defaults to a linear projection.
    estimator_opts : Mapping[str, Any], optional
        Options passed to :func:`make_entropy_estimator` when
        ``entropy_estimator`` is a string.
    proposal_opts : Mapping[str, Any], optional
        Options passed to :func:`make_proposal` when ``conditional_proposal``
        is a string.

    Raises
    ------
    ValueError
        If ``dim`` or ``enc_dim`` is not positive, or a component dimension
        differs from ``enc_dim``.
    TypeError
        If a supplied component does not implement its required interface.

    """

    def __init__(
        self,
        dim: int,
        enc_dim: int,
        conditional_proposal: Proposal | str | None = "gaussian",
        entropy_estimator: EntropyEstimator | str | None = "gaussian",
        encoder: nn.Module | None = None,
        estimator_opts: Mapping[str, Any] | None = None,
        proposal_opts: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if enc_dim <= 0:
            raise ValueError("enc_dim must be positive")

        if conditional_proposal is None:
            conditional_proposal = "gaussian"
        if entropy_estimator is None:
            entropy_estimator = "gaussian"
        if isinstance(conditional_proposal, str):
            conditional_proposal = make_proposal(
                conditional_proposal,
                enc_dim,
                proposal_opts,
            )
        elif proposal_opts:
            raise ValueError(
                "proposal_opts require conditional_proposal to be a string"
            )
        if isinstance(entropy_estimator, str):
            entropy_estimator = make_entropy_estimator(
                entropy_estimator,
                enc_dim,
                estimator_opts,
            )
        elif estimator_opts:
            raise ValueError(
                "estimator_opts require entropy_estimator to be a string"
            )
        if not isinstance(conditional_proposal, Proposal):
            raise TypeError(
                "conditional_proposal must be an instance of Proposal"
            )
        if not isinstance(entropy_estimator, EntropyEstimator):
            raise TypeError(
                "entropy_estimator must be an instance of EntropyEstimator"
            )
        if conditional_proposal.dim != enc_dim:
            raise ValueError("conditional_proposal.dim must match enc_dim")
        if entropy_estimator.dim != enc_dim:
            raise ValueError("entropy_estimator.dim must match enc_dim")

        self.dim = dim
        self.enc_dim = enc_dim
        self.encoder = encoder if encoder is not None else nn.Linear(dim, enc_dim)
        self.conditional_proposal = conditional_proposal
        self.entropy_estimator = entropy_estimator

    def compute_forward(self, batch: MIBatch) -> JointBAOutput:
        """Encode paired inputs and evaluate entropy and conditional density.

        Parameters
        ----------
        batch : MIBatch
            Paired, unmasked observations.

        Returns
        -------
        JointBAOutput
            Encoded inputs, conditional parameters, log-density, and entropy.

        Raises
        ------
        NotImplementedError
            If either observation mask is present.
        """
        if batch.x_mask is not None or batch.y_mask is not None:
            raise NotImplementedError("JointBA does not yet support masks")

        hx = self.encoder(batch.x)
        hy = self.encoder(batch.y)
        conditional_params = self.conditional_proposal(hy)
        conditional_log_prob = self.conditional_proposal.log_prob(
            hx,
            conditional_params,
        )
        entropy = self.entropy_estimator(hx)
        return JointBAOutput(
            hx=hx,
            hy=hy,
            conditional_log_prob=conditional_log_prob,
            entropy=entropy,
            conditional_params=conditional_params,
            batch_size=hx.shape[:-1],
        )

    def compute_objectives(
        self,
        predictions: JointBAOutput,
    ) -> ObjectiveOutput:
        """Compute the negative Barber-Agakov lower bound.

        Parameters
        ----------
        predictions : JointBAOutput
            Output returned by :meth:`compute_forward`.

        Returns
        -------
        ObjectiveOutput
            Scalar loss and bound diagnostics.
        """
        ba_loss, details = joint_ba_loss(predictions)
        entropy_objective = self.entropy_estimator.compute_objectives(
            predictions.entropy
        )
        loss = ba_loss
        details["ba_loss"] = ba_loss.detach()
        if entropy_objective is not None:
            loss = loss + entropy_objective.loss
            details["entropy_loss"] = entropy_objective.loss.detach()
        return ObjectiveOutput(
            loss=loss,
            estimate=-ba_loss,
            metrics=details,
            batch_size=[],
        )

    def estimate(self, batch: MIBatch) -> MIEstimate:
        """Estimate the Barber-Agakov lower bound.

        Parameters
        ----------
        batch : MIBatch
            Paired observations.

        Returns
        -------
        MIEstimate
            Scalar lower-bound estimate and diagnostics.
        """
        predictions = self.compute_forward(batch)
        objective = self.compute_objectives(predictions)
        return MIEstimate(
            value=objective.estimate,
            details=objective.metrics,
            batch_size=[],
        )


class PairwiseBAOutput(TensorClass):
    r"""Predictions produced by pairwise Barber-Agakov estimation.

    Parameters
    ----------
    hx : Tensor
        Encoded observations with shape ``(*, count, enc_dim)``.
    conditional_log_prob : Tensor
        Directed conditional log-densities with shape
        ``(*, count, count)``. Entry ``[..., i, j]`` evaluates
        :math:`q(x_i \mid x_j)`.
    entropy : Tensor
        Marginal entropy contributions with shape ``(*, count)``.
    conditional_params : TensorDict
        Proposal parameters with batch shape ``(*, count, count)``.
    mask : Tensor
        Boolean valid-position mask with shape ``(*, count)``.
    """

    hx: Tensor
    conditional_log_prob: Tensor
    entropy: Tensor
    conditional_params: TensorDict
    mask: Tensor


def pairwise_ba_loss(
    predictions: PairwiseBAOutput,
) -> tuple[Tensor, TensorDict]:
    r"""Compute the mean symmetric BA loss over all distinct pairs.

    For each ordered pair ``(i, j)``, the directed lower bound is

    .. math::

        H(x_i) + \mathbb{E}[\log q(x_i \mid x_j)].

    The two directions are averaged for each unordered pair. All leading
    dimensions are flattened into the sample dimension, masked observations
    are excluded, and the diagonal is reported as zero.

    Parameters
    ----------
    predictions : PairwiseBAOutput
        Encoded observations, directed log-densities, entropies, and mask.

    Returns
    -------
    tuple[Tensor, TensorDict]
        Mean loss over unique pairs and pairwise diagnostics.

    Raises
    ------
    ValueError
        If prediction shapes are incompatible, a valid log-density is not
        finite, or any pair has no jointly valid samples.
    """
    hx = predictions.hx
    conditional_log_prob = predictions.conditional_log_prob
    entropy = predictions.entropy.detach()
    mask = predictions.mask
    if hx.ndim < 3:
        raise ValueError("hx must have shape (*, count, enc_dim)")
    count = hx.shape[-2]
    expected_pair_shape = (*hx.shape[:-1], count)
    if conditional_log_prob.shape != expected_pair_shape:
        raise ValueError(
            "conditional_log_prob must have shape (*, count, count)"
        )
    if entropy.shape != hx.shape[:-1]:
        raise ValueError("entropy must have shape (*, count)")
    if mask.shape != hx.shape[:-1]:
        raise ValueError("mask must have shape (*, count)")

    conditional_log_prob = conditional_log_prob.reshape(-1, count, count)
    entropy = entropy.reshape(-1, count)
    mask = mask.reshape(-1, count).bool()
    pair_valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    pair_counts = pair_valid.sum(dim=0)
    upper_triangle = torch.triu(
        torch.ones(count, count, dtype=torch.bool, device=hx.device),
        diagonal=1,
    )
    diagonal = torch.eye(count, dtype=torch.bool, device=hx.device)
    missing = upper_triangle & (pair_counts == 0)
    if missing.any():
        i, j = missing.nonzero(as_tuple=False)[0].tolist()
        raise ValueError(f"pair ({i}, {j}) has no jointly valid samples")
    if not torch.all(torch.isfinite(conditional_log_prob[pair_valid])):
        raise ValueError("valid conditional log-probabilities must be finite")

    directed_estimate = entropy.unsqueeze(-1) + conditional_log_prob
    estimate_vec = (directed_estimate + directed_estimate.transpose(-1, -2)) / 2
    estimate_vec = estimate_vec.masked_fill(~pair_valid, 0)
    estimate_vec = estimate_vec.masked_fill(diagonal.unsqueeze(0), 0)
    estimate_matrix = estimate_vec.sum(dim=0) / pair_counts.clamp_min(1)
    estimate_matrix = estimate_matrix.masked_fill(diagonal, 0)
    loss_matrix = -estimate_matrix
    loss = loss_matrix[upper_triangle].mean()
    valid_counts = pair_counts.masked_fill(diagonal, 0)
    details = TensorDict(
        {
            "loss_vec": -estimate_vec,
            "loss_matrix": loss_matrix,
            "estimate_vec": estimate_vec,
            "estimate_matrix": estimate_matrix,
            "entropy_vec": entropy,
            "conditional_log_prob": conditional_log_prob,
            "mask": mask,
            "valid_counts": valid_counts,
        },
        batch_size=[],
    )
    return loss, details


class PairwiseBA(
    nn.Module,
    TrainableEstimator[PairwiseMIBatch, PairwiseBAOutput, MIEstimate],
):
    """Jointly estimate a matrix of pairwise Barber-Agakov bounds.

    Parameters
    ----------
    dim : int
        Feature dimension of each observed variable.
    enc_dim : int
        Feature dimension produced for each variable by the encoder.
    count : int
        Number of variables in the input's penultimate dimension.
    conditional_proposal : Proposal or str, default="gaussian"
        Shared conditional proposal instance or factory name.
    entropy_estimator : EntropyEstimator or str, default="gaussian"
        Shared marginal entropy estimator instance or factory name.
    encoder : nn.Module, optional
        Module mapping ``(*, count, dim)`` to ``(*, count, enc_dim)``.
        Defaults to independent linear projections implemented by
        :class:`MultiMLP`.
    estimator_opts : Mapping[str, Any], optional
        Options passed to :func:`make_entropy_estimator`.
    proposal_opts : Mapping[str, Any], optional
        Options passed to :func:`make_proposal`.
    """

    def __init__(
        self,
        dim: int,
        enc_dim: int,
        count: int,
        conditional_proposal: Proposal | str | None = "gaussian",
        entropy_estimator: EntropyEstimator | str | None = "gaussian",
        encoder: nn.Module | None = None,
        estimator_opts: Mapping[str, Any] | None = None,
        proposal_opts: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if enc_dim <= 0:
            raise ValueError("enc_dim must be positive")
        if count < 2:
            raise ValueError("pairwise BA requires count to be at least two")

        if conditional_proposal is None:
            conditional_proposal = "gaussian"
        if entropy_estimator is None:
            entropy_estimator = "gaussian"
        if isinstance(conditional_proposal, str):
            conditional_proposal = make_proposal(
                conditional_proposal,
                enc_dim,
                proposal_opts,
            )
        elif proposal_opts:
            raise ValueError(
                "proposal_opts require conditional_proposal to be a string"
            )
        if isinstance(entropy_estimator, str):
            entropy_estimator = make_entropy_estimator(
                entropy_estimator,
                enc_dim,
                estimator_opts,
            )
        elif estimator_opts:
            raise ValueError(
                "estimator_opts require entropy_estimator to be a string"
            )
        if not isinstance(conditional_proposal, Proposal):
            raise TypeError(
                "conditional_proposal must be an instance of Proposal"
            )
        if not isinstance(entropy_estimator, EntropyEstimator):
            raise TypeError(
                "entropy_estimator must be an instance of EntropyEstimator"
            )
        if conditional_proposal.dim != enc_dim:
            raise ValueError("conditional_proposal.dim must match enc_dim")
        if entropy_estimator.dim != enc_dim:
            raise ValueError("entropy_estimator.dim must match enc_dim")

        self.dim = dim
        self.enc_dim = enc_dim
        self.count = count
        self.encoder = encoder if encoder is not None else MultiMLP(
            input_dim=dim,
            count=count,
            output_dim=enc_dim,
            hidden_dim=(),
        )
        self.conditional_proposal = conditional_proposal
        self.entropy_estimator = entropy_estimator

    def compute_forward(self, batch: PairwiseMIBatch) -> PairwiseBAOutput:
        """Encode observations and evaluate every directed BA proposal.

        Parameters
        ----------
        batch : PairwiseMIBatch
            Observations with shape ``(*, count, dim)``.

        Returns
        -------
        PairwiseBAOutput
            Encoded observations and all pairwise density terms.
        """
        x = batch.x
        if x.ndim < 3:
            raise ValueError("x must have shape (*, count, dim)")
        if x.shape[-2] != self.count:
            raise ValueError(
                f"expected count dimension {self.count}, got {x.shape[-2]}"
            )
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"expected feature dimension {self.dim}, got {x.shape[-1]}"
            )
        mask = self._normalize_mask(batch.x_mask, x)
        masked_x = x.masked_fill(~mask.unsqueeze(-1), 0)
        hx = self.encoder(masked_x)
        expected_hx_shape = (*x.shape[:-1], self.enc_dim)
        if hx.shape != expected_hx_shape:
            raise ValueError(
                "encoder must return shape (*, count, enc_dim); "
                f"expected {expected_hx_shape}, got {tuple(hx.shape)}"
            )

        condition = hx.unsqueeze(-3).expand(
            *hx.shape[:-2],
            self.count,
            self.count,
            self.enc_dim,
        )
        target = hx.unsqueeze(-2).expand_as(condition)
        conditional_params = self.conditional_proposal(condition)
        conditional_log_prob = self.conditional_proposal.log_prob(
            target,
            conditional_params,
        )

        valid_entropy = self.entropy_estimator(hx[mask])
        entropy = hx.new_zeros(mask.shape)
        entropy[mask] = valid_entropy
        return PairwiseBAOutput(
            hx=hx,
            conditional_log_prob=conditional_log_prob,
            entropy=entropy,
            conditional_params=conditional_params,
            mask=mask,
            batch_size=hx.shape[:-2],
        )

    def _normalize_mask(self, mask: Tensor | None, x: Tensor) -> Tensor:
        """Return a boolean mask with shape ``(*, count)``."""
        expected_shape = x.shape[:-1]
        if mask is None:
            return torch.ones(expected_shape, dtype=torch.bool, device=x.device)
        if mask.shape == (*expected_shape, 1):
            mask = mask.squeeze(-1)
        if mask.shape != expected_shape:
            raise ValueError(
                "x_mask must have shape (*, count) or (*, count, 1)"
            )
        return mask.to(device=x.device, dtype=torch.bool)

    def compute_objectives(
        self,
        predictions: PairwiseBAOutput,
    ) -> ObjectiveOutput:
        """Compute the mean BA objective over all unique pairs."""
        ba_loss, details = pairwise_ba_loss(predictions)
        entropy_objective = self.entropy_estimator.compute_objectives(
            predictions.entropy[predictions.mask]
        )
        loss = ba_loss
        details["ba_loss"] = ba_loss.detach()
        if entropy_objective is not None:
            loss = loss + entropy_objective.loss
            details["entropy_loss"] = entropy_objective.loss.detach()
        return ObjectiveOutput(
            loss=loss,
            estimate=details["estimate_matrix"],
            metrics=details,
            batch_size=[],
        )

    def estimate(self, batch: PairwiseMIBatch) -> MIEstimate:
        """Estimate a symmetric pairwise mutual-information matrix."""
        predictions = self.compute_forward(batch)
        objective = self.compute_objectives(predictions)
        return MIEstimate(
            value=objective.estimate,
            details=objective.metrics,
            batch_size=[],
        )


class SampledPairwiseBAOutput(TensorClass):
    """Predictions produced by BA over sampled variable pairs.

    Parameters
    ----------
    hx : Tensor
        Encoded sampled pairs with shape
        ``(batch, num_samples, 2, enc_dim)``.
    conditional_log_prob : Tensor
        Directed conditional log-densities with shape
        ``(batch, num_samples, 2)``. The last dimension contains
        ``log q(left | right)`` and ``log q(right | left)``.
    entropy : Tensor
        Marginal entropy contributions with shape
        ``(batch, num_samples, 2)``.
    conditional_params : TensorDict
        Proposal parameters with batch shape ``(batch, num_samples, 2)``.
    position_indices : Tensor
        Sampled source indices with shape ``(batch, num_samples, 2)``.
    """

    hx: Tensor
    conditional_log_prob: Tensor
    entropy: Tensor
    conditional_params: TensorDict
    position_indices: Tensor


def sampled_pairwise_ba_loss(
    predictions: SampledPairwiseBAOutput,
) -> tuple[Tensor, TensorDict]:
    """Compute BA over sampled variable pairs without negative samples.

    The two directed BA estimates are averaged for each sampled pair, then
    averaged over observations and sampled pair slots.

    Parameters
    ----------
    predictions : SampledPairwiseBAOutput
        Encoded pairs and their BA density terms.

    Returns
    -------
    tuple[Tensor, TensorDict]
        Scalar loss and per-observation and per-pair diagnostics.

    Raises
    ------
    ValueError
        If prediction shapes are incompatible or a density is not finite.
    """
    hx = predictions.hx
    conditional_log_prob = predictions.conditional_log_prob
    entropy = predictions.entropy.detach()
    position_indices = predictions.position_indices
    if hx.ndim != 4 or hx.shape[-2] != 2:
        raise ValueError(
            "hx must have shape (batch, num_samples, 2, enc_dim)"
        )
    expected_shape = hx.shape[:-1]
    if conditional_log_prob.shape != expected_shape:
        raise ValueError(
            "conditional_log_prob must have shape (batch, num_samples, 2)"
        )
    if entropy.shape != expected_shape:
        raise ValueError("entropy must have shape (batch, num_samples, 2)")
    if position_indices.shape != expected_shape:
        raise ValueError(
            "position_indices must have shape (batch, num_samples, 2)"
        )
    if hx.shape[1] < 1:
        raise ValueError("at least one sampled position pair is required")
    if not torch.all(torch.isfinite(conditional_log_prob)):
        raise ValueError("conditional log-probabilities must be finite")

    directed_estimate = entropy + conditional_log_prob
    estimate_vec = directed_estimate.mean(dim=-1)
    estimate_by_pair = estimate_vec.mean(dim=0)
    loss_vec = -estimate_vec
    loss_by_pair = -estimate_by_pair
    loss = loss_by_pair.mean()
    details = TensorDict(
        {
            "loss_vec": loss_vec,
            "loss_by_pair": loss_by_pair,
            "estimate_vec": estimate_vec,
            "estimate_by_pair": estimate_by_pair,
            "directed_estimate": directed_estimate,
            "entropy_vec": entropy,
            "conditional_log_prob": conditional_log_prob,
            "position_indices": position_indices,
        },
        batch_size=[],
    )
    return loss, details


class SampledPairwiseBA(
    nn.Module,
    TrainableEstimator[
        PairwiseMIBatch,
        SampledPairwiseBAOutput,
        MIEstimate,
    ],
):
    """Estimate BA bounds over randomly sampled variable pairs.

    A shared encoder, proposal, and entropy estimator are applied to pairs
    sampled from tensors shaped ``(*, count, features)``. Sampling selects
    variables only; unlike contrastive estimators, this objective uses no
    positive or negative sample construction.

    Parameters
    ----------
    dim : int
        Feature dimension of each observed variable.
    enc_dim : int
        Feature dimension produced by the shared encoder.
    sample_size : int
        Maximum number of variable pairs sampled per observation and pass.
    conditional_proposal : Proposal or str, default="gaussian"
        Shared conditional proposal instance or factory name.
    entropy_estimator : EntropyEstimator or str, default="gaussian"
        Shared marginal entropy estimator instance or factory name.
    encoder : nn.Module, optional
        Shared module mapping ``dim`` features to ``enc_dim`` features.
        Defaults to a linear projection.
    estimator_opts : Mapping[str, Any], optional
        Options passed to :func:`make_entropy_estimator`.
    proposal_opts : Mapping[str, Any], optional
        Options passed to :func:`make_proposal`.
    """

    def __init__(
        self,
        dim: int,
        enc_dim: int,
        sample_size: int,
        conditional_proposal: Proposal | str | None = "gaussian",
        entropy_estimator: EntropyEstimator | str | None = "gaussian",
        encoder: nn.Module | None = None,
        estimator_opts: Mapping[str, Any] | None = None,
        proposal_opts: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if enc_dim <= 0:
            raise ValueError("enc_dim must be positive")
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")

        if conditional_proposal is None:
            conditional_proposal = "gaussian"
        if entropy_estimator is None:
            entropy_estimator = "gaussian"
        if isinstance(conditional_proposal, str):
            conditional_proposal = make_proposal(
                conditional_proposal,
                enc_dim,
                proposal_opts,
            )
        elif proposal_opts:
            raise ValueError(
                "proposal_opts require conditional_proposal to be a string"
            )
        if isinstance(entropy_estimator, str):
            entropy_estimator = make_entropy_estimator(
                entropy_estimator,
                enc_dim,
                estimator_opts,
            )
        elif estimator_opts:
            raise ValueError(
                "estimator_opts require entropy_estimator to be a string"
            )
        if not isinstance(conditional_proposal, Proposal):
            raise TypeError(
                "conditional_proposal must be an instance of Proposal"
            )
        if not isinstance(entropy_estimator, EntropyEstimator):
            raise TypeError(
                "entropy_estimator must be an instance of EntropyEstimator"
            )
        if conditional_proposal.dim != enc_dim:
            raise ValueError("conditional_proposal.dim must match enc_dim")
        if entropy_estimator.dim != enc_dim:
            raise ValueError("entropy_estimator.dim must match enc_dim")

        self.dim = dim
        self.enc_dim = enc_dim
        self.sample_size = sample_size
        self.encoder = encoder if encoder is not None else nn.Linear(dim, enc_dim)
        self.conditional_proposal = conditional_proposal
        self.entropy_estimator = entropy_estimator

    def compute_forward(
        self,
        batch: PairwiseMIBatch,
    ) -> SampledPairwiseBAOutput:
        """Sample valid variable pairs and evaluate both BA directions.

        Parameters
        ----------
        batch : PairwiseMIBatch
            Observations with shape ``(*, count, dim)`` and an optional mask
            shaped ``(*, count)`` or ``(*, count, 1)``.

        Returns
        -------
        SampledPairwiseBAOutput
            Encoded sampled pairs, proposal terms, and source indices.
        """
        x = batch.x
        if x.ndim < 3:
            raise ValueError("x must have shape (*, count, dim)")
        count, feature_dim = x.shape[-2:]
        if feature_dim != self.dim:
            raise ValueError(
                f"expected feature dimension {self.dim}, got {feature_dim}"
            )
        x = x.reshape(-1, count, feature_dim)
        batch_size = x.shape[0]
        mask = self._normalize_mask(batch.x_mask, batch.x)
        mask = mask.reshape(batch_size, count)
        valid_counts = mask.sum(dim=-1)
        minimum_valid = int(valid_counts.min().item())
        if minimum_valid < 2:
            raise ValueError(
                "every sample must contain at least two valid positions"
            )
        num_samples = min(self.sample_size, minimum_valid)
        position_indices = self._sample_position_pairs(mask, num_samples)

        batch_indices = torch.arange(batch_size, device=x.device)[:, None, None]
        sampled = x[batch_indices, position_indices]
        hx = self.encoder(sampled.reshape(-1, feature_dim)).reshape(
            batch_size,
            num_samples,
            2,
            -1,
        )
        expected_hx_shape = (batch_size, num_samples, 2, self.enc_dim)
        if hx.shape != expected_hx_shape:
            raise ValueError(
                "encoder must return shape (batch, num_samples, 2, enc_dim); "
                f"expected {expected_hx_shape}, got {tuple(hx.shape)}"
            )

        target = hx
        condition = hx.flip(dims=(-2,))
        conditional_params = self.conditional_proposal(condition)
        conditional_log_prob = self.conditional_proposal.log_prob(
            target,
            conditional_params,
        )
        entropy = self.entropy_estimator(hx)
        return SampledPairwiseBAOutput(
            hx=hx,
            conditional_log_prob=conditional_log_prob,
            entropy=entropy,
            conditional_params=conditional_params,
            position_indices=position_indices,
            batch_size=[batch_size, num_samples],
        )

    @staticmethod
    def _normalize_mask(mask: Tensor | None, x: Tensor) -> Tensor:
        """Return a boolean mask matching the variable dimensions."""
        expected_shape = x.shape[:-1]
        if mask is None:
            return torch.ones(expected_shape, dtype=torch.bool, device=x.device)
        if mask.shape == (*expected_shape, 1):
            mask = mask.squeeze(-1)
        if mask.shape != expected_shape:
            raise ValueError(
                "x_mask must have shape (*, count) or (*, count, 1)"
            )
        return mask.to(device=x.device, dtype=torch.bool)

    @staticmethod
    def _sample_position_pairs(mask: Tensor, num_samples: int) -> Tensor:
        """Sample valid position pairs by shared valid-position ordinal."""
        minimum_valid = int(mask.sum(dim=-1).min().item())
        left_ordinals = torch.randperm(
            minimum_valid,
            device=mask.device,
        )[:num_samples]
        right_ordinals = torch.randint(
            minimum_valid - 1,
            (num_samples,),
            device=mask.device,
        )
        right_ordinals = right_ordinals + (
            right_ordinals >= left_ordinals
        )

        valid_positions = torch.stack(
            [
                sample_mask.nonzero(as_tuple=False).squeeze(-1)[:minimum_valid]
                for sample_mask in mask
            ]
        )
        left = valid_positions[:, left_ordinals]
        right = valid_positions[:, right_ordinals]
        return torch.stack((left, right), dim=-1)

    def compute_objectives(
        self,
        predictions: SampledPairwiseBAOutput,
    ) -> ObjectiveOutput:
        """Compute the BA objective over sampled variable pairs."""
        ba_loss, details = sampled_pairwise_ba_loss(predictions)
        entropy_objective = self.entropy_estimator.compute_objectives(
            predictions.entropy
        )
        loss = ba_loss
        details["ba_loss"] = ba_loss.detach()
        if entropy_objective is not None:
            loss = loss + entropy_objective.loss
            details["entropy_loss"] = entropy_objective.loss.detach()
        return ObjectiveOutput(
            loss=loss,
            estimate=details["estimate_vec"].mean(),
            metrics=details,
            batch_size=[],
        )

    def estimate(self, batch: PairwiseMIBatch) -> MIEstimate:
        """Estimate mean MI across newly sampled variable pairs."""
        predictions = self.compute_forward(batch)
        objective = self.compute_objectives(predictions)
        return MIEstimate(
            value=objective.estimate,
            details=objective.metrics,
            batch_size=[],
        )


__all__ = [
    "EntropyEstimator",
    "FlowEntropyEstimator",
    "FlowProposal",
    "GaussianEntropyEstimator",
    "GaussianProposal",
    "JointBA",
    "JointBAOutput",
    "PairwiseBA",
    "PairwiseBAOutput",
    "SampledPairwiseBA",
    "SampledPairwiseBAOutput",
    "Proposal",
    "StandardNormalEntropyEstimator",
    "make_entropy_estimator",
    "make_proposal",
    "joint_ba_loss",
    "pairwise_ba_loss",
    "sampled_pairwise_ba_loss",
]

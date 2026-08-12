"""Interfaces for invertible neural-network transformations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from tensordict import TensorClass
from torch import Tensor, nn
from torch.distributions import Distribution, Independent, Normal
from torch.nn import functional as F


class InvertibleOutput(TensorClass):
    """Output of an invertible transformation.

    Parameters
    ----------
    value : Tensor
        Transformed tensor.
    log_abs_det : Tensor
        Logarithm of the absolute Jacobian determinant for each transformed
        value.
    """

    value: Tensor
    log_abs_det: Tensor


class FlowDensityOutput(TensorClass):
    """Sample or density evaluation produced by a normalizing flow.

    Parameters
    ----------
    value : Tensor
        Value in the modeled data space.
    latent : Tensor
        Corresponding value in the prior's latent space.
    log_prob : Tensor
        Log-density of ``value`` under the flow distribution.
    log_abs_det : Tensor
        Log-absolute-determinant used for this transformation direction.
    """

    value: Tensor
    latent: Tensor
    log_prob: Tensor
    log_abs_det: Tensor


class Invertible(nn.Module, ABC):
    """Interface for an invertible neural-network transformation."""

    @abstractmethod
    def forward(self, x: Tensor) -> InvertibleOutput:
        """Apply the forward transformation.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        InvertibleOutput
            Transformed tensor and forward log-absolute-determinant.
        """
        ...

    @abstractmethod
    def inverse(self, y: Tensor) -> InvertibleOutput:
        """Apply the inverse transformation.

        Parameters
        ----------
        y : Tensor
            Transformed tensor.

        Returns
        -------
        InvertibleOutput
            Reconstructed input and inverse log-absolute-determinant.
        """
        ...


class InvertibleLinear(Invertible):
    r"""Apply an invertible affine transformation.

    The forward transformation has the same convention as
    :class:`torch.nn.Linear`:

    .. math::

        y = x W^\mathsf{T} + b.

    The square weight is parameterized as ``P @ L @ U``. Initialization first
    samples an orthogonal matrix and decomposes it into PLU factors. ``P`` and
    the signs of the diagonal of ``U`` remain fixed, while the strict
    triangles and logarithm of the diagonal magnitudes are learned. The
    resulting matrix is therefore invertible by construction. :meth:`inverse`
    uses triangular solves rather than explicitly constructing a matrix
    inverse.

    Parameters
    ----------
    dim : int
        Input and output feature dimension.
    bias : bool, default=True
        Whether to include an additive bias.

    Attributes
    ----------
    l_params : nn.Parameter
        Unconstrained parameters for the strict lower triangle of ``L``.
    u_params : nn.Parameter
        Unconstrained parameters for the strict upper triangle of ``U``.
    log_diag : nn.Parameter
        Logarithm of the absolute diagonal of ``U``.
    P : Tensor
        Fixed permutation matrix.
    sign_diag : Tensor
        Fixed signs of the diagonal of ``U``.
    weight : Tensor
        Assembled square transformation matrix with shape ``(dim, dim)``.
    bias : nn.Parameter or None
        Optional bias with shape ``(dim,)``.
    """

    P: Tensor
    sign_diag: Tensor

    def __init__(self, dim: int, bias: bool = True) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.l_params = nn.Parameter(torch.empty(dim, dim))
        self.u_params = nn.Parameter(torch.empty(dim, dim))
        self.log_diag = nn.Parameter(torch.empty(dim))
        self.register_buffer("P", torch.empty(dim, dim))
        self.register_buffer("sign_diag", torch.empty(dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize from the PLU decomposition of an orthogonal matrix."""
        with torch.no_grad():
            initial_weight = torch.empty_like(self.l_params)
            nn.init.orthogonal_(initial_weight)
            lu, pivots = torch.linalg.lu_factor(initial_weight)
            permutation, lower, upper = torch.lu_unpack(lu, pivots)
            diagonal = upper.diagonal()

            self.P.copy_(permutation)
            self.sign_diag.copy_(diagonal.sign())
            self.l_params.copy_(torch.tril(lower, diagonal=-1))
            self.u_params.copy_(torch.triu(upper, diagonal=1))
            self.log_diag.copy_(diagonal.abs().log())
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def L(self) -> Tensor:
        """Return the unit lower-triangular factor."""
        identity = torch.eye(
            self.dim,
            dtype=self.l_params.dtype,
            device=self.l_params.device,
        )
        return torch.tril(self.l_params, diagonal=-1) + identity

    def U(self) -> Tensor:
        """Return the upper-triangular factor with nonzero diagonal."""
        return torch.triu(self.u_params, diagonal=1) + torch.diag(
            self.sign_diag * self.log_diag.exp()
        )

    @property
    def weight(self) -> Tensor:
        """Return the assembled invertible weight matrix."""
        return self.P @ self.L() @ self.U()

    @property
    def log_abs_det(self) -> Tensor:
        """Return the log-absolute-determinant of the weight."""
        return self.log_diag.sum()

    def forward(self, x: Tensor) -> InvertibleOutput:
        """Apply the affine transformation.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(*, dim)``.

        Returns
        -------
        InvertibleOutput
            Transformed values and forward log-absolute-determinants with
            shapes ``(*, dim)`` and ``(*)``, respectively.

        Raises
        ------
        ValueError
            If the input feature dimension is incompatible.
        """
        self._validate_input(x)
        batch_size = x.shape[:-1]
        return InvertibleOutput(
            value=F.linear(x, self.weight, self.bias),
            log_abs_det=self.log_abs_det.expand(batch_size),
            batch_size=batch_size,
        )

    def inverse(self, y: Tensor) -> InvertibleOutput:
        """Invert the affine transformation using an LU solve.

        Parameters
        ----------
        y : Tensor
            Transformed values with shape ``(*, dim)``.

        Returns
        -------
        InvertibleOutput
            Reconstructed inputs and inverse log-absolute-determinants with
            shapes ``(*, dim)`` and ``(*)``, respectively.

        Raises
        ------
        ValueError
            If the input feature dimension is incompatible.
        """
        self._validate_input(y)
        centered = y if self.bias is None else y - self.bias
        flattened = centered.reshape(-1, self.dim)
        right_hand_side = self.P.transpose(0, 1) @ flattened.transpose(0, 1)
        intermediate = torch.linalg.solve_triangular(
            self.L(),
            right_hand_side,
            upper=False,
            unitriangular=True,
        )
        solution = torch.linalg.solve_triangular(
            self.U(),
            intermediate,
            upper=True,
        )
        batch_size = y.shape[:-1]
        return InvertibleOutput(
            value=solution.transpose(0, 1).reshape_as(centered),
            log_abs_det=(-self.log_abs_det).expand(batch_size),
            batch_size=batch_size,
        )

    def _validate_input(self, x: Tensor) -> None:
        """Validate the trailing feature dimension."""
        if x.ndim == 0 or x.shape[-1] != self.dim:
            raise ValueError(f"input must have trailing dimension {self.dim}")


class InvertibleMLP(Invertible):
    """Stack invertible affine layers into a reversible module.

    Because :class:`InvertibleLinear` is square, every requested width must be
    identical. Without nonlinear transformations, this stack remains an
    affine map; the class primarily provides composition and reverse-order
    inversion.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : Sequence[int]
        Feature dimension for each intermediate layer.
    output_dim : int
        Output feature dimension.

    Raises
    ------
    ValueError
        If any dimension is non-positive or the dimensions are not identical.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        dimensions = (input_dim, *hidden_dims, output_dim)
        if any(dim <= 0 for dim in dimensions):
            raise ValueError("all dimensions must be positive")
        if any(dim != input_dim for dim in dimensions[1:]):
            raise ValueError(
                "all dimensions must match for an invertible linear stack"
            )

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.output_dim = output_dim
        self.dim = input_dim
        self.layers = nn.ModuleList(
            InvertibleLinear(input_dim) for _ in range(len(dimensions) - 1)
        )

    def forward(self, x: Tensor) -> InvertibleOutput:
        """Apply every invertible layer in order.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(*, input_dim)``.

        Returns
        -------
        InvertibleOutput
            Transformed values and accumulated forward
            log-absolute-determinants.
        """
        log_abs_det = x.new_zeros(x.shape[:-1])
        for layer in self.layers:
            output = layer(x)
            x = output.value
            log_abs_det = log_abs_det + output.log_abs_det
        return InvertibleOutput(
            value=x,
            log_abs_det=log_abs_det,
            batch_size=x.shape[:-1],
        )

    def inverse(self, y: Tensor) -> InvertibleOutput:
        """Apply every layer inverse in reverse order.

        Parameters
        ----------
        y : Tensor
            Transformed values with shape ``(*, output_dim)``.

        Returns
        -------
        InvertibleOutput
            Reconstructed inputs and accumulated inverse
            log-absolute-determinants.
        """
        log_abs_det = y.new_zeros(y.shape[:-1])
        for layer in reversed(self.layers):
            output = layer.inverse(y)
            y = output.value
            log_abs_det = log_abs_det + output.log_abs_det
        return InvertibleOutput(
            value=y,
            log_abs_det=log_abs_det,
            batch_size=y.shape[:-1],
        )


class FlowDensityEstimator(nn.Module):
    r"""Density model obtained by transforming a prior distribution.

    If ``z`` follows the configured prior and ``x = f(z)``, the modeled
    density is

    .. math::

        \log p_X(x) = \log p_Z(f^{-1}(x))
        + \log |\det J_{f^{-1}}(x)|.

    Parameters
    ----------
    transform : Invertible
        Invertible transformation mapping prior samples into data space.
    prior : Distribution, optional
        Prior distribution providing ``sample`` and ``log_prob``. Defaults to
        an independent standard Gaussian whose dimension is inferred from the
        transform.
    """

    prior_loc: Tensor | None
    prior_scale: Tensor | None

    def __init__(
        self,
        transform: Invertible,
        prior: Distribution | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(transform, Invertible):
            raise TypeError("transform must be an Invertible")
        if prior is not None and not isinstance(prior, Distribution):
            raise TypeError("prior must be a torch Distribution")

        self.transform = transform
        self._prior = prior
        if prior is None:
            dim = getattr(transform, "dim", None)
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(
                    "transform must expose a positive integer dim when prior "
                    "is omitted"
                )
            self.register_buffer("prior_loc", torch.zeros(dim))
            self.register_buffer("prior_scale", torch.ones(dim))
        else:
            self.register_buffer("prior_loc", None)
            self.register_buffer("prior_scale", None)

    @property
    def prior(self) -> Distribution:
        """Return the configured or default prior distribution."""
        if self._prior is not None:
            return self._prior
        assert self.prior_loc is not None
        assert self.prior_scale is not None
        return Independent(Normal(self.prior_loc, self.prior_scale), 1)

    def forward(
        self,
        sample_shape: torch.Size | Sequence[int] = torch.Size(),
    ) -> FlowDensityOutput:
        """Sample from the prior and transform into data space.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int], default=torch.Size()
            Leading shape of the requested samples.

        Returns
        -------
        FlowDensityOutput
            Data-space samples, their latent values, log-densities, and
            forward log-absolute-determinants.
        """
        latent = self.prior.sample(torch.Size(sample_shape))
        transformed = self.transform(latent)
        prior_log_prob = self.prior.log_prob(latent)
        self._validate_log_prob_shapes(prior_log_prob, transformed.log_abs_det)
        return FlowDensityOutput(
            value=transformed.value,
            latent=latent,
            log_prob=prior_log_prob - transformed.log_abs_det,
            log_abs_det=transformed.log_abs_det,
            batch_size=prior_log_prob.shape,
        )

    def evaluate(self, value: Tensor) -> FlowDensityOutput:
        """Evaluate data-space values using the inverse transformation.

        Parameters
        ----------
        value : Tensor
            Values in the modeled data space.

        Returns
        -------
        FlowDensityOutput
            Input values, recovered latent values, log-densities, and inverse
            log-absolute-determinants.
        """
        inverted = self.transform.inverse(value)
        prior_log_prob = self.prior.log_prob(inverted.value)
        self._validate_log_prob_shapes(prior_log_prob, inverted.log_abs_det)
        return FlowDensityOutput(
            value=value,
            latent=inverted.value,
            log_prob=prior_log_prob + inverted.log_abs_det,
            log_abs_det=inverted.log_abs_det,
            batch_size=prior_log_prob.shape,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        """Return the flow log-density of arbitrary data-space values."""
        return self.evaluate(value).log_prob

    def prob(self, value: Tensor) -> Tensor:
        """Return the flow density of arbitrary data-space values."""
        return self.log_prob(value).exp()

    @staticmethod
    def _validate_log_prob_shapes(
        prior_log_prob: Tensor,
        log_abs_det: Tensor,
    ) -> None:
        """Validate compatibility of prior density and flow Jacobian."""
        if prior_log_prob.shape != log_abs_det.shape:
            raise ValueError(
                "prior log_prob and transform log_abs_det must have "
                "identical shapes"
            )


__all__ = [
    "FlowDensityEstimator",
    "FlowDensityOutput",
    "Invertible",
    "InvertibleLinear",
    "InvertibleMLP",
    "InvertibleOutput",
]

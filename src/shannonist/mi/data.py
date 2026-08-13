import math
from collections.abc import Sequence

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import Dataset


def tensordict_collate(samples: Sequence[TensorDict]) -> TensorDict:
    """Stack unbatched TensorDict samples into a batch.

    Parameters
    ----------
    samples : Sequence[TensorDict]
        Unbatched samples returned by a dataset.

    Returns
    -------
    TensorDict
        TensorDict with a new leading batch dimension.

    Raises
    ------
    ValueError
        If ``samples`` is empty.
    """
    if not samples:
        raise ValueError("cannot collate an empty sequence")
    return torch.stack(list(samples), dim=0)


def tensordict_passthrough(sample: TensorDict) -> TensorDict:
    """Return an already-batched TensorDict without additional collation.

    Parameters
    ----------
    sample : TensorDict
        Batch constructed by the dataset itself.

    Returns
    -------
    TensorDict
        The same TensorDict instance.
    """
    return sample


def _pairwise_gaussian_parameters(
    mutual_information: Tensor | Sequence[Sequence[float]],
    dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate pairwise MI and construct its Gaussian correlation factor."""
    mi = torch.as_tensor(mutual_information, dtype=torch.float64)
    if mi.ndim != 2 or mi.shape[0] != mi.shape[1]:
        raise ValueError("mutual_information must be a square matrix")
    if mi.shape[0] < 2:
        raise ValueError("mutual_information must describe at least two variables")
    if not torch.isfinite(mi).all():
        raise ValueError("mutual_information must contain only finite values")
    if (mi < 0).any():
        raise ValueError("mutual_information must be nonnegative")
    if not torch.allclose(mi, mi.transpose(0, 1)):
        raise ValueError("mutual_information must be symmetric")
    if not torch.allclose(torch.diag(mi), torch.zeros_like(torch.diag(mi))):
        raise ValueError("mutual_information diagonal must be zero")

    correlation = torch.sqrt(-torch.expm1(-2.0 * mi / dim))
    correlation.fill_diagonal_(1.0)
    eigenvalues, eigenvectors = torch.linalg.eigh(correlation)
    tolerance = 1e-8 * max(1.0, eigenvalues.abs().max().item())
    if eigenvalues.min().item() < -tolerance:
        raise ValueError(
            "mutual_information produces a correlation matrix that is not "
            "positive semidefinite"
        )

    eigenvalues = eigenvalues.clamp_min(0)
    factor = eigenvectors @ torch.diag_embed(torch.sqrt(eigenvalues))
    dtype = torch.get_default_dtype()
    return mi.to(dtype), correlation.to(dtype), factor.to(dtype)


class CorrelatedGausian(Dataset[TensorDict]):
    r"""Synthetic paired Gaussian data with prescribed mutual information.

    Samples follow

    .. math::

        X \sim \mathcal{N}(0, I_d), \qquad
        Y = \rho X + \sqrt{1 - \rho^2}\,\epsilon,

    where :math:`\epsilon \sim \mathcal{N}(0, I_d)` is independent of
    :math:`X` and

    .. math::

        \rho = \sqrt{1 - \exp(-2 I^* / d)}.

    Parameters
    ----------
    mutual_information : float
        Desired mutual information in nats.
    dim : int, default=1
        Dimensionality of each Gaussian observation.
    num_samples : int, default=10000
        Number of samples exposed by the dataset.

    Attributes
    ----------
    rho : float
        Correlation coefficient corresponding to ``mutual_information``.
    """

    def __init__(
        self,
        mutual_information: float,
        dim: int = 1,
        num_samples: int = 10_000,
    ) -> None:
        if mutual_information < 0:
            raise ValueError("mutual_information must be nonnegative")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_samples < 0:
            raise ValueError("num_samples must be nonnegative")

        self.mutual_information = mutual_information
        self.dim = dim
        self.num_samples = num_samples
        self.rho = math.sqrt(-math.expm1(-2.0 * mutual_information / dim))

    def __len__(self) -> int:
        """Return the number of samples exposed by the dataset."""
        return self.num_samples

    def __getitem__(self, index: int) -> TensorDict:
        """Generate a correlated Gaussian pair.

        Parameters
        ----------
        index : int
            Sample index. Values outside the dataset bounds are rejected; the
            sample itself is generated lazily and independently of the index.

        Returns
        -------
        TensorDict
            Unbatched TensorDict containing ``x`` and ``y``, each with shape
            ``(dim,)``.

        Raises
        ------
        IndexError
            If ``index`` is outside the dataset bounds.
        """
        if not -self.num_samples <= index < self.num_samples:
            raise IndexError("dataset index out of range")

        x = torch.randn(self.dim)
        epsilon = torch.randn(self.dim)
        noise_scale = math.sqrt(1.0 - self.rho**2)
        y = self.rho * x + noise_scale * epsilon
        return TensorDict({"x": x, "y": y}, batch_size=[])


class PairwiseCorrelatedGaussian(Dataset[TensorDict]):
    r"""Gaussian vectors with prescribed pairwise mutual information.

    For a pairwise mutual-information matrix ``M``, off-diagonal correlations
    are computed as

    .. math::

        \rho_{ij} = \sqrt{1 - \exp(-2 M_{ij} / d)}.

    The resulting correlation matrix ``R`` is factored as
    :math:`R = A A^\mathsf{T}`. Each sample draws an independent latent matrix
    :math:`Z \in \mathbb{R}^{k \times d}` and returns :math:`X = A Z`.

    Parameters
    ----------
    mutual_information : Tensor or Sequence[Sequence[float]]
        Symmetric ``(count, count)`` matrix of desired pairwise mutual
        information in nats. The diagonal must be zero and is replaced by unit
        self-correlation when constructing the correlation matrix.
    dim : int, default=1
        Dimensionality of each Gaussian variable.
    num_samples : int, default=10000
        Number of samples exposed by the dataset.

    Attributes
    ----------
    count : int
        Number of Gaussian variables.
    correlation : Tensor
        Validated ``(count, count)`` correlation matrix.
    factor : Tensor
        Matrix factor satisfying ``factor @ factor.T == correlation`` up to
        numerical precision.

    Raises
    ------
    ValueError
        If the MI matrix is malformed or produces a correlation matrix that is
        not positive semidefinite.
    """

    def __init__(
        self,
        mutual_information: Tensor | Sequence[Sequence[float]],
        dim: int = 1,
        num_samples: int = 10_000,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_samples < 0:
            raise ValueError("num_samples must be nonnegative")

        mi, correlation, factor = _pairwise_gaussian_parameters(
            mutual_information,
            dim,
        )
        self.mutual_information = mi
        self.dim = dim
        self.num_samples = num_samples
        self.count = mi.shape[0]
        self.correlation = correlation
        self.factor = factor

    def __len__(self) -> int:
        """Return the number of samples exposed by the dataset."""
        return self.num_samples

    def __getitem__(self, index: int) -> TensorDict:
        """Generate a collection of correlated Gaussian vectors.

        Parameters
        ----------
        index : int
            Sample index. Values outside the dataset bounds are rejected; the
            sample itself is generated lazily and independently of the index.

        Returns
        -------
        TensorDict
            Unbatched TensorDict containing ``x`` with shape ``(count, dim)``.

        Raises
        ------
        IndexError
            If ``index`` is outside the dataset bounds.
        """
        if not -self.num_samples <= index < self.num_samples:
            raise IndexError("dataset index out of range")

        latent = torch.randn(self.count, self.dim, dtype=self.factor.dtype)
        return TensorDict({"x": self.factor @ latent}, batch_size=[])


class ConditionedPairwiseCorrelatedGaussian(Dataset[TensorDict]):
    """Mixture of two pairwise Gaussian regimes with noisy context vectors.

    Each item randomly selects one of two
    :class:`PairwiseCorrelatedGaussian` distributions. Its conditioning vector
    is sampled from an isotropic Gaussian whose mean identifies the selected
    regime. The two context distributions share a covariance, allowing their
    degree of overlap to be controlled independently of the MI matrices.

    Parameters
    ----------
    mutual_information : Sequence[Tensor or Sequence[Sequence[float]]]
        Exactly two pairwise mutual-information matrices.
    dim : int, default=1
        Feature dimension of each Gaussian variable.
    cond_dim : int, optional
        Conditioning-vector dimension. Defaults to ``dim``.
    context_separation : float, default=1.0
        Distance between the two isotropic-Gaussian context means.
    context_std : float, default=1.0
        Shared standard deviation of the context distributions.
    num_samples : int, default=10000
        Number of samples exposed by the dataset.
    """

    def __init__(
        self,
        mutual_information: Sequence[
            Tensor | Sequence[Sequence[float]]
        ],
        dim: int = 1,
        cond_dim: int | None = None,
        context_separation: float = 1.0,
        context_std: float = 1.0,
        num_samples: int = 10_000,
    ) -> None:
        if len(mutual_information) != 2:
            raise ValueError("mutual_information must contain two matrices")
        if cond_dim is None:
            cond_dim = dim
        if cond_dim <= 0:
            raise ValueError("cond_dim must be positive")
        if context_separation < 0:
            raise ValueError("context_separation must be nonnegative")
        if context_std <= 0:
            raise ValueError("context_std must be positive")

        self.components = tuple(
            PairwiseCorrelatedGaussian(matrix, dim, num_samples)
            for matrix in mutual_information
        )
        if self.components[0].count != self.components[1].count:
            raise ValueError("both MI matrices must have the same size")
        self.mutual_information = torch.stack(
            [component.mutual_information for component in self.components]
        )
        self.dim = dim
        self.cond_dim = cond_dim
        self.count = self.components[0].count
        self.context_separation = context_separation
        self.context_std = context_std
        self.num_samples = num_samples

    def __len__(self) -> int:
        """Return the number of mixture samples."""
        return self.num_samples

    def __getitem__(self, index: int) -> TensorDict:
        """Draw one Gaussian regime and its noisy conditioning vector."""
        if not -self.num_samples <= index < self.num_samples:
            raise IndexError("dataset index out of range")
        regime = int(torch.randint(2, ()).item())
        x = self.components[regime][index]["x"]
        direction = 2 * regime - 1
        context_mean = direction * self.context_separation / 2
        cond = context_mean + self.context_std * torch.randn(self.cond_dim)
        return TensorDict(
            {
                "x": x,
                "cond": cond,
                "regime": torch.tensor(regime, dtype=torch.long),
            },
            batch_size=[],
        )


class LatentPairwiseCorrelatedGaussian(Dataset[TensorDict]):
    r"""Batched pairwise Gaussian data with covariance-preserving contexts.

    Every item creates ``count`` independent standard-Gaussian context vectors
    and selects ``batch_size`` of them without replacement. Context covariance
    is subtracted from the residual covariance before sampling, so introducing
    sample identity does not change the requested marginal pairwise MI.

    The returned variables follow

    .. math::

        X_{b,i} = \sqrt{\alpha} Z_b + E_{b,i},

    where :math:`E_b` has variable covariance
    :math:`R - \alpha\mathbf{1}\mathbf{1}^\mathsf{T}` and ``R`` is determined
    by the requested pairwise mutual-information matrix. Consequently, the
    marginal covariance of :math:`X_b` remains exactly ``R``. Different batch
    members use independent contexts and residuals.

    Each dataset item is a complete batch and should therefore be consumed
    directly or through a ``DataLoader`` with ``batch_size=None``.

    Parameters
    ----------
    count : int
        Number of latent Gaussian distributions in the conditioning pool.
    batch_size : int
        Number of distinct pool members selected for each generated batch.
    mutual_information : Tensor or Sequence[Sequence[float]]
        Symmetric matrix of target conditional pairwise MI values in nats.
    dim : int, default=1
        Feature dimension of every context and Gaussian variable.
    num_batches : int, default=10000
        Number of lazily generated batches exposed by the dataset.
    context_fraction : float, default=0.5
        Fraction of the maximum valid shared-context covariance to allocate to
        :math:`Z`. Must lie between zero and one, inclusive.

    Attributes
    ----------
    context_strength : float
        Shared covariance :math:`\alpha` allocated to sample context.
    residual_correlation : Tensor
        Residual covariance before adding sample context.
    residual_factor : Tensor
        Factor of ``residual_correlation`` used to sample residuals.
    correlation : Tensor
        Conditional correlation matrix implied by ``mutual_information``.
    factor : Tensor
        Factor satisfying ``factor @ factor.T == correlation``.
    """

    def __init__(
        self,
        count: int,
        batch_size: int,
        mutual_information: Tensor | Sequence[Sequence[float]],
        dim: int = 1,
        num_batches: int = 10_000,
        context_fraction: float = 0.5,
    ) -> None:
        if count <= 0:
            raise ValueError("count must be positive")
        if batch_size < 2:
            raise ValueError("batch_size must be at least two")
        if batch_size > count:
            raise ValueError("batch_size cannot exceed count")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_batches < 0:
            raise ValueError("num_batches must be nonnegative")
        if not 0.0 <= context_fraction <= 1.0:
            raise ValueError("context_fraction must be between zero and one")

        mi, correlation, factor = _pairwise_gaussian_parameters(
            mutual_information,
            dim,
        )
        self.count = count
        self.batch_size = batch_size
        self.dim = dim
        self.num_batches = num_batches
        self.variable_count = mi.shape[0]
        self.mutual_information = mi
        self.correlation = correlation
        self.factor = factor
        self.context_fraction = context_fraction
        maximum_context_strength = self._maximum_context_strength(correlation)
        self.maximum_context_strength = maximum_context_strength
        self.context_strength = context_fraction * maximum_context_strength
        context_covariance = self.context_strength * torch.ones_like(correlation)
        residual_correlation = correlation - context_covariance
        residual_eigenvalues, residual_eigenvectors = torch.linalg.eigh(
            residual_correlation.to(torch.float64)
        )
        tolerance = 1e-8 * max(
            1.0,
            residual_eigenvalues.abs().max().item(),
        )
        if residual_eigenvalues.min().item() < -tolerance:
            raise ValueError("context_fraction produces a non-PSD residual")
        residual_eigenvalues = residual_eigenvalues.clamp_min(0)
        residual_factor = residual_eigenvectors @ torch.diag_embed(
            torch.sqrt(residual_eigenvalues)
        )
        self.residual_correlation = residual_correlation
        self.residual_factor = residual_factor.to(factor.dtype)

        residual_variance = 1.0 - self.context_strength
        conditional_correlation = residual_correlation / residual_variance
        conditional_mi = -dim / 2 * torch.log1p(
            -(conditional_correlation**2)
        )
        conditional_mi.fill_diagonal_(0)
        self.conditional_mutual_information = conditional_mi

    def __len__(self) -> int:
        """Return the number of batches exposed by the dataset."""
        return self.num_batches

    def __getitem__(self, index: int) -> TensorDict:
        """Generate a conditionally pairwise-correlated Gaussian batch.

        Parameters
        ----------
        index : int
            Batch index. Samples are generated lazily and independently of the
            index after bounds validation.

        Returns
        -------
        TensorDict
            Batched TensorDict containing ``x`` with shape
            ``(batch_size, variable_count, dim)``, ``z`` with shape
            ``(batch_size, dim)``, and distinct ``component_index`` values.

        Raises
        ------
        IndexError
            If ``index`` is outside the dataset bounds.
        """
        if not -self.num_batches <= index < self.num_batches:
            raise IndexError("dataset index out of range")

        context_pool = torch.randn(
            self.count,
            self.dim,
            dtype=self.factor.dtype,
        )
        component_index = torch.randperm(self.count)[: self.batch_size]
        z = context_pool[component_index]
        epsilon = torch.randn(
            self.batch_size,
            self.variable_count,
            self.dim,
            dtype=self.factor.dtype,
        )
        residual = torch.einsum(
            "ij,bjd->bid",
            self.residual_factor,
            epsilon,
        )
        x = math.sqrt(self.context_strength) * z.unsqueeze(1) + residual
        return TensorDict(
            {
                "x": x,
                "z": z,
                "component_index": component_index,
            },
            batch_size=[self.batch_size],
        )

    @staticmethod
    def _maximum_context_strength(correlation: Tensor) -> float:
        r"""Return the largest alpha for which ``R - alpha 11^T`` is PSD."""
        matrix = correlation.to(torch.float64)
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        tolerance = 1e-10 * max(1.0, eigenvalues.abs().max().item())
        positive = eigenvalues > tolerance
        ones = torch.ones(matrix.shape[0], dtype=matrix.dtype)
        coordinates = eigenvectors.transpose(0, 1) @ ones
        if (~positive).any() and coordinates[~positive].norm().item() > tolerance:
            return 0.0
        denominator = ((coordinates[positive] ** 2) / eigenvalues[positive]).sum()
        return float(denominator.reciprocal().item())


class MixtureLatentPairwiseCorrelatedGaussian(Dataset[TensorDict]):
    """Two latent Gaussian regimes with correlated conditioning bags.

    Every dataset item randomly selects one of two
    :class:`LatentPairwiseCorrelatedGaussian` components. For each observation,
    the component's latent ``z`` is shared across a bag of stochastic context
    tokens. The regime adds an opposing mean offset, while small independent
    token noise prevents the bag from degenerating into repeated values.

    Parameters
    ----------
    count : int
        Number of latent components available to each regime.
    batch_size : int
        Number of observations in each generated batch.
    mutual_information : Sequence[Tensor or Sequence[Sequence[float]]]
        Exactly two pairwise MI matrices.
    dim : int, default=1
        Feature dimension of observations and conditioning tokens.
    context_count : int, default=8
        Number of stochastic latent tokens per observation.
    regime_separation : float, default=4.0
        Euclidean separation between the two context-distribution means.
    context_noise_std : float, default=0.1
        Independent token-noise standard deviation.
    context_keep_probability : float, default=0.8
        Probability that each context token is marked valid. At least one token
        is always retained per observation.
    num_batches : int, default=10000
        Number of complete batches exposed by the dataset.
    context_fraction : float, default=0.05
        Shared latent covariance fraction used by both Gaussian components.
    """

    def __init__(
        self,
        count: int,
        batch_size: int,
        mutual_information: Sequence[
            Tensor | Sequence[Sequence[float]]
        ],
        dim: int = 1,
        context_count: int = 8,
        regime_separation: float = 4.0,
        context_noise_std: float = 0.1,
        context_keep_probability: float = 0.8,
        num_batches: int = 10_000,
        context_fraction: float = 0.05,
    ) -> None:
        if len(mutual_information) != 2:
            raise ValueError("mutual_information must contain two matrices")
        if context_count <= 0:
            raise ValueError("context_count must be positive")
        if regime_separation < 0:
            raise ValueError("regime_separation must be nonnegative")
        if context_noise_std < 0:
            raise ValueError("context_noise_std must be nonnegative")
        if not 0 < context_keep_probability <= 1:
            raise ValueError(
                "context_keep_probability must be in the interval (0, 1]"
            )

        self.components = tuple(
            LatentPairwiseCorrelatedGaussian(
                count=count,
                batch_size=batch_size,
                mutual_information=matrix,
                dim=dim,
                num_batches=num_batches,
                context_fraction=context_fraction,
            )
            for matrix in mutual_information
        )
        if self.components[0].variable_count != self.components[1].variable_count:
            raise ValueError("both MI matrices must have the same size")
        self.mutual_information = torch.stack(
            [component.mutual_information for component in self.components]
        )
        self.conditional_mutual_information = torch.stack(
            [
                component.conditional_mutual_information
                for component in self.components
            ]
        )
        self.count = count
        self.batch_size = batch_size
        self.dim = dim
        self.context_count = context_count
        self.variable_count = self.components[0].variable_count
        self.regime_separation = regime_separation
        self.context_noise_std = context_noise_std
        self.context_keep_probability = context_keep_probability
        self.num_batches = num_batches
        self.context_fraction = context_fraction

    def __len__(self) -> int:
        """Return the number of complete mixed-regime batches."""
        return self.num_batches

    def __getitem__(self, index: int) -> TensorDict:
        """Draw one regime and create its correlated latent-token bags."""
        if not -self.num_batches <= index < self.num_batches:
            raise IndexError("dataset index out of range")
        regime = int(torch.randint(2, ()).item())
        component_batch = self.components[regime][index]
        z = component_batch["z"]
        direction = 2 * regime - 1
        coordinate_offset = self.regime_separation / (2 * math.sqrt(self.dim))
        context_mean = direction * coordinate_offset
        noise = self.context_noise_std * torch.randn(
            self.batch_size,
            self.context_count,
            self.dim,
            dtype=z.dtype,
        )
        context = z.unsqueeze(1) + context_mean + noise
        context_mask = torch.rand(
            self.batch_size,
            self.context_count,
        ) < self.context_keep_probability
        missing = ~context_mask.any(dim=-1)
        context_mask[missing, 0] = True
        return TensorDict(
            {
                "x": component_batch["x"],
                "context": context,
                "context_mask": context_mask,
                "regime": torch.full(
                    (self.batch_size,),
                    regime,
                    dtype=torch.long,
                ),
            },
            batch_size=[self.batch_size],
        )


__all__ = [
    "ConditionedPairwiseCorrelatedGaussian",
    "CorrelatedGausian",
    "LatentPairwiseCorrelatedGaussian",
    "MixtureLatentPairwiseCorrelatedGaussian",
    "PairwiseCorrelatedGaussian",
    "tensordict_collate",
    "tensordict_passthrough",
]

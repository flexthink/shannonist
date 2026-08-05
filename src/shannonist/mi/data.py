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

        self.mutual_information = mi.to(torch.get_default_dtype())
        self.dim = dim
        self.num_samples = num_samples
        self.count = mi.shape[0]
        self.correlation = correlation.to(torch.get_default_dtype())
        self.factor = factor.to(torch.get_default_dtype())

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


__all__ = [
    "CorrelatedGausian",
    "PairwiseCorrelatedGaussian",
    "tensordict_collate",
]

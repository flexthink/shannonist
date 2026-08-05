import math

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CorrelatedGausian(Dataset[dict[str, Tensor]]):
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

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Generate a correlated Gaussian pair.

        Parameters
        ----------
        index : int
            Sample index. Values outside the dataset bounds are rejected; the
            sample itself is generated lazily and independently of the index.

        Returns
        -------
        dict[str, Tensor]
            Dictionary containing ``x`` and ``y``, each with shape ``(dim,)``.

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
        return {"x": x, "y": y}


__all__ = ["CorrelatedGausian"]

from tensordict import TensorClass, TensorDict
from torch import Tensor


class MIBatch(TensorClass):
    """Paired observations used to estimate mutual information.

    Sequence-valued observations may use shapes ``(batch, length, features)``.
    Their optional masks then have shapes ``(batch, length, mask)`` and use
    nonzero values to identify valid positions.

    Parameters
    ----------
    x : Tensor
        Observations of the first random variable.
    y : Tensor
        Corresponding observations of the second random variable.
    x_mask : Tensor, optional
        Length mask for ``x``. Required only when the estimator needs to
        distinguish valid and padded positions.
    y_mask : Tensor, optional
        Length mask for ``y``. Required only when the estimator needs to
        distinguish valid and padded positions.
    """

    x: Tensor
    y: Tensor
    x_mask: Tensor | None = None
    y_mask: Tensor | None = None


class MIEstimate(TensorClass):
    """Mutual-information estimate and optional diagnostic values.

    Parameters
    ----------
    value : Tensor
        Estimated mutual information.
    details : TensorDict, optional
        Estimator-specific diagnostic values.
    """

    value: Tensor
    details: TensorDict | None = None


class PairwiseMIBatch(TensorClass):
    """Observations used to estimate pairwise mutual information.

    Parameters
    ----------
    x : Tensor
        Observations with shape ``(batch, count, features)``.
    """

    x: Tensor


__all__ = ["MIBatch", "MIEstimate", "PairwiseMIBatch"]

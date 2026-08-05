from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tensordict import TensorClass, TensorDict
from torch import Tensor


InputT = TypeVar("InputT", bound=TensorClass)
PredictionT = TypeVar("PredictionT", bound=TensorClass)
EstimateT = TypeVar("EstimateT", bound=TensorClass)


class ObjectiveOutput(TensorClass):
    """Output of a differentiable estimator objective.

    Parameters
    ----------
    loss : Tensor
        Scalar loss to minimize.
    metrics : TensorDict, optional
        Detached or diagnostic values associated with the objective.
    """

    loss: Tensor
    metrics: TensorDict | None = None


class Estimator(ABC, Generic[InputT, EstimateT]):
    """Interface for estimating a quantity from a batch."""

    @abstractmethod
    def estimate(self, batch: InputT) -> EstimateT:
        """Compute an estimate from a batch.

        Parameters
        ----------
        batch : InputT
            Typed input batch expected by the estimator.

        Returns
        -------
        EstimateT
            Typed estimate produced from the batch.
        """
        ...


class TrainableEstimator(
    Estimator[InputT, EstimateT],
    Generic[InputT, PredictionT, EstimateT],
):
    """Estimator separating model execution from objective computation."""

    @abstractmethod
    def compute_forward(self, batch: InputT) -> PredictionT:
        """Compute model predictions and auxiliary outputs.

        Parameters
        ----------
        batch : InputT
            Typed input batch expected by the estimator.

        Returns
        -------
        PredictionT
            Typed model predictions used to construct the objective.
        """
        ...

    @abstractmethod
    def compute_objectives(self, predictions: PredictionT) -> ObjectiveOutput:
        """Construct an objective from precomputed model predictions.

        This method computes a loss but does not run backpropagation, update
        parameters, execute model modules, or otherwise manage a training loop.

        Parameters
        ----------
        predictions : PredictionT
            Output previously returned by :meth:`compute_forward`.

        Returns
        -------
        ObjectiveOutput
            Loss and optional metrics for external optimization.
        """
        ...


__all__ = [
    "EstimateT",
    "Estimator",
    "InputT",
    "ObjectiveOutput",
    "PredictionT",
    "TrainableEstimator",
]

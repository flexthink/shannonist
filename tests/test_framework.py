import inspect

import pytest
import torch
from typing import Any

from tensordict import TensorClass

from shannonist.framework import Estimator, ObjectiveOutput, TrainableEstimator


class Input(TensorClass):
    value: torch.Tensor


class Prediction(TensorClass):
    value: torch.Tensor


class Estimate(TensorClass):
    value: torch.Tensor


class ConcreteEstimator(TrainableEstimator[Input, Prediction, Estimate]):
    """Minimal implementation of the trainable estimator contract."""

    def compute_forward(self, batch: Input) -> Prediction:
        return Prediction(value=batch.value * 2, batch_size=batch.batch_size)

    def compute_objectives(self, predictions: Prediction) -> ObjectiveOutput:
        mean = predictions.value.mean()
        return ObjectiveOutput(loss=mean, estimate=mean, batch_size=[])

    def estimate(
        self,
        batch: Input,
        options: dict[str, Any] | None = None,
    ) -> Estimate:
        del options
        predictions = self.compute_forward(batch)
        return Estimate(value=predictions.value.mean(), batch_size=[])


def test_estimator_interfaces_are_abstract() -> None:
    assert inspect.isabstract(Estimator)
    assert inspect.isabstract(TrainableEstimator)
    with pytest.raises(TypeError):
        Estimator()


def test_trainable_estimator_separates_forward_and_objective() -> None:
    estimator = ConcreteEstimator()
    batch = Input(value=torch.arange(4, dtype=torch.float32), batch_size=[4])

    predictions = estimator.compute_forward(batch)
    objective = estimator.compute_objectives(predictions)
    estimate = estimator.estimate(batch)

    assert torch.equal(predictions.value, batch.value * 2)
    assert objective.loss.item() == pytest.approx(3.0)
    assert objective.estimate.item() == pytest.approx(3.0)
    assert estimate.value.item() == pytest.approx(3.0)

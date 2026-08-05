from collections.abc import Iterable

import pytest
import torch
from torch import Tensor, nn
from torch.optim import SGD, Optimizer
from torch.utils.data import TensorDataset

from shannonist.core import Brain, RunOpts, Stage, parse_arguments


class RegressionBrain(Brain[list[Tensor], Tensor]):
    """Small concrete brain used to exercise the generic data loop."""

    def compute_forward(self, batch: list[Tensor], stage: Stage) -> Tensor:
        del stage
        return self.modules["model"](batch[0])

    def compute_objectives(
        self,
        predictions: Tensor,
        batch: list[Tensor],
        stage: Stage,
    ) -> Tensor:
        del stage
        return torch.nn.functional.mse_loss(predictions, batch[1])


def optimizer_factory(parameters: Iterable[nn.Parameter]) -> Optimizer:
    """Construct the optimizer used by core-loop tests."""
    return SGD(parameters, lr=0.1)


def test_parse_arguments_separates_runtime_options_and_overrides() -> None:
    param_file, run_opts, overrides = parse_arguments(
        [
            "hparams.yaml",
            "--device",
            "cuda:1",
            "--epochs=3",
            "--batch_size",
            "16",
        ]
    )

    assert param_file == "hparams.yaml"
    assert run_opts == RunOpts(device="cuda:1")
    assert overrides == "epochs: 3\nbatch_size: 16"


@pytest.mark.parametrize(
    "arguments",
    [
        ["hparams.yaml", "orphan"],
        ["hparams.yaml", "--epochs"],
    ],
)
def test_parse_arguments_rejects_malformed_overrides(arguments: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_arguments(arguments)


def test_brain_fit_updates_parameters_and_evaluate_returns_loss() -> None:
    model = nn.Linear(1, 1)
    brain = RegressionBrain(
        modules={"model": model},
        opt_class=optimizer_factory,
        hparams={"label": "test"},
    )
    dataset = TensorDataset(
        torch.arange(8, dtype=torch.float32).unsqueeze(1),
        (2 * torch.arange(8, dtype=torch.float32)).unsqueeze(1),
    )
    weight_before = model.weight.detach().clone()

    brain.fit(2, dataset, train_loader_kwargs={"batch_size": 4})
    test_loss = brain.evaluate(dataset, {"batch_size": 4})

    assert not torch.equal(model.weight, weight_before)
    assert isinstance(test_loss, float)
    assert test_loss >= 0
    assert brain.hparams.label == "test"


def test_brain_rejects_empty_data_loader() -> None:
    brain = RegressionBrain(
        modules={"model": nn.Linear(1, 1)},
        opt_class=optimizer_factory,
    )
    empty = TensorDataset(torch.empty(0, 1), torch.empty(0, 1))

    with pytest.raises(ValueError, match="test data loader is empty"):
        brain.evaluate(empty, {"batch_size": 1})

import argparse
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from types import SimpleNamespace
from typing import Any, Generic, TypeVar

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


BatchT = TypeVar("BatchT")
ForwardT = TypeVar("ForwardT")


class Stage(Enum):
    """Stage of an experiment data loop."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()


@dataclass(frozen=True)
class RunOpts:
    """Runtime controls for an experiment.

    Parameters
    ----------
    device : str, default="cpu"
        Torch device on which modules and batches are placed.
    """

    device: str = "cpu"


def parse_arguments(
    arg_list: Sequence[str] | None = None,
) -> tuple[str, RunOpts, str]:
    """Parse a recipe parameter file, runtime options, and YAML overrides.

    Unknown ``--key=value`` or ``--key value`` arguments are converted to a
    YAML string suitable for :func:`hyperpyyaml.load_hyperpyyaml`.

    Parameters
    ----------
    arg_list : Sequence[str], optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    tuple[str, RunOpts, str]
        Hyperparameter filename, runtime options, and HyperPyYAML overrides.
    """
    parser = argparse.ArgumentParser(description="Run a Shannonist recipe")
    parser.add_argument("param_file", help="HyperPyYAML parameter file")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used for the experiment, such as cpu or cuda:0",
    )
    parsed, unknown = parser.parse_known_args(arg_list)
    overrides = _overrides_to_yaml(unknown)
    return parsed.param_file, RunOpts(device=parsed.device), overrides


def _overrides_to_yaml(arguments: Sequence[str]) -> str:
    """Convert unknown command-line options into a YAML mapping."""
    lines: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if not argument.startswith("--"):
            raise ValueError(f"unexpected override value: {argument}")

        option = argument[2:]
        if "=" in option:
            key, value = option.split("=", 1)
        else:
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise ValueError(f"override --{option} requires a value")
            key = option
            value = arguments[index + 1]
            index += 1

        lines.append(f"{key}: {value}")
        index += 1

    return "\n".join(lines)


class Brain(ABC, Generic[BatchT, ForwardT]):
    """Minimal SpeechBrain-style wrapper around PyTorch training loops.

    Subclasses implement :meth:`compute_forward` and
    :meth:`compute_objectives`. The base class manages module modes, device
    placement, optimization, epoch iteration, and validation.

    Parameters
    ----------
    modules : Mapping[str, nn.Module]
        Named modules trained and evaluated by the brain.
    opt_class : callable
        Optimizer factory accepting an iterable of parameters.
    hparams : Mapping[str, Any], optional
        Recipe hyperparameters, also exposed through attribute access.
    run_opts : RunOpts, optional
        Runtime controls. Defaults to CPU execution.
    """

    def __init__(
        self,
        modules: Mapping[str, nn.Module],
        opt_class: Callable[[Iterable[nn.Parameter]], Optimizer],
        hparams: Mapping[str, Any] | None = None,
        run_opts: RunOpts | None = None,
    ) -> None:
        self.run_opts = run_opts if run_opts is not None else RunOpts()
        self.device = torch.device(self.run_opts.device)
        self.modules = nn.ModuleDict(modules).to(self.device)
        self.hparams = SimpleNamespace(**dict(hparams or {}))
        self.optimizer = opt_class(self.modules.parameters())

    @abstractmethod
    def compute_forward(self, batch: BatchT, stage: Stage) -> ForwardT:
        """Compute model outputs for a batch.

        Parameters
        ----------
        batch : BatchT
            Batch from the current data loader.
        stage : Stage
            Current experiment stage.

        Returns
        -------
        ForwardT
            Model output consumed by :meth:`compute_objectives`.
        """
        ...

    @abstractmethod
    def compute_objectives(
        self,
        predictions: ForwardT,
        batch: BatchT,
        stage: Stage,
    ) -> Tensor:
        """Compute the scalar loss minimized by the loop.

        Parameters
        ----------
        predictions : ForwardT
            Output returned by :meth:`compute_forward`.
        batch : BatchT
            Batch used to produce ``predictions``.
        stage : Stage
            Current experiment stage.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        ...

    def fit(
        self,
        epoch_counter: int | Iterable[int],
        train_set: Dataset[Any] | DataLoader[Any],
        valid_set: Dataset[Any] | DataLoader[Any] | None = None,
        train_loader_kwargs: Mapping[str, Any] | None = None,
        valid_loader_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Train modules for multiple epochs and optionally validate.

        Parameters
        ----------
        epoch_counter : int or Iterable[int]
            Number of epochs or explicit epoch identifiers.
        train_set : Dataset or DataLoader
            Training data.
        valid_set : Dataset or DataLoader, optional
            Validation data.
        train_loader_kwargs : Mapping[str, Any], optional
            Arguments used when constructing the training data loader.
        valid_loader_kwargs : Mapping[str, Any], optional
            Arguments used when constructing the validation data loader.
        """
        train_loader = self.make_dataloader(train_set, train_loader_kwargs)
        valid_loader = (
            self.make_dataloader(valid_set, valid_loader_kwargs)
            if valid_set is not None
            else None
        )
        epochs = (
            range(1, epoch_counter + 1)
            if isinstance(epoch_counter, int)
            else epoch_counter
        )

        for epoch in epochs:
            self.on_stage_start(Stage.TRAIN, epoch)
            train_loss = self._run_stage(train_loader, Stage.TRAIN, epoch)
            self.on_stage_end(Stage.TRAIN, train_loss, epoch)

            if valid_loader is not None:
                self.on_stage_start(Stage.VALID, epoch)
                valid_loss = self._run_stage(valid_loader, Stage.VALID, epoch)
                self.on_stage_end(Stage.VALID, valid_loss, epoch)

    def evaluate(
        self,
        test_set: Dataset[Any] | DataLoader[Any],
        test_loader_kwargs: Mapping[str, Any] | None = None,
    ) -> float:
        """Evaluate modules on a test dataset.

        Parameters
        ----------
        test_set : Dataset or DataLoader
            Test data.
        test_loader_kwargs : Mapping[str, Any], optional
            Arguments used when constructing the test data loader.

        Returns
        -------
        float
            Mean test loss.
        """
        loader = self.make_dataloader(test_set, test_loader_kwargs)
        self.on_stage_start(Stage.TEST, None)
        loss = self._run_stage(loader, Stage.TEST, None)
        self.on_stage_end(Stage.TEST, loss, None)
        return loss

    def make_dataloader(
        self,
        dataset: Dataset[Any] | DataLoader[Any],
        loader_kwargs: Mapping[str, Any] | None = None,
    ) -> DataLoader[Any]:
        """Return a data loader for a dataset or pass one through unchanged."""
        if isinstance(dataset, DataLoader):
            return dataset
        return DataLoader(dataset, **dict(loader_kwargs or {}))

    def fit_batch(self, batch: BatchT) -> Tensor:
        """Optimize modules on one batch and return its detached loss."""
        batch = _move_to_device(batch, self.device)
        self.optimizer.zero_grad()
        predictions = self.compute_forward(batch, Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def evaluate_batch(self, batch: BatchT, stage: Stage) -> Tensor:
        """Evaluate one batch without recording gradients."""
        batch = _move_to_device(batch, self.device)
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage)
            loss = self.compute_objectives(predictions, batch, stage)
        return loss.detach()

    def on_stage_start(self, stage: Stage, epoch: int | None) -> None:
        """Run a hook before a data stage begins."""

    def on_stage_end(
        self,
        stage: Stage,
        stage_loss: float,
        epoch: int | None,
    ) -> None:
        """Report the mean loss when a data stage ends."""
        epoch_label = f" epoch={epoch}" if epoch is not None else ""
        print(f"stage={stage.name.lower()}{epoch_label} loss={stage_loss:.6f}")

    def _run_stage(
        self,
        loader: DataLoader[Any],
        stage: Stage,
        epoch: int | None,
    ) -> float:
        """Run one stage with a live batch-level loss display."""
        is_training = stage is Stage.TRAIN
        self.modules.train(is_training)
        total = 0.0
        count = 0
        epoch_label = f" epoch {epoch}" if epoch is not None else ""
        batches = tqdm(
            loader,
            desc=f"{stage.name.lower()}{epoch_label}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )
        for batch in batches:
            loss = (
                self.fit_batch(batch)
                if is_training
                else self.evaluate_batch(batch, stage)
            )
            total += loss.item()
            count += 1
            batches.set_postfix(loss=f"{total / count:.6f}")

        if count == 0:
            raise ValueError(f"{stage.name.lower()} data loader is empty")
        return total / count


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors in a batch to a device."""
    if isinstance(value, Tensor):
        return value.to(device)
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, Mapping):
        return type(value)(
            (key, _move_to_device(item, device)) for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


__all__ = ["Brain", "RunOpts", "Stage", "parse_arguments"]

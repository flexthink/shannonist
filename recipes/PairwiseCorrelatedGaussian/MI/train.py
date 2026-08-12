"""Train a pairwise MI estimator on correlated Gaussian observations."""

from collections.abc import Sequence

import torch
from hyperpyyaml import load_hyperpyyaml
from tensordict import TensorDict
from torch import Tensor

from shannonist.core import Brain, RunOpts, Stage, parse_arguments
from shannonist.mi import (
    PairwiseBA,
    PairwiseBAOutput,
    PairwiseFLO,
    PairwiseFLOOutput,
    PairwiseMIBatch,
)


PairwiseMIPrediction = PairwiseFLOOutput | PairwiseBAOutput
PairwiseMIEstimator = PairwiseFLO | PairwiseBA


class PairwiseMIBrain(Brain[TensorDict, PairwiseMIPrediction]):
    """SpeechBrain-style training loop for pairwise MI estimation."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._stage_estimates: list[Tensor] = []

    def compute_forward(
        self,
        batch: TensorDict,
        stage: Stage,
    ) -> PairwiseMIPrediction:
        """Construct a pairwise MI batch and run the estimator.

        Parameters
        ----------
        batch : TensorDict
            TensorDict containing ``x`` with shape
            ``(batch, count, features)``.
        stage : Stage
            Current experiment stage.

        Returns
        -------
        PairwiseMIPrediction
            Estimator-specific predictions and auxiliary outputs.
        """
        del stage
        mi_batch = PairwiseMIBatch(
            x=batch["x"],
            batch_size=batch.batch_size,
        )
        estimator = self._estimator()
        return estimator.compute_forward(mi_batch)

    def compute_objectives(
        self,
        predictions: PairwiseMIPrediction,
        batch: TensorDict,
        stage: Stage,
    ) -> Tensor:
        """Compute the joint objective and retain its pairwise lower bounds.

        Parameters
        ----------
        predictions : PairwiseMIPrediction
            Output produced by :meth:`compute_forward`.
        batch : TensorDict
            Input batch. It is unused because the required tensors are in
            ``predictions``.
        stage : Stage
            Current experiment stage.

        Returns
        -------
        Tensor
            Mean estimator loss over unique pairs.
        """
        del batch, stage
        objective = self._estimator().compute_objectives(predictions)
        self._stage_estimates.append(objective.estimate.detach().cpu())
        return objective.loss

    def on_stage_start(self, stage: Stage, epoch: int | None) -> None:
        """Reset pairwise estimates before a stage begins."""
        del stage, epoch
        self._stage_estimates = []

    def on_stage_end(
        self,
        stage: Stage,
        stage_loss: float,
        epoch: int | None,
    ) -> None:
        """Print learned and ground-truth MI matrices side by side."""
        learned = torch.stack(self._stage_estimates).mean(dim=0)
        ground_truth = torch.as_tensor(
            self.hparams.mutual_information,
            dtype=learned.dtype,
        )
        epoch_label = f" epoch={epoch}" if epoch is not None else ""
        print(f"stage={stage.name.lower()}{epoch_label} loss={stage_loss:.6f}")
        print("learned MI lower bound       | ground truth MI")
        for learned_row, truth_row in zip(learned, ground_truth):
            learned_text = _format_matrix_row(learned_row)
            truth_text = _format_matrix_row(truth_row)
            print(f"{learned_text} | {truth_text}")

    def _estimator(self) -> PairwiseMIEstimator:
        """Return the configured estimator with a checked type."""
        estimator = self.modules["estimator"]
        if not isinstance(estimator, (PairwiseFLO, PairwiseBA)):
            raise TypeError(
                "the estimator module must be a PairwiseFLO or PairwiseBA"
            )
        return estimator


def _format_matrix_row(row: Tensor) -> str:
    """Format one matrix row with stable column widths."""
    return "[" + " ".join(f"{value.item():8.4f}" for value in row) + "]"


def main(arg_list: Sequence[str] | None = None) -> None:
    """Load recipe hyperparameters and run pairwise MI training.

    Parameters
    ----------
    arg_list : Sequence[str], optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    param_file, run_opts, overrides = parse_arguments(arg_list)
    with open(param_file, encoding="utf-8") as yaml_file:
        hparams = load_hyperpyyaml(yaml_file, overrides)

    torch.manual_seed(hparams["seed"])
    brain = PairwiseMIBrain(
        modules={"estimator": hparams["estimator"]},
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=RunOpts(device=run_opts.device),
    )
    brain.fit(
        epoch_counter=hparams["number_of_epochs"],
        train_set=hparams["train_set"],
        valid_set=hparams["valid_set"],
        train_loader_kwargs=hparams["train_loader_kwargs"],
        valid_loader_kwargs=hparams["valid_loader_kwargs"],
    )


if __name__ == "__main__":
    main()

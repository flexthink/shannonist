"""Train conditional BA on a mixture of correlated Gaussian regimes."""

from collections.abc import Sequence

import torch
from hyperpyyaml import load_hyperpyyaml
from tensordict import TensorDict
from torch import Tensor

from shannonist.core import Brain, RunOpts, Stage, parse_arguments
from shannonist.mi import PairwiseBA, PairwiseBAOutput, PairwiseMIBatch


class ConditionalMIBrain(Brain[TensorDict, PairwiseBAOutput]):
    """Train and report conditional pairwise BA bounds by regime."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._stage_estimates: list[list[Tensor]] = [[], []]

    def compute_forward(
        self,
        batch: TensorDict,
        stage: Stage,
    ) -> PairwiseBAOutput:
        """Evaluate conditioned BA on a mixed-regime batch."""
        del stage
        return self._estimator().compute_forward(
            PairwiseMIBatch(
                x=batch["x"],
                cond=batch["cond"],
                batch_size=batch.batch_size,
            )
        )

    def compute_objectives(
        self,
        predictions: PairwiseBAOutput,
        batch: TensorDict,
        stage: Stage,
    ) -> Tensor:
        """Compute the mixed objective and retain per-regime estimates."""
        objective = self._estimator().compute_objectives(predictions)
        regimes = batch["regime"]
        for regime in range(2):
            selected = regimes == regime
            if selected.any():
                regime_objective = self._estimator().compute_objectives(
                    predictions[selected]
                )
                self._stage_estimates[regime].append(
                    regime_objective.estimate.detach().cpu()
                )
        del stage
        return objective.loss

    def on_stage_start(self, stage: Stage, epoch: int | None) -> None:
        """Reset accumulated regime estimates."""
        del stage, epoch
        self._stage_estimates = [[], []]

    def on_stage_end(
        self,
        stage: Stage,
        stage_loss: float,
        epoch: int | None,
    ) -> None:
        """Print learned and true conditional MI matrices for each regime."""
        epoch_label = f" epoch={epoch}" if epoch is not None else ""
        print(f"stage={stage.name.lower()}{epoch_label} loss={stage_loss:.6f}")
        truths = torch.as_tensor(self.hparams.mutual_information)
        for regime, estimates in enumerate(self._stage_estimates):
            if not estimates:
                continue
            learned = torch.stack(estimates).mean(dim=0)
            truth = truths[regime].to(learned)
            print(f"regime={regime} learned conditional MI | ground truth MI")
            for learned_row, truth_row in zip(learned, truth):
                print(
                    f"{_format_matrix_row(learned_row)} | "
                    f"{_format_matrix_row(truth_row)}"
                )

    def _estimator(self) -> PairwiseBA:
        """Return the configured conditional BA estimator."""
        estimator = self.modules["estimator"]
        if not isinstance(estimator, PairwiseBA):
            raise TypeError("the estimator module must be a PairwiseBA")
        return estimator


def _format_matrix_row(row: Tensor) -> str:
    """Format one matrix row with stable column widths."""
    return "[" + " ".join(f"{value.item():8.4f}" for value in row) + "]"


def main(arg_list: Sequence[str] | None = None) -> None:
    """Load hyperparameters and train conditional BA."""
    param_file, run_opts, overrides = parse_arguments(arg_list)
    with open(param_file, encoding="utf-8") as yaml_file:
        hparams = load_hyperpyyaml(yaml_file, overrides)

    torch.manual_seed(hparams["seed"])
    brain = ConditionalMIBrain(
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

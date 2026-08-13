"""Train conditional BA on two latent correlated-Gaussian regimes."""

from collections.abc import Sequence

import torch
from hyperpyyaml import load_hyperpyyaml
from tensordict import TensorDict
from torch import Tensor

from shannonist.core import Brain, RunOpts, Stage, parse_arguments
from shannonist.mi import (
    PairwiseMIBatch,
    SampledPairwiseBA,
    SampledPairwiseBAOutput,
)


class MixtureConditionalMIBrain(
    Brain[TensorDict, SampledPairwiseBAOutput]
):
    """Learn both conditional MI matrices from attention-pooled contexts."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._matrix_sums = torch.empty(0)
        self._matrix_counts = torch.empty(0)
        self._regime_estimates: list[list[Tensor]] = [[], []]

    def compute_forward(
        self,
        batch: TensorDict,
        stage: Stage,
    ) -> SampledPairwiseBAOutput:
        """Pool the latent bag and condition sampled pairwise BA on it."""
        del stage
        return self._estimator().compute_forward(
            PairwiseMIBatch(
                x=batch["x"],
                cond=batch["context"],
                cond_mask=batch["context_mask"],
                batch_size=batch.batch_size,
            )
        )

    def compute_objectives(
        self,
        predictions: SampledPairwiseBAOutput,
        batch: TensorDict,
        stage: Stage,
    ) -> Tensor:
        """Compute loss and accumulate a matrix for the selected regime."""
        del stage
        objective = self._estimator().compute_objectives(predictions)
        metrics = objective.metrics
        assert metrics is not None
        regime_values = batch["regime"].unique()
        if regime_values.numel() != 1:
            raise ValueError("each generated batch must contain one regime")
        regime = int(regime_values.item())
        self._regime_estimates[regime].append(objective.estimate.detach().cpu())

        indices = predictions.position_indices.detach().cpu()
        estimates = metrics["estimate_vec"].detach().cpu().reshape(-1)
        left = indices[..., 0].reshape(-1)
        right = indices[..., 1].reshape(-1)
        regimes = torch.full_like(left, regime)
        ones = torch.ones_like(estimates)
        self._matrix_sums.index_put_(
            (regimes, left, right), estimates, accumulate=True
        )
        self._matrix_sums.index_put_(
            (regimes, right, left), estimates, accumulate=True
        )
        self._matrix_counts.index_put_(
            (regimes, left, right), ones, accumulate=True
        )
        self._matrix_counts.index_put_(
            (regimes, right, left), ones, accumulate=True
        )
        return objective.loss

    def on_stage_start(self, stage: Stage, epoch: int | None) -> None:
        """Reset per-regime matrix accumulators."""
        del stage, epoch
        count = int(self.hparams.sequence_length)
        self._matrix_sums = torch.zeros(2, count, count)
        self._matrix_counts = torch.zeros(2, count, count)
        self._regime_estimates = [[], []]

    def on_stage_end(
        self,
        stage: Stage,
        stage_loss: float,
        epoch: int | None,
    ) -> None:
        """Print both learned conditional MI matrices and their targets."""
        epoch_label = f" epoch={epoch}" if epoch is not None else ""
        print(f"stage={stage.name.lower()}{epoch_label} loss={stage_loss:.6f}")
        targets = self._targets()
        for regime in range(2):
            estimates = self._regime_estimates[regime]
            if not estimates:
                print(f"regime={regime} not sampled")
                continue
            scalar = torch.stack(estimates).mean().item()
            matrix = self._matrix_sums[regime] / self._matrix_counts[regime]
            matrix.fill_diagonal_(0)
            print(
                f"regime={regime} "
                f"learned_conditional_mi_lower_bound={scalar:.6f}"
            )
            print("learned conditional MI       | ground truth conditional MI")
            for learned_row, truth_row in zip(matrix, targets[regime]):
                print(
                    f"{_format_matrix_row(learned_row)} | "
                    f"{_format_matrix_row(truth_row)}"
                )

    def _targets(self) -> Tensor:
        """Return the exact per-regime conditional Gaussian MI matrices."""
        target = getattr(
            self.hparams.train_set,
            "conditional_mutual_information",
            None,
        )
        if not isinstance(target, Tensor):
            raise TypeError(
                "train_set must expose conditional_mutual_information"
            )
        return target

    def _estimator(self) -> SampledPairwiseBA:
        """Return the checked sampled BA estimator."""
        estimator = self.modules["estimator"]
        if not isinstance(estimator, SampledPairwiseBA):
            raise TypeError("estimator module must be SampledPairwiseBA")
        return estimator


def _format_matrix_row(row: Tensor) -> str:
    """Format one matrix row with stable column widths."""
    return "[" + " ".join(f"{value.item():8.4f}" for value in row) + "]"


def main(arg_list: Sequence[str] | None = None) -> None:
    """Load hyperparameters and train the two-regime conditional estimator."""
    param_file, run_opts, overrides = parse_arguments(arg_list)
    with open(param_file, encoding="utf-8") as yaml_file:
        hparams = load_hyperpyyaml(yaml_file, overrides)

    torch.manual_seed(hparams["seed"])
    brain = MixtureConditionalMIBrain(
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

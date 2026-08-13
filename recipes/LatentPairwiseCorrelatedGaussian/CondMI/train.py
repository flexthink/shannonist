"""Train conditional BA on latent pairwise correlated Gaussians."""

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


class ConditionalLatentMIBrain(
    Brain[TensorDict, SampledPairwiseBAOutput]
):
    """Train sampled pairwise BA conditioned on the dataset latent vector."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._stage_estimates: list[Tensor] = []
        self._stage_targets: list[Tensor] = []
        self._stage_matrix_sum = torch.empty(0)
        self._stage_matrix_count = torch.empty(0)

    def compute_forward(
        self,
        batch: TensorDict,
        stage: Stage,
    ) -> SampledPairwiseBAOutput:
        """Run conditioned BA with ``z`` kept separate from pair context."""
        del stage
        return self._estimator().compute_forward(
            PairwiseMIBatch(
                x=batch["x"],
                cond=batch["z"],
                batch_size=batch.batch_size,
            )
        )

    def compute_objectives(
        self,
        predictions: SampledPairwiseBAOutput,
        batch: TensorDict,
        stage: Stage,
    ) -> Tensor:
        """Compute loss and accumulate sampled conditional-MI estimates."""
        del batch, stage
        objective = self._estimator().compute_objectives(predictions)
        metrics = objective.metrics
        assert metrics is not None
        pair_estimates = metrics["estimate_vec"].detach().cpu()
        self._stage_estimates.append(objective.estimate.detach().cpu())

        pair_indices = predictions.position_indices.detach().cpu()
        left = pair_indices[..., 0].reshape(-1)
        right = pair_indices[..., 1].reshape(-1)
        flat_estimates = pair_estimates.reshape(-1)
        ones = torch.ones_like(flat_estimates)
        self._stage_matrix_sum.index_put_(
            (left, right), flat_estimates, accumulate=True
        )
        self._stage_matrix_sum.index_put_(
            (right, left), flat_estimates, accumulate=True
        )
        self._stage_matrix_count.index_put_((left, right), ones, accumulate=True)
        self._stage_matrix_count.index_put_((right, left), ones, accumulate=True)

        target = self._conditional_target().to(
            device=predictions.position_indices.device
        )
        sampled_target = target[
            predictions.position_indices[..., 0],
            predictions.position_indices[..., 1],
        ].mean()
        self._stage_targets.append(sampled_target.detach().cpu())
        return objective.loss

    def on_stage_start(self, stage: Stage, epoch: int | None) -> None:
        """Reset accumulated sampled estimates."""
        del stage, epoch
        self._stage_estimates = []
        self._stage_targets = []
        count = int(self.hparams.sequence_length)
        self._stage_matrix_sum = torch.zeros(count, count)
        self._stage_matrix_count = torch.zeros(count, count)

    def on_stage_end(
        self,
        stage: Stage,
        stage_loss: float,
        epoch: int | None,
    ) -> None:
        """Report sampled and matrix-valued conditional MI bounds."""
        learned = torch.stack(self._stage_estimates).mean().item()
        target = torch.stack(self._stage_targets).mean().item()
        epoch_label = f" epoch={epoch}" if epoch is not None else ""
        print(
            f"stage={stage.name.lower()}{epoch_label} "
            f"loss={stage_loss:.6f} "
            f"learned_conditional_mi_lower_bound={learned:.6f} "
            f"target_conditional_mi={target:.6f}"
        )
        if stage is Stage.VALID:
            learned_matrix = self._stage_matrix_sum / self._stage_matrix_count
            learned_matrix.fill_diagonal_(0)
            ground_truth = self._conditional_target().to(learned_matrix)
            print("learned validation conditional MI | ground truth conditional MI")
            for learned_row, truth_row in zip(learned_matrix, ground_truth):
                print(
                    f"{_format_matrix_row(learned_row)} | "
                    f"{_format_matrix_row(truth_row)}"
                )

    def _conditional_target(self) -> Tensor:
        """Return the exact Gaussian MI matrix conditioned on ``z``."""
        dataset = self.hparams.train_set
        target = getattr(dataset, "conditional_mutual_information", None)
        if not isinstance(target, Tensor):
            raise TypeError(
                "train_set must expose conditional_mutual_information"
            )
        return target

    def _estimator(self) -> SampledPairwiseBA:
        """Return the configured sampled BA estimator."""
        estimator = self.modules["estimator"]
        if not isinstance(estimator, SampledPairwiseBA):
            raise TypeError("the estimator module must be a SampledPairwiseBA")
        return estimator


def _format_matrix_row(row: Tensor) -> str:
    """Format one matrix row with stable column widths."""
    return "[" + " ".join(f"{value.item():8.4f}" for value in row) + "]"


def main(arg_list: Sequence[str] | None = None) -> None:
    """Load hyperparameters and train conditional sampled BA."""
    param_file, run_opts, overrides = parse_arguments(arg_list)
    with open(param_file, encoding="utf-8") as yaml_file:
        hparams = load_hyperpyyaml(yaml_file, overrides)

    torch.manual_seed(hparams["seed"])
    brain = ConditionalLatentMIBrain(
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

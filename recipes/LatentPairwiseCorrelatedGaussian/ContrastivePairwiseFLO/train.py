"""Train contrastive pairwise FLO on latent-conditioned sequences."""

from collections.abc import Sequence

import torch
from hyperpyyaml import load_hyperpyyaml
from tensordict import TensorDict
from torch import Tensor

from shannonist.core import Brain, RunOpts, Stage, parse_arguments
from shannonist.mi import (
    ContrastivePairwiseFLO,
    ContrastivePairwiseFLOOutput,
    PairwiseMIBatch,
)


class ContrastivePairwiseFLOBrain(
    Brain[TensorDict, ContrastivePairwiseFLOOutput]
):
    """SpeechBrain-style loop for latent-conditioned contrastive FLO."""

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
    ) -> ContrastivePairwiseFLOOutput:
        """Run the estimator on one dataset-generated batch.

        Parameters
        ----------
        batch : TensorDict
            A complete batch produced by the dataset. ``x`` has shape
            ``(batch_size, sequence_length, dim)``.
        stage : Stage
            Current experiment stage.

        Returns
        -------
        ContrastivePairwiseFLOOutput
            Encoded sampled pairs and their critic potentials.
        """
        del stage
        mi_batch = PairwiseMIBatch(
            x=batch["x"],
            batch_size=batch.batch_size,
        )
        return self._estimator().compute_forward(mi_batch)

    def compute_objectives(
        self,
        predictions: ContrastivePairwiseFLOOutput,
        batch: TensorDict,
        stage: Stage,
    ) -> Tensor:
        """Compute FLO and retain learned and ground-truth MI.

        Parameters
        ----------
        predictions : ContrastivePairwiseFLOOutput
            Output produced by :meth:`compute_forward`.
        batch : TensorDict
            Dataset-generated batch. It is unused because all required model
            values are contained in ``predictions``.
        stage : Stage
            Current experiment stage.

        Returns
        -------
        Tensor
            Mean contrastive FLO loss.
        """
        del batch, stage
        objective = self._estimator().compute_objectives(predictions)
        metrics = objective.metrics
        assert metrics is not None
        pair_estimates = -metrics["loss_by_pair"].detach().cpu()
        self._stage_estimates.append(pair_estimates.mean())

        # Position pairs are shared across the batch by the estimator. Record
        # both directions so the reported validation matrix is symmetric.
        pair_indices = predictions.position_indices[0].detach().cpu()
        left = pair_indices[:, 0]
        right = pair_indices[:, 1]
        ones = torch.ones_like(pair_estimates)
        self._stage_matrix_sum.index_put_(
            (left, right), pair_estimates, accumulate=True
        )
        self._stage_matrix_sum.index_put_(
            (right, left), pair_estimates, accumulate=True
        )
        self._stage_matrix_count.index_put_((left, right), ones, accumulate=True)
        self._stage_matrix_count.index_put_((right, left), ones, accumulate=True)

        matrix = torch.as_tensor(
            self.hparams.mutual_information,
            device=predictions.position_indices.device,
        )
        indices = predictions.position_indices
        sampled_target = matrix[indices[..., 0], indices[..., 1]].mean()
        self._stage_targets.append(sampled_target.detach().cpu())
        return objective.loss

    def on_stage_start(self, stage: Stage, epoch: int | None) -> None:
        """Reset accumulated estimates and sampled targets."""
        del stage, epoch
        self._stage_estimates = []
        self._stage_targets = []
        sequence_length = int(self.hparams.sequence_length)
        self._stage_matrix_sum = torch.zeros(sequence_length, sequence_length)
        self._stage_matrix_count = torch.zeros(sequence_length, sequence_length)

    def on_stage_end(
        self,
        stage: Stage,
        stage_loss: float,
        epoch: int | None,
    ) -> None:
        """Report the learned bound and sampled ground truth."""
        learned = torch.stack(self._stage_estimates).mean().item()
        target = torch.stack(self._stage_targets).mean().item()
        epoch_label = f" epoch={epoch}" if epoch is not None else ""
        print(
            f"stage={stage.name.lower()}{epoch_label} "
            f"loss={stage_loss:.6f} "
            f"learned_mi_lower_bound={learned:.6f} "
            f"target_mi={target:.6f}"
        )
        if stage is Stage.VALID:
            learned_matrix = self._stage_matrix_sum / self._stage_matrix_count
            learned_matrix.fill_diagonal_(0)
            ground_truth = torch.as_tensor(
                self.hparams.mutual_information,
                dtype=learned_matrix.dtype,
            )
            print("learned validation MI bound | ground truth MI")
            for learned_row, truth_row in zip(learned_matrix, ground_truth):
                print(
                    f"{_format_matrix_row(learned_row)} | "
                    f"{_format_matrix_row(truth_row)}"
                )

    def _estimator(self) -> ContrastivePairwiseFLO:
        """Return the configured estimator with a checked type."""
        estimator = self.modules["estimator"]
        if not isinstance(estimator, ContrastivePairwiseFLO):
            raise TypeError(
                "the estimator module must be a ContrastivePairwiseFLO"
            )
        return estimator


def _format_matrix_row(row: Tensor) -> str:
    """Format one matrix row with stable column widths."""
    return "[" + " ".join(f"{value.item():8.4f}" for value in row) + "]"


def main(arg_list: Sequence[str] | None = None) -> None:
    """Load hyperparameters and train contrastive pairwise FLO.

    Parameters
    ----------
    arg_list : Sequence[str], optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    param_file, run_opts, overrides = parse_arguments(arg_list)
    with open(param_file, encoding="utf-8") as yaml_file:
        hparams = load_hyperpyyaml(yaml_file, overrides)

    torch.manual_seed(hparams["seed"])
    brain = ContrastivePairwiseFLOBrain(
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

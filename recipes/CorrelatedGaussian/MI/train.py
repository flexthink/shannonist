"""Train an MI estimator on synthetic correlated Gaussian observations."""

from collections.abc import Sequence

import torch
from hyperpyyaml import load_hyperpyyaml
from tensordict import TensorDict
from torch import Tensor

from shannonist.core import Brain, RunOpts, Stage, parse_arguments
from shannonist.mi import (
    JointFLO,
    JointFLOOutput,
    MIBatch,
    JointBA,
    JointBAOutput,
)


MIPrediction = JointFLOOutput | JointBAOutput
MIEstimator = JointFLO | JointBA


class MIBrain(Brain[TensorDict, MIPrediction]):
    """SpeechBrain-style training loop for a paired MI estimator."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._stage_estimates: list[Tensor] = []

    def compute_forward(
        self,
        batch: TensorDict,
        stage: Stage,
    ) -> MIPrediction:
        """Construct an MI batch and run the configured estimator.

        Parameters
        ----------
        batch : TensorDict
            TensorDict containing paired ``x`` and ``y`` observations.
        stage : Stage
            Current experiment stage.

        Returns
        -------
        MIPrediction
            Estimator-specific predictions and auxiliary outputs.
        """
        mi_batch = MIBatch(
            x=batch["x"],
            y=batch["y"],
            batch_size=batch["x"].shape[:1],
        )
        del stage
        return self._estimator().compute_forward(mi_batch)

    def compute_objectives(
        self,
        predictions: MIPrediction,
        batch: TensorDict,
        stage: Stage,
    ) -> Tensor:
        """Return the loss minimized by the training loop.

        Parameters
        ----------
        predictions : MIPrediction
            Estimator output produced by :meth:`compute_forward`.
        batch : TensorDict
            Input batch. It is unused because the estimator embeds all values
            required by its objective in the structured output.
        stage : Stage
            Current experiment stage.

        Returns
        -------
        Tensor
            Loss minimized by the configured estimator.
        """
        del batch, stage
        objective = self._estimator().compute_objectives(predictions)
        self._stage_estimates.append(objective.estimate.detach().cpu())
        return objective.loss

    def on_stage_start(self, stage: Stage, epoch: int | None) -> None:
        """Reset accumulated MI estimates before each stage."""
        del stage, epoch
        self._stage_estimates = []

    def _estimator(self) -> MIEstimator:
        """Return the configured paired estimator with a checked type."""
        estimator = self.modules["estimator"]
        if not isinstance(estimator, (JointFLO, JointBA)):
            raise TypeError(
                "the estimator module must be a JointFLO or JointBA"
            )
        return estimator

    def on_stage_end(
        self,
        stage: Stage,
        stage_loss: float,
        epoch: int | None,
    ) -> None:
        """Report loss, estimated MI, and the target MI after each stage."""
        mi = torch.stack(self._stage_estimates).mean().item()
        epoch_label = f" epoch={epoch}" if epoch is not None else ""
        print(
            f"stage={stage.name.lower()}{epoch_label} "
            f"loss={stage_loss:.6f} mi={mi:.6f} "
            f"target_mi={self.hparams.mutual_information:.6f}"
        )


def main(arg_list: Sequence[str] | None = None) -> None:
    """Load recipe hyperparameters and run MI-estimator training.

    Parameters
    ----------
    arg_list : Sequence[str], optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    param_file, run_opts, overrides = parse_arguments(arg_list)
    with open(param_file, encoding="utf-8") as yaml_file:
        hparams = load_hyperpyyaml(yaml_file, overrides)

    torch.manual_seed(hparams["seed"])
    brain = MIBrain(
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

"""Train FLO on synthetic correlated Gaussian observations."""

from collections.abc import Sequence

import torch
from hyperpyyaml import load_hyperpyyaml
from torch import Tensor

from shannonist.core import Brain, RunOpts, Stage, parse_arguments
from shannonist.mi import BilinearFLO, BilinearFLOOutput, MIBatch


class FLOBrain(Brain[dict[str, Tensor], BilinearFLOOutput]):
    """SpeechBrain-style training loop for a FLO estimator."""

    def compute_forward(
        self,
        batch: dict[str, Tensor],
        stage: Stage,
    ) -> BilinearFLOOutput:
        """Construct an MI batch and run the FLO estimator.

        Parameters
        ----------
        batch : dict[str, Tensor]
            Dictionary containing paired ``x`` and ``y`` observations.
        stage : Stage
            Current experiment stage.

        Returns
        -------
        BilinearFLOOutput
            Critic predictions and auxiliary outputs used by the FLO loss.
        """
        mi_batch = MIBatch(
            x=batch["x"],
            y=batch["y"],
            batch_size=batch["x"].shape[:1],
        )
        estimator = self.modules["estimator"]
        if not isinstance(estimator, BilinearFLO):
            raise TypeError("the estimator module must be a BilinearFLO")

        del stage
        return estimator.compute_forward(mi_batch)

    def compute_objectives(
        self,
        predictions: BilinearFLOOutput,
        batch: dict[str, Tensor],
        stage: Stage,
    ) -> Tensor:
        """Return the loss minimized by the training loop.

        Parameters
        ----------
        predictions : BilinearFLOOutput
            FLO output produced by :meth:`compute_forward`.
        batch : dict[str, Tensor]
            Input batch. It is unused because FLO embeds its loss in the
            structured output.
        stage : Stage
            Current experiment stage.

        Returns
        -------
        Tensor
            FLO loss, equal to the negative MI estimate.
        """
        del batch, stage
        estimator = self.modules["estimator"]
        if not isinstance(estimator, BilinearFLO):
            raise TypeError("the estimator module must be a BilinearFLO")
        return estimator.compute_objectives(predictions).loss

    def on_stage_end(
        self,
        stage: Stage,
        stage_loss: float,
        epoch: int | None,
    ) -> None:
        """Report loss, estimated MI, and the target MI after each stage."""
        epoch_label = f" epoch={epoch}" if epoch is not None else ""
        print(
            f"stage={stage.name.lower()}{epoch_label} "
            f"loss={stage_loss:.6f} mi={-stage_loss:.6f} "
            f"target_mi={self.hparams.mutual_information:.6f}"
        )


def main(arg_list: Sequence[str] | None = None) -> None:
    """Load recipe hyperparameters and run FLO training.

    Parameters
    ----------
    arg_list : Sequence[str], optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    param_file, run_opts, overrides = parse_arguments(arg_list)
    with open(param_file, encoding="utf-8") as yaml_file:
        hparams = load_hyperpyyaml(yaml_file, overrides)

    torch.manual_seed(hparams["seed"])
    brain = FLOBrain(
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

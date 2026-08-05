import torch
from tensordict import TensorClass, TensorDict
from torch import Tensor, nn

from shannonist.framework import ObjectiveOutput, TrainableEstimator
from shannonist.mi.types import MIBatch, MIEstimate
from shannonist.models.critic import BilinearCritic, BilinearCriticOutput


class BilinearFLOOutput(TensorClass):
    """Predictions and auxiliary outputs produced by bilinear FLO.

    Parameters
    ----------
    critic : BilinearCriticOutput
        Encoded representations and potential values produced by the critic.
    """

    critic: BilinearCriticOutput


def LossFLO(predictions: BilinearFLOOutput) -> tuple[Tensor, TensorDict]:
    """Compute the contrastive Fenchel-Legendre optimization loss.

    Positive pairs occupy matching positions in the two critic
    representations. All non-matching pairs are treated as negative samples.

    Parameters
    ----------
    predictions : BilinearFLOOutput
        Precomputed critic representations and potential values.

    Returns
    -------
    tuple[Tensor, TensorDict]
        Mean FLO loss and diagnostic tensors containing the per-example loss,
        similarity matrix, and potential values.

    Raises
    ------
    ValueError
        If feature shapes are incompatible or fewer than two paired samples
        are available.

    References
    ----------
    Qing Guo et al., "Tight Mutual Information Estimation With Contrastive
    Fenchel-Legendre Optimization," NeurIPS 2022.
    https://arxiv.org/abs/2107.01131
    """
    hx = predictions.critic.hx
    hy = predictions.critic.hy
    u = predictions.critic.u
    if hx.ndim != 2 or hy.ndim != 2:
        raise ValueError("critic representations must have shape (batch, features)")
    if hx.shape != hy.shape:
        raise ValueError("critic representations must have identical shapes")

    batch_size = hx.shape[0]
    if batch_size < 2:
        raise ValueError("FLO requires at least two samples")

    similarity = hx @ hy.transpose(0, 1)
    positive_mask = torch.eye(
        batch_size,
        dtype=torch.bool,
        device=similarity.device,
    )
    g = similarity[positive_mask].reshape(batch_size, 1)
    g0 = similarity[~positive_mask].reshape(batch_size, batch_size - 1)
    g0_logsumexp = torch.logsumexp(g0, dim=1, keepdim=True)
    loss_vec = u + torch.exp(-u + g0_logsumexp - g) / (batch_size - 1) - 1
    loss = loss_vec.mean()

    details = TensorDict(
        {
            "loss_vec": loss_vec,
            "similarity": similarity,
            "u": u,
        },
        batch_size=[],
    )
    return loss, details


class BilinearFLO(
    nn.Module,
    TrainableEstimator[MIBatch, BilinearFLOOutput, MIEstimate],
):
    """Bilinear FLO mutual-information estimator.

    The estimator constructs its own :class:`BilinearCritic` from the supplied
    encoders and potential. Model execution and objective computation are kept
    separate, and optimization remains the caller's responsibility.

    Parameters
    ----------
    encoder_x : nn.Module
        Module mapping ``x`` observations to feature representations.
    encoder_y : nn.Module
        Module mapping ``y`` observations to feature representations.
    potential : nn.Module
        Module computing the FLO potential for aligned representations.
    tau : float, default=1.0
        Initial value of the learnable temperature parameter.
    use_norm : bool, default=True
        Whether to L2-normalize encoded representations.
    """

    def __init__(
        self,
        encoder_x: nn.Module,
        encoder_y: nn.Module,
        potential: nn.Module,
        tau: float = 1.0,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.critic = BilinearCritic(
            encoder_x=encoder_x,
            encoder_y=encoder_y,
            potential=potential,
            tau=tau,
            use_norm=use_norm,
        )

    def compute_forward(self, batch: MIBatch) -> BilinearFLOOutput:
        """Compute critic predictions for a paired batch.

        Parameters
        ----------
        batch : MIBatch
            Paired, unmasked observations.

        Returns
        -------
        BilinearFLOOutput
            Critic representations and potential values used by FLO.

        Raises
        ------
        NotImplementedError
            If either observation mask is present.
        """
        self._validate_masks(batch)
        critic_output = self.critic(batch.x, batch.y)
        return BilinearFLOOutput(
            critic=critic_output,
            batch_size=critic_output.batch_size,
        )

    def compute_objectives(
        self,
        predictions: BilinearFLOOutput,
    ) -> ObjectiveOutput:
        """Compute the differentiable FLO objective from predictions.

        Parameters
        ----------
        predictions : BilinearFLOOutput
            Output previously returned by :meth:`compute_forward`.

        Returns
        -------
        ObjectiveOutput
            Scalar FLO loss and diagnostic tensors.
        """
        loss, details = LossFLO(predictions)
        return ObjectiveOutput(loss=loss, metrics=details, batch_size=[])

    def MI(self, predictions: BilinearFLOOutput) -> tuple[Tensor, TensorDict]:
        """Estimate mutual information from precomputed predictions.

        Parameters
        ----------
        predictions : BilinearFLOOutput
            Output previously returned by :meth:`compute_forward`.

        Returns
        -------
        tuple[Tensor, TensorDict]
            Mutual-information estimate and FLO diagnostic tensors.
        """
        objective = self.compute_objectives(predictions)
        details = objective.metrics
        assert details is not None
        return -objective.loss, details

    def estimate(self, batch: MIBatch) -> MIEstimate:
        """Estimate mutual information for a batch.

        Parameters
        ----------
        batch : MIBatch
            Paired, unmasked observations.

        Returns
        -------
        MIEstimate
            FLO mutual-information estimate and diagnostic tensors.
        """
        predictions = self.compute_forward(batch)
        value, details = self.MI(predictions)
        return MIEstimate(value=value, details=details, batch_size=[])

    @staticmethod
    def _validate_masks(batch: MIBatch) -> None:
        """Reject masks until masked FLO estimation is implemented."""
        if batch.x_mask is not None or batch.y_mask is not None:
            raise NotImplementedError("BilinearFLO does not yet support masks")


__all__ = ["BilinearFLO", "BilinearFLOOutput", "LossFLO"]

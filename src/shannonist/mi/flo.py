import torch
from tensordict import TensorClass, TensorDict
from torch import Tensor, nn

from shannonist.framework import ObjectiveOutput, TrainableEstimator
from shannonist.mi.types import MIBatch, MIEstimate, PairwiseMIBatch
from shannonist.models.critic import (
    BilinearCritic,
    BilinearCriticOutput,
    PairwiseCritic,
)
from shannonist.models.mlp import MultiMLP


class BilinearFLOOutput(TensorClass):
    """Predictions and auxiliary outputs produced by bilinear FLO.

    Parameters
    ----------
    critic : BilinearCriticOutput
        Encoded representations and potential values produced by the critic.
    """

    critic: BilinearCriticOutput


def flo_loss(predictions: BilinearFLOOutput) -> tuple[Tensor, TensorDict]:
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
        loss, details = flo_loss(predictions)
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


class PairwiseFLOOutput(TensorClass):
    """Predictions and auxiliary outputs produced by pairwise FLO.

    Parameters
    ----------
    hx : Tensor
        Temperature-scaled representations with shape
        ``(*, count, feature_dim)``.
    u : Tensor
        Symmetric FLO potentials with shape ``(*, count, count)``.
    mask : Tensor
        Boolean valid-position mask with shape ``(*, count)``.
    """

    hx: Tensor
    u: Tensor
    mask: Tensor


def pairwise_flo_loss(
    predictions: PairwiseFLOOutput,
) -> tuple[Tensor, TensorDict]:
    """Compute jointly optimized FLO losses for all distinct pairs.

    All leading dimensions are flattened into the sample dimension. For pair
    ``(i, j)``, only samples where both mask positions are valid participate in
    positives, negatives, or averaging. FLO is evaluated independently for
    every ordered pair, the two directions are averaged to enforce symmetry,
    and the scalar training loss is the mean over unique off-diagonal pairs.

    Parameters
    ----------
    predictions : PairwiseFLOOutput
        Precomputed representations and symmetric potential values.

    Returns
    -------
    tuple[Tensor, TensorDict]
        Mean loss over unique pairs and diagnostic tensors. The
        ``loss_matrix`` diagnostic has shape ``(count, count)`` and a zero
        diagonal.

    Raises
    ------
    ValueError
        If prediction shapes are incompatible or any pair has fewer than two
        jointly valid samples.
    """
    hx = predictions.hx
    u = predictions.u
    mask = predictions.mask
    if hx.ndim < 3:
        raise ValueError("hx must have shape (*, count, feature_dim)")
    if u.shape != (*hx.shape[:-1], hx.shape[-2]):
        raise ValueError("u must have shape (*, count, count)")
    if mask.shape != hx.shape[:-1]:
        raise ValueError("mask must have shape (*, count)")

    count = hx.shape[-2]
    feature_dim = hx.shape[-1]
    hx = hx.reshape(-1, count, feature_dim)
    u = u.reshape(-1, count, count)
    mask = mask.reshape(-1, count).bool()
    sample_count = hx.shape[0]
    zero = hx.sum() * 0
    pair_losses: dict[tuple[int, int], Tensor] = {}
    loss_vec = hx.new_zeros(sample_count, count, count)
    valid_counts = torch.zeros(count, count, dtype=torch.long, device=hx.device)

    for i in range(count):
        for j in range(i + 1, count):
            valid = mask[:, i] & mask[:, j]
            valid_count = int(valid.sum().item())
            if valid_count < 2:
                raise ValueError(
                    f"pair ({i}, {j}) has fewer than two jointly valid samples"
                )

            hx_i = hx[valid, i]
            hx_j = hx[valid, j]
            potential = u[valid, i, j]
            directional_vectors = []
            for left, right in ((hx_i, hx_j), (hx_j, hx_i)):
                pair_similarity = left @ right.transpose(0, 1)
                positive_mask = torch.eye(
                    valid_count,
                    dtype=torch.bool,
                    device=hx.device,
                )
                g = pair_similarity[positive_mask]
                g0 = pair_similarity[~positive_mask].reshape(
                    valid_count,
                    valid_count - 1,
                )
                directional_vectors.append(
                    potential
                    + torch.exp(
                        -potential + torch.logsumexp(g0, dim=1) - g
                    )
                    / (valid_count - 1)
                    - 1
                )

            pair_loss_vec = torch.stack(directional_vectors).mean(dim=0)
            pair_loss = pair_loss_vec.mean()
            pair_losses[(i, j)] = pair_loss
            loss_vec[valid, i, j] = pair_loss_vec
            loss_vec[valid, j, i] = pair_loss_vec
            valid_counts[i, j] = valid_count
            valid_counts[j, i] = valid_count

    loss_matrix = torch.stack(
        [
            torch.stack(
                [
                    zero
                    if i == j
                    else pair_losses[(min(i, j), max(i, j))]
                    for j in range(count)
                ]
            )
            for i in range(count)
        ]
    )
    loss = torch.stack(list(pair_losses.values())).mean()
    similarity = torch.einsum("nif,mjf->ijnm", hx, hx)
    details = TensorDict(
        {
            "loss_vec": loss_vec,
            "loss_matrix": loss_matrix,
            "similarity": similarity,
            "u": u,
            "mask": mask,
            "valid_counts": valid_counts,
        },
        batch_size=[],
    )
    return loss, details


class PairwiseFLO(
    nn.Module,
    TrainableEstimator[PairwiseMIBatch, PairwiseFLOOutput, MIEstimate],
):
    """Joint FLO estimator for pairwise mutual-information matrices.

    The estimator creates one :class:`PairwiseCritic` whose ``MultiMLP``
    encoder and symmetric potential are optimized jointly across all unique
    pairs. The diagonal is excluded from training and reported as zero.

    Parameters
    ----------
    encoder : MultiMLP
        Parallel encoder for all count positions.
    count : int
        Number of variables in the input's penultimate dimension.
    tau : float, default=1.0
        Initial value of the learnable temperature parameter.
    use_norm : bool, default=True
        Whether to L2-normalize encoded representations.
    """

    def __init__(
        self,
        encoder: MultiMLP,
        count: int,
        tau: float = 1.0,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if count < 2:
            raise ValueError("pairwise FLO requires count to be at least two")
        self.count = count
        self.critic = PairwiseCritic(
            encoder=encoder,
            count=count,
            use_norm=use_norm,
        )
        self.tau = nn.Parameter(torch.as_tensor([tau]))

    def compute_forward(self, batch: PairwiseMIBatch) -> PairwiseFLOOutput:
        """Compute encoded representations and pairwise potentials.

        Parameters
        ----------
        batch : PairwiseMIBatch
            Observations with shape ``(*, count, features)``.

        Returns
        -------
        PairwiseFLOOutput
            Temperature-scaled representations, symmetric potentials, and the
            normalized valid-position mask.
        """
        hx = self.critic.encode(batch.x)
        u = self.critic.compute_interactions(hx)
        mask = self._normalize_mask(batch.x_mask, hx)
        return PairwiseFLOOutput(
            hx=hx / torch.sqrt(self.tau),
            u=u,
            mask=mask,
            batch_size=hx.shape[:-2],
        )

    def _normalize_mask(self, mask: Tensor | None, hx: Tensor) -> Tensor:
        """Return a boolean mask with shape ``(*, count)``.

        Parameters
        ----------
        mask : Tensor, optional
            Input mask with shape ``(*, count)`` or ``(*, count, 1)``.
        hx : Tensor
            Encoded representations defining the expected leading shape.

        Returns
        -------
        Tensor
            Boolean valid-position mask.

        Raises
        ------
        ValueError
            If the mask shape is incompatible with the encoded input.
        """
        expected_shape = hx.shape[:-1]
        if mask is None:
            return torch.ones(expected_shape, dtype=torch.bool, device=hx.device)
        if mask.shape == (*expected_shape, 1):
            mask = mask.squeeze(-1)
        if mask.shape != expected_shape:
            raise ValueError(
                "x_mask must have shape (*, count) or (*, count, 1)"
            )
        return mask.to(device=hx.device, dtype=torch.bool)

    def compute_objectives(
        self,
        predictions: PairwiseFLOOutput,
    ) -> ObjectiveOutput:
        """Compute the mean FLO objective over all unique pairs.

        Parameters
        ----------
        predictions : PairwiseFLOOutput
            Output previously returned by :meth:`compute_forward`.

        Returns
        -------
        ObjectiveOutput
            Scalar joint loss and pairwise diagnostics.
        """
        loss, details = pairwise_flo_loss(predictions)
        return ObjectiveOutput(loss=loss, metrics=details, batch_size=[])

    def estimate(self, batch: PairwiseMIBatch) -> MIEstimate:
        """Estimate a symmetric pairwise mutual-information matrix.

        Parameters
        ----------
        batch : PairwiseMIBatch
            Observations with shape ``(*, count, features)``.

        Returns
        -------
        MIEstimate
            Estimate with shape ``(count, count)`` and a zero diagonal.
        """
        predictions = self.compute_forward(batch)
        objective = self.compute_objectives(predictions)
        details = objective.metrics
        assert details is not None
        return MIEstimate(
            value=-details["loss_matrix"],
            details=details,
            batch_size=[],
        )


__all__ = [
    "BilinearFLO",
    "BilinearFLOOutput",
    "PairwiseFLO",
    "PairwiseFLOOutput",
    "flo_loss",
    "pairwise_flo_loss",
]

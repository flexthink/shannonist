import torch
from tensordict import TensorClass, TensorDict
from torch import Tensor, nn

from shannonist.framework import ObjectiveOutput, TrainableEstimator
from shannonist.mi.types import MIBatch, MIEstimate, PairwiseMIBatch
from shannonist.models.critic import (
    BilinearCritic,
    BilinearCriticOutput,
    PairwiseCritic,
    SymmetricPairwiseCritic,
)
from shannonist.models.mlp import MLP, MultiMLP


class JointFLOOutput(TensorClass):
    """Predictions and auxiliary outputs produced by joint FLO.

    Parameters
    ----------
    critic : BilinearCriticOutput
        Encoded representations and potential values produced by the critic.
    """

    critic: BilinearCriticOutput


def flo_loss(predictions: JointFLOOutput) -> tuple[Tensor, TensorDict]:
    """Compute the contrastive Fenchel-Legendre optimization loss.

    Positive pairs occupy matching positions in the two critic
    representations. All non-matching pairs are treated as negative samples.

    Parameters
    ----------
    predictions : JointFLOOutput
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


class JointFLO(
    nn.Module,
    TrainableEstimator[MIBatch, JointFLOOutput, MIEstimate],
):
    """Joint FLO mutual-information estimator.

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

    def compute_forward(self, batch: MIBatch) -> JointFLOOutput:
        """Compute critic predictions for a paired batch.

        Parameters
        ----------
        batch : MIBatch
            Paired, unmasked observations.

        Returns
        -------
        JointFLOOutput
            Critic representations and potential values used by FLO.

        Raises
        ------
        NotImplementedError
            If either observation mask is present.
        """
        self._validate_masks(batch)
        critic_output = self.critic(batch.x, batch.y)
        return JointFLOOutput(
            critic=critic_output,
            batch_size=critic_output.batch_size,
        )

    def compute_objectives(
        self,
        predictions: JointFLOOutput,
    ) -> ObjectiveOutput:
        """Compute the differentiable FLO objective from predictions.

        Parameters
        ----------
        predictions : JointFLOOutput
            Output previously returned by :meth:`compute_forward`.

        Returns
        -------
        ObjectiveOutput
            Scalar FLO loss and diagnostic tensors.
        """
        loss, details = flo_loss(predictions)
        return ObjectiveOutput(
            loss=loss,
            estimate=-loss,
            metrics=details,
            batch_size=[],
        )

    def MI(self, predictions: JointFLOOutput) -> tuple[Tensor, TensorDict]:
        """Estimate mutual information from precomputed predictions.

        Parameters
        ----------
        predictions : JointFLOOutput
            Output previously returned by :meth:`compute_forward`.

        Returns
        -------
        tuple[Tensor, TensorDict]
            Mutual-information estimate and FLO diagnostic tensors.
        """
        objective = self.compute_objectives(predictions)
        details = objective.metrics
        assert details is not None
        return objective.estimate, details

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
            raise NotImplementedError("JointFLO does not yet support masks")


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

    # A sample belongs to pair (i, j) only if both variable positions exist.
    pair_valid = mask.transpose(0, 1).unsqueeze(1) & mask.transpose(0, 1).unsqueeze(0)
    pair_counts = pair_valid.sum(dim=-1)
    upper_triangle = torch.triu(
        torch.ones(count, count, dtype=torch.bool, device=hx.device),
        diagonal=1,
    )
    diagonal = torch.eye(count, dtype=torch.bool, device=hx.device)
    insufficient = upper_triangle & (pair_counts < 2)
    if insufficient.any():
        i, j = insufficient.nonzero(as_tuple=False)[0].tolist()
        raise ValueError(
            f"pair ({i}, {j}) has fewer than two jointly valid samples"
        )

    similarity = torch.einsum("nif,mjf->ijnm", hx, hx)
    identity = torch.eye(sample_count, dtype=torch.bool, device=hx.device)
    valid_combinations = pair_valid.unsqueeze(-1) & pair_valid.unsqueeze(-2)
    negative_mask = valid_combinations & ~identity
    g = similarity.diagonal(dim1=-2, dim2=-1)
    g0_logsumexp = similarity.masked_fill(~negative_mask, -torch.inf).logsumexp(
        dim=-1
    )

    u_by_pair = u.permute(1, 2, 0)
    pair_indices = torch.arange(count, device=hx.device)
    canonical_direction = pair_indices[:, None] <= pair_indices[None, :]
    symmetric_u = torch.where(
        canonical_direction.unsqueeze(-1),
        u_by_pair,
        u_by_pair.transpose(0, 1),
    )
    denominator = (pair_counts - 1).clamp_min(1).unsqueeze(-1)
    directed_loss_vec = (
        symmetric_u
        + torch.exp(-symmetric_u + g0_logsumexp - g) / denominator
        - 1
    )
    directed_loss_vec = directed_loss_vec.masked_fill(~pair_valid, 0)
    symmetric_loss_vec = (
        directed_loss_vec + directed_loss_vec.transpose(0, 1)
    ) / 2
    symmetric_loss_vec = symmetric_loss_vec.masked_fill(
        diagonal.unsqueeze(-1),
        0,
    )
    loss_vec = symmetric_loss_vec.permute(2, 0, 1)
    loss_matrix = symmetric_loss_vec.sum(dim=-1) / pair_counts.clamp_min(1)
    loss_matrix = loss_matrix.masked_fill(diagonal, 0)
    loss = loss_matrix[upper_triangle].mean()
    valid_counts = pair_counts.masked_fill(diagonal, 0)
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
        return ObjectiveOutput(
            loss=loss,
            estimate=-details["loss_matrix"],
            metrics=details,
            batch_size=[],
        )

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
            value=objective.estimate,
            details=details,
            batch_size=[],
        )


class ContrastivePairwiseFLOOutput(TensorClass):
    """Predictions produced by contrastive pairwise FLO.

    Parameters
    ----------
    hx : Tensor
        Temperature-scaled representations with shape
        ``(batch, num_samples, 2, feature_dim)``.
    u : Tensor
        Symmetric critic potentials with shape
        ``(batch, num_samples, 2, 2)``.
    position_indices : Tensor
        Sampled sequence indices with shape ``(batch, num_samples, 2)``.
    """

    hx: Tensor
    u: Tensor
    position_indices: Tensor


def contrastive_pairwise_flo_loss(
    predictions: ContrastivePairwiseFLOOutput,
) -> tuple[Tensor, TensorDict]:
    """Compute FLO over sampled within-sequence position pairs.

    Each sampled pair slot defines a separate FLO problem across the batch.
    Matching batch indices are joint observations; non-matching batch indices
    are product-of-marginals observations. The two interaction directions are
    averaged, followed by averaging over sampled pairs and batch items.

    Parameters
    ----------
    predictions : ContrastivePairwiseFLOOutput
        Encoded sampled pairs and their symmetric potentials.

    Returns
    -------
    tuple[Tensor, TensorDict]
        Scalar mean loss and diagnostic tensors.

    Raises
    ------
    ValueError
        If prediction shapes are incompatible or the batch contains fewer
        than two samples.
    """
    hx = predictions.hx
    u = predictions.u
    position_indices = predictions.position_indices
    if hx.ndim != 4 or hx.shape[-2] != 2:
        raise ValueError(
            "hx must have shape (batch, num_samples, 2, feature_dim)"
        )
    batch_size, num_samples = hx.shape[:2]
    if batch_size < 2:
        raise ValueError("contrastive pairwise FLO requires batch size >= 2")
    if num_samples < 1:
        raise ValueError("at least one sampled position pair is required")
    if u.shape != (batch_size, num_samples, 2, 2):
        raise ValueError("u must have shape (batch, num_samples, 2, 2)")
    if position_indices.shape != (batch_size, num_samples, 2):
        raise ValueError(
            "position_indices must have shape (batch, num_samples, 2)"
        )

    left = hx[:, :, 0]
    right = hx[:, :, 1]
    forward_similarity = torch.einsum("bnf,cnf->nbc", left, right)
    reverse_similarity = torch.einsum("bnf,cnf->nbc", right, left)
    similarity = torch.stack((forward_similarity, reverse_similarity), dim=1)

    independent = ~torch.eye(
        batch_size,
        dtype=torch.bool,
        device=hx.device,
    )
    joint = similarity.diagonal(dim1=-2, dim2=-1)
    independent_logsumexp = similarity.masked_fill(
        ~independent,
        -torch.inf,
    ).logsumexp(dim=-1)

    potential = u[:, :, 0, 1].transpose(0, 1).unsqueeze(1)
    directed_loss = (
        potential
        + torch.exp(-potential + independent_logsumexp - joint)
        / (batch_size - 1)
        - 1
    )
    loss_vec = directed_loss.mean(dim=1).transpose(0, 1)
    loss_by_pair = loss_vec.mean(dim=0)
    loss = loss_by_pair.mean()

    details = TensorDict(
        {
            "loss_vec": loss_vec,
            "loss_by_pair": loss_by_pair,
            "estimate_vec": -loss_vec,
            "estimate_by_pair": -loss_by_pair,
            "similarity": similarity,
            "u": u,
            "position_indices": position_indices,
        },
        batch_size=[],
    )
    return loss, details


class ContrastivePairwiseFLO(
    nn.Module,
    TrainableEstimator[
        PairwiseMIBatch,
        ContrastivePairwiseFLOOutput,
        MIEstimate,
    ],
):
    """Estimate MI between randomly sampled positions in sequences.

    A shared encoder is applied to both members of every position pair. For
    each sampled pair slot, aligned batch items form the joint distribution
    and all differently indexed batch items form the independent distribution.
    Padded positions are never sampled.

    Parameters
    ----------
    encoder : MLP
        Encoder shared by every sequence position.
    sample_size : int
        Maximum number of position pairs sampled per sequence and pass. The
        actual number is capped by the smallest valid sequence length in the
        batch.
    tau : float, default=1.0
        Initial value of the learnable temperature parameter.
    use_norm : bool, default=True
        Whether to L2-normalize encoded representations.
    """

    def __init__(
        self,
        encoder: MLP,
        sample_size: int,
        tau: float = 1.0,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        self.sample_size = sample_size
        self.critic = SymmetricPairwiseCritic(
            encoder=encoder,
            use_norm=use_norm,
        )
        self.tau = nn.Parameter(torch.as_tensor([tau]))

    def compute_forward(
        self,
        batch: PairwiseMIBatch,
    ) -> ContrastivePairwiseFLOOutput:
        """Sample valid position pairs and evaluate the shared critic.

        Parameters
        ----------
        batch : PairwiseMIBatch
            Sequence observations with shape ``(*, length, features)`` and an
            optional mask of shape ``(*, length)`` or ``(*, length, 1)``.
            Leading dimensions are flattened into one mandatory batch axis.

        Returns
        -------
        ContrastivePairwiseFLOOutput
            Encoded pairs, potentials, and their source position indices.

        Raises
        ------
        ValueError
            If input or mask shapes are invalid, the flattened batch has fewer
            than two items, or a sample has no valid positions.
        """
        x = batch.x
        if x.ndim < 3:
            raise ValueError("x must have shape (*, length, features)")
        length, feature_dim = x.shape[-2:]
        x = x.reshape(-1, length, feature_dim)
        batch_size = x.shape[0]
        if batch_size < 2:
            raise ValueError("contrastive pairwise FLO requires batch size >= 2")

        mask = self._normalize_mask(batch.x_mask, batch.x)
        mask = mask.reshape(batch_size, length)
        valid_counts = mask.sum(dim=-1)
        minimum_valid = int(valid_counts.min().item())
        if minimum_valid < 1:
            raise ValueError("every sample must contain at least one valid position")
        num_samples = min(self.sample_size, minimum_valid)
        position_indices = self._sample_position_pairs(mask, num_samples)

        batch_indices = torch.arange(batch_size, device=x.device)[:, None, None]
        sampled = x[batch_indices, position_indices]
        hx = self.critic.encode(sampled)
        u = self.critic.compute_interactions(hx)
        return ContrastivePairwiseFLOOutput(
            hx=hx / torch.sqrt(self.tau),
            u=u,
            position_indices=position_indices,
            batch_size=[batch_size, num_samples],
        )

    @staticmethod
    def _normalize_mask(mask: Tensor | None, x: Tensor) -> Tensor:
        """Return a boolean mask matching the sequence dimensions.

        Parameters
        ----------
        mask : Tensor, optional
            Mask with shape ``(*, length)`` or ``(*, length, 1)``.
        x : Tensor
            Input with shape ``(*, length, features)``.

        Returns
        -------
        Tensor
            Boolean mask with shape ``(*, length)``.
        """
        expected_shape = x.shape[:-1]
        if mask is None:
            return torch.ones(expected_shape, dtype=torch.bool, device=x.device)
        if mask.shape == (*expected_shape, 1):
            mask = mask.squeeze(-1)
        if mask.shape != expected_shape:
            raise ValueError(
                "x_mask must have shape (*, length) or (*, length, 1)"
            )
        return mask.to(device=x.device, dtype=torch.bool)

    @staticmethod
    def _sample_position_pairs(mask: Tensor, num_samples: int) -> Tensor:
        """Sample position-matched pairs without selecting masked positions.

        Positions are matched across samples by valid-token ordinal. Thus a
        sampled pair of ordinals is shared by the whole batch, even when mask
        layouts differ. The first ordinals are sampled without replacement.
        Whenever all samples have at least two valid positions, the second
        ordinal differs from the first. A one-position sequence necessarily
        pairs that position with itself.

        Parameters
        ----------
        mask : Tensor
            Boolean validity mask with shape ``(batch, length)``.
        num_samples : int
            Number of pairs to draw for every batch item.

        Returns
        -------
        Tensor
            Sampled indices with shape ``(batch, num_samples, 2)``.
        """
        minimum_valid = int(mask.sum(dim=-1).min().item())
        left_ordinals = torch.randperm(
            minimum_valid,
            device=mask.device,
        )[:num_samples]
        if minimum_valid == 1:
            right_ordinals = left_ordinals
        else:
            right_ordinals = torch.randint(
                minimum_valid - 1,
                (num_samples,),
                device=mask.device,
            )
            right_ordinals = right_ordinals + (
                right_ordinals >= left_ordinals
            )

        valid_positions = torch.stack(
            [
                sample_mask.nonzero(as_tuple=False).squeeze(-1)[:minimum_valid]
                for sample_mask in mask
            ]
        )
        left = valid_positions[:, left_ordinals]
        right = valid_positions[:, right_ordinals]
        return torch.stack((left, right), dim=-1)

    def compute_objectives(
        self,
        predictions: ContrastivePairwiseFLOOutput,
    ) -> ObjectiveOutput:
        """Compute the mean FLO objective over sampled position pairs.

        Parameters
        ----------
        predictions : ContrastivePairwiseFLOOutput
            Output previously returned by :meth:`compute_forward`.

        Returns
        -------
        ObjectiveOutput
            Scalar FLO loss and pair-level diagnostics.
        """
        loss, details = contrastive_pairwise_flo_loss(predictions)
        return ObjectiveOutput(
            loss=loss,
            estimate=-loss,
            metrics=details,
            batch_size=[],
        )

    def estimate(self, batch: PairwiseMIBatch) -> MIEstimate:
        """Estimate mean MI across newly sampled sequence-position pairs.

        Parameters
        ----------
        batch : PairwiseMIBatch
            Masked or unmasked sequence observations.

        Returns
        -------
        MIEstimate
            Scalar sampled FLO lower bound and diagnostic tensors.
        """
        predictions = self.compute_forward(batch)
        objective = self.compute_objectives(predictions)
        return MIEstimate(
            value=objective.estimate,
            details=objective.metrics,
            batch_size=[],
        )


# Compatibility with the spelling originally requested in the public API.
ContrastivePairwiseFLow = ContrastivePairwiseFLO


__all__ = [
    "JointFLO",
    "JointFLOOutput",
    "ContrastivePairwiseFLO",
    "ContrastivePairwiseFLow",
    "ContrastivePairwiseFLOOutput",
    "PairwiseFLO",
    "PairwiseFLOOutput",
    "contrastive_pairwise_flo_loss",
    "flo_loss",
    "pairwise_flo_loss",
]

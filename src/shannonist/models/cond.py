"""Conditioning-vector aggregation modules."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class Conditioning(nn.Module, ABC):
    """Interface for preprocessing optional conditioning inputs."""

    @abstractmethod
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Convert a raw conditioning input into a conditioning vector."""
        ...


class IdentityConditioning(Conditioning):
    """Return an already-vectorized conditioning tensor unchanged."""

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return ``x`` without modification."""
        return x


class AttentionPoolingConditioning(Conditioning):
    """Pool feature bags with learned masked attention.

    A shared linear scorer produces one attention logit for every position.
    Softmax normalization is performed over the count dimension after masked
    positions have been excluded. Rows with no valid positions produce a zero
    vector.

    Parameters
    ----------
    dim : int
        Input and output feature dimension.
    bias : bool, default=True
        Whether the attention scorer includes a scalar bias.
    """

    def __init__(self, dim: int, bias: bool = True) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.score = nn.Linear(dim, 1, bias=bias)
        nn.init.xavier_uniform_(self.score.weight)
        if self.score.bias is not None:
            nn.init.zeros_(self.score.bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Pool valid positions over the count dimension.

        Parameters
        ----------
        x : Tensor
            Feature tensor with shape ``(batch, count, dim)``.
        mask : Tensor, optional
            Valid-position mask with shape ``(batch, count)`` or
            ``(batch, count, 1)``. Nonzero values identify included positions.
            If omitted, every position is included.

        Returns
        -------
        Tensor
            Attention-pooled features with shape ``(batch, dim)``.

        Raises
        ------
        ValueError
            If an input shape is incompatible with the module.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, count, dim)")
        if x.shape[-1] != self.dim:
            raise ValueError(f"x must have trailing dimension {self.dim}")

        expected_mask_shape = x.shape[:-1]
        if mask is None:
            mask = torch.ones(
                expected_mask_shape,
                dtype=torch.bool,
                device=x.device,
            )
        else:
            if mask.shape == (*expected_mask_shape, 1):
                mask = mask.squeeze(-1)
            if mask.shape != expected_mask_shape:
                raise ValueError(
                    "mask must have shape (batch, count) or "
                    "(batch, count, 1)"
                )
            mask = mask.to(device=x.device, dtype=torch.bool)

        logits = self.score(x).squeeze(-1)
        minimum = torch.finfo(logits.dtype).min
        weights = torch.softmax(logits.masked_fill(~mask, minimum), dim=-1)
        has_valid = mask.any(dim=-1, keepdim=True)
        weights = torch.where(has_valid, weights, torch.zeros_like(weights))
        return torch.einsum("bc,bcf->bf", weights, x)


class TransformerConditioning(Conditioning):
    """Encode feature sequences and mean-pool valid positions.

    Parameters
    ----------
    dim : int
        Input, transformer, and output feature dimension.
    num_layers : int, default=1
        Number of transformer encoder layers.
    num_heads : int, default=1
        Number of self-attention heads in each encoder layer.
    dim_feedforward : int, optional
        Transformer feed-forward width. Defaults to ``4 * dim``.
    dropout : float, default=0.0
        Dropout probability within transformer encoder layers.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 1,
        num_heads: int = 1,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if dim_feedforward is None:
            dim_feedforward = 4 * dim
        if dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1)")

        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Encode and average unmasked positions over the time axis.

        Parameters
        ----------
        x : Tensor
            Input tensor with shape ``(batch, time, dim)``.
        mask : Tensor, optional
            Valid-position mask with shape ``(batch, time)`` or
            ``(batch, time, 1)``. Nonzero values identify included positions.

        Returns
        -------
        Tensor
            Mean-pooled transformer states with shape ``(batch, dim)``.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, time, dim)")
        if x.shape[-1] != self.dim:
            raise ValueError(f"x must have trailing dimension {self.dim}")
        valid = _normalize_sequence_mask(mask, x)
        has_valid = valid.any(dim=-1)

        # PyTorch attention cannot consume a row whose every key is masked.
        # Expose one zero placeholder for those rows, then zero their pooled
        # outputs after encoding.
        safe_valid = valid.clone()
        safe_valid[~has_valid, 0] = True
        safe_x = x.masked_fill(~valid.unsqueeze(-1), 0)
        encoded = self.encoder(
            safe_x,
            src_key_padding_mask=~safe_valid,
        )
        encoded = encoded.masked_fill(~valid.unsqueeze(-1), 0)
        denominator = valid.sum(dim=-1, keepdim=True).clamp_min(1)
        pooled = encoded.sum(dim=-2) / denominator.to(encoded.dtype)
        return pooled.masked_fill(~has_valid.unsqueeze(-1), 0)


def _normalize_sequence_mask(mask: Tensor | None, x: Tensor) -> Tensor:
    """Return a boolean mask matching a batched sequence input."""
    expected_shape = x.shape[:-1]
    if mask is None:
        return torch.ones(expected_shape, dtype=torch.bool, device=x.device)
    if mask.shape == (*expected_shape, 1):
        mask = mask.squeeze(-1)
    if mask.shape != expected_shape:
        raise ValueError(
            "mask must have shape (batch, time) or (batch, time, 1)"
        )
    return mask.to(device=x.device, dtype=torch.bool)


def make_conditioning(
    name: str,
    dim: int,
    opts: Mapping[str, Any] | None = None,
) -> Conditioning:
    """Construct a conditioning preprocessor from a recognized name.

    Parameters
    ----------
    name : str
        One of ``"identity"``, ``"attention_pooling"``, or ``"transformer"``.
    dim : int
        Conditioning feature dimension.
    opts : Mapping[str, Any], optional
        Constructor options for the selected conditioning module.

    Returns
    -------
    Conditioning
        Constructed conditioning module.

    Raises
    ------
    ValueError
        If ``name`` is unknown or identity conditioning receives options.
    """
    options = dict(opts or {})
    normalized_name = name.lower().replace("-", "_")
    if normalized_name in {"identity", "none", "passthrough"}:
        if options:
            unexpected = ", ".join(sorted(options))
            raise ValueError(
                f"identity conditioning does not accept options: {unexpected}"
            )
        return IdentityConditioning()
    if normalized_name in {"attention", "attention_pooling"}:
        return AttentionPoolingConditioning(dim=dim, **options)
    if normalized_name in {"transformer", "transformer_encoder"}:
        return TransformerConditioning(dim=dim, **options)
    raise ValueError(f"unknown conditioning: {name}")


__all__ = [
    "AttentionPoolingConditioning",
    "Conditioning",
    "IdentityConditioning",
    "TransformerConditioning",
    "make_conditioning",
]

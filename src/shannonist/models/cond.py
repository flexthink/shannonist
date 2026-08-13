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


def make_conditioning(
    name: str,
    dim: int,
    opts: Mapping[str, Any] | None = None,
) -> Conditioning:
    """Construct a conditioning preprocessor from a recognized name.

    Parameters
    ----------
    name : str
        One of ``"identity"`` or ``"attention_pooling"``.
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
    raise ValueError(f"unknown conditioning: {name}")


__all__ = [
    "AttentionPoolingConditioning",
    "Conditioning",
    "IdentityConditioning",
    "make_conditioning",
]

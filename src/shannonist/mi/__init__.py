"""Mutual-information estimation interfaces and implementations."""

from shannonist.mi.data import (
    CorrelatedGausian,
    LatentPairwiseCorrelatedGaussian,
    LatentPairwiseCorrelatentGaussian,
    PairwiseCorrelatedGaussian,
    tensordict_collate,
    tensordict_passthrough,
)
from shannonist.mi.flo import (
    BilinearFLO,
    BilinearFLOOutput,
    ContrastivePairwiseFLO,
    ContrastivePairwiseFLow,
    ContrastivePairwiseFLOOutput,
    PairwiseFLO,
    PairwiseFLOOutput,
    contrastive_pairwise_flo_loss,
    flo_loss,
    pairwise_flo_loss,
)
from shannonist.mi.types import MIBatch, MIEstimate, PairwiseMIBatch

__all__ = [
    "BilinearFLO",
    "BilinearFLOOutput",
    "ContrastivePairwiseFLO",
    "ContrastivePairwiseFLow",
    "ContrastivePairwiseFLOOutput",
    "CorrelatedGausian",
    "LatentPairwiseCorrelatedGaussian",
    "LatentPairwiseCorrelatentGaussian",
    "PairwiseFLO",
    "PairwiseFLOOutput",
    "PairwiseCorrelatedGaussian",
    "PairwiseMIBatch",
    "contrastive_pairwise_flo_loss",
    "flo_loss",
    "pairwise_flo_loss",
    "tensordict_collate",
    "tensordict_passthrough",
    "MIBatch",
    "MIEstimate",
]

"""Mutual-information estimation interfaces and implementations."""

from shannonist.mi.data import (
    CorrelatedGausian,
    PairwiseCorrelatedGaussian,
    tensordict_collate,
)
from shannonist.mi.flo import (
    BilinearFLO,
    BilinearFLOOutput,
    PairwiseFLO,
    PairwiseFLOOutput,
    flo_loss,
    pairwise_flo_loss,
)
from shannonist.mi.types import MIBatch, MIEstimate, PairwiseMIBatch

__all__ = [
    "BilinearFLO",
    "BilinearFLOOutput",
    "CorrelatedGausian",
    "PairwiseFLO",
    "PairwiseFLOOutput",
    "PairwiseCorrelatedGaussian",
    "PairwiseMIBatch",
    "flo_loss",
    "pairwise_flo_loss",
    "tensordict_collate",
    "MIBatch",
    "MIEstimate",
]

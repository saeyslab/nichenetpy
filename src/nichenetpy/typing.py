from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, TypeVar

from numpy import dtype, float64, ndarray
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

type nichenet_matrix = (
    csr_matrix[float64] | csc_matrix[float64] | csr_array[float64] | csc_array[float64] | ndarray[Any, dtype[float64]]
)
"""Matrix types that are supported in NicheNetPy."""

type gene_t = str | int
"""Gene symbol types that are supported in NicheNetPy."""

type target_eval_metric_t = Literal["aupr"] | Literal["aupr_corrected"] | Literal["auroc"] | Literal["pearson"]
"""Metrics to evaluate the target prediction. """

type ligand_eval_metric_t = (
    Literal["aupr"]
    | Literal["aupr_corrected"]
    | Literal["auroc"]
    | Literal["pearson"]
    | Literal["map"]
    | Literal["ndcg"]
)
"""Metrics to evaluate the ligand ranking. """

type eval_metric_t = target_eval_metric_t | ligand_eval_metric_t
"""Evaluation metrics. """

type ObjectiveFunction = Callable[[Sequence[float] | ndarray[Any, dtype[float64]]], float]
"""Objective function for the source weight optimization"""


class MetricScoreFunction(Protocol):
    """Function that computes the metric score (which decides the best metric to rank the ligands with)"""

    def __call__(self, **kwargs: float) -> float: ...  # noqa: D102


GeneSymbol = TypeVar("GeneSymbol", str, int)
"""Gene symbol types that are supported in NicheNetPy."""

EvalMetric = TypeVar("EvalMetric", target_eval_metric_t, ligand_eval_metric_t)
"""Evaluation metrics. """

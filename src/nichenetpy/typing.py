from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, TypeVar

from numpy import dtype, floating, ndarray
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

type nichenet_matrix = (
    csr_matrix[floating]
    | csc_matrix[floating]
    | csr_array[floating]
    | csc_array[floating]
    | ndarray[Any, dtype[floating]]
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

type ObjectiveFunction = Callable[[Sequence[float] | ndarray[Any, dtype[floating]]], float]
"""Objective function for the source weight optimization"""


class MetricScoreFunction(Protocol):
    """Function that computes the metric score (which decides the best metric to rank the ligands with)"""

    def __call__(self, **kwargs: float) -> float: ...  # noqa: D102


GeneSymbol = TypeVar("GeneSymbol", str, int)
"""Gene symbol types that are supported in NicheNetPy."""

EvalMetric = TypeVar("EvalMetric", target_eval_metric_t, ligand_eval_metric_t)
"""Evaluation metrics. """


def validate_matrix_type(obj: object, err_msg: str) -> None:
    """
    Check if the object is an ungeneric nichenet_matrix and raise a TypeError if it isn't.

    Parameters
    ----------
    obj : object
        the object to validate
    err_msg : str
        the error message

    Raises
    ------
    TypeError
        if the object is not an (ungeneric) nichenet_matrix
    """
    if not isinstance(obj, (csr_matrix, csc_matrix, csr_array, csc_array, ndarray)):
        raise TypeError(err_msg)

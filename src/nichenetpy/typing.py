from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, TypeVar, overload

from numpy import bool_, dtype, floating, integer, ndarray
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
"""Gene symbol types that are supported in NicheNetPy (numpy integers are accepted as single genes and converted to
int)."""

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

type ObjectiveFunction = Callable[[Sequence[float] | ndarray[Any, dtype[floating]]], float | floating]
"""Objective function for the source weight optimization"""


class MetricScoreFunction(Protocol):
    """Function that computes the metric score (which decides the best metric to rank the ligands with)"""

    def __call__(self, **kwargs: float) -> float: ...  # noqa: D102


GeneSymbol = TypeVar("GeneSymbol", str, int)
"""Gene symbol types that are supported in NicheNetPy."""

EvalMetric = TypeVar("EvalMetric", target_eval_metric_t, ligand_eval_metric_t)
"""Evaluation metrics. """


@overload
def _as_gene(obj: GeneSymbol, name: str = "gene") -> GeneSymbol: ...


@overload
def _as_gene(obj: object, name: str = "gene") -> gene_t: ...


def _as_gene(obj: object, name: str = "gene") -> gene_t:
    """
    Return a single gene as a Python str or int.

    Parameters
    ----------
    obj : gene_t
        the object to check
    name : str
        the name of the argument, used in the error message

    Returns
    -------
    gene_t
        the gene, with numpy strings and integers converted to str and int

    Raises
    ------
    TypeError
        if obj isn't a single gene (a bool is not a gene)

    Notes
    -----
    numpy integers (e.g. from a numpy array or a DataFrame column) are accepted at runtime, although they're not part
    of gene_t, so the genes that are returned or stored are always Python values. Collections of genes (gene sets,
    backgrounds, ...) aren't converted, they work for lookups because np.int64(1) == 1 (with the same hash).
    """
    if not isinstance(obj, (bool, bool_)):
        if isinstance(obj, str):
            return str(obj)  # a Python str for a numpy str
        if isinstance(obj, (int, integer)):
            return int(obj)
    raise TypeError(f"{name} should have type gene_t, was {type(obj)}")


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

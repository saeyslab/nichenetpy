from collections.abc import Callable, Iterable, Sequence, Collection
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

type gene_combination_t = tuple[str, ...] | tuple[int, ...]
"""Gene combination types that are supported in NicheNetPy (a combination of ligands, e.g. ligands acting together)."""

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

GeneCombination = tuple[GeneSymbol, ...]
"""Gene combination types that are supported in NicheNetPy, linked to GeneSymbol (the genes of a combination have the
same type as the other genes). Used as GeneCombination[GeneSymbol]."""

EvalMetric = TypeVar("EvalMetric", target_eval_metric_t, ligand_eval_metric_t)
"""Evaluation metrics. """


def _combination_name(members: Iterable[GeneSymbol]) -> GeneSymbol | GeneCombination[GeneSymbol]:
    """
    Return the name of a combination of ligands in a model.

    Parameters
    ----------
    members : Iterable of gene_t
        the ligands of the combination

    Returns
    -------
    gene_t or gene_combination_t
        the gene symbols joined by "-" (e.g. "TNF-LTB"), or the tuple of the members for int gene ids
    """
    members = tuple(members)
    if all(isinstance(member, str) for member in members):
        return "-".join(members)  # type: ignore # GeneSymbol is str here
    return members


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
    raise TypeError(
        f"{name} should have type gene_t, was {type(obj)}"
    )  # this is a weird placement but otherwise typechecker complains


def _as_ligand(
    obj: GeneSymbol | GeneCombination[GeneSymbol], name: str = "ligand"
) -> GeneSymbol | GeneCombination[GeneSymbol]:
    """
    Return the name of a ligand in a model: a single gene (see `_as_gene`), or the name of a combination given as a
    tuple of genes (see `_combination_name`: "A-B" for gene symbols, the tuple for int gene ids, the gene itself for a
    tuple of one gene).

    Parameters
    ----------
    obj : gene_t or gene_combination_t
        the object to check
    name : str
        the name of the argument, used in the error message

    Returns
    -------
    gene_t or gene_combination_t
        the ligand, with numpy strings and integers converted to str and int

    Raises
    ------
    TypeError
        if obj isn't a single gene or a non-empty tuple of genes
    """
    if isinstance(obj, tuple):
        members = tuple(_as_gene(e, name) for e in obj)
        if len(members) == 0:
            raise TypeError(f"{name} should be a gene or a non-empty tuple of genes, was an empty tuple")
        return members[0] if len(members) == 1 else _combination_name(members)
    return _as_gene(obj, name)


def _check_iterable(values: object, name: str, elements: str) -> None:
    """
    Check that an argument is an Iterable and not a single str (a str is an Iterable of its characters).

    Parameters
    ----------
    values : object
        the argument
    name : str
        the name of the argument, used in the error message
    elements : str
        what the Iterable contains, used in the error message (e.g. "gene symbols")

    Raises
    ------
    TypeError
        if values is a str or isn't an Iterable
    """
    if isinstance(values, str):
        raise TypeError(f"{name} should be an Iterable of {elements}, not a single str: {values!r}")
    if not isinstance(values, Iterable):
        raise TypeError(f"{name} should be an Iterable of {elements}, was {type(values)}")


def _reiterable[T](values: Iterable[T]) -> Iterable[T]:
    # a Collection (list, tuple, set, dict, numpy array, pandas Index, range, dict views) produces a new iterator
    # every time it's iterated; anything else (e.g. a generator) is stored in a list
    return values if isinstance(values, Collection) else list(values)


def _as_metrics(metrics: Iterable[EvalMetric], name: str) -> Iterable[EvalMetric]:
    """
    Check evaluation metrics and return them so they can be used more than once and in a fixed order.

    Parameters
    ----------
    metrics : Iterable of str
        the evaluation metrics
    name : str
        the name of the argument, used in the error message

    Returns
    -------
    Iterable of str
        a set in sorted order (the order of a set changes between Python processes), an iterator (e.g. a generator) in
        a list, any other Iterable as it is

    Raises
    ------
    TypeError
        if metrics is a str (a single metric would be split into its characters) or isn't an Iterable
    """
    _check_iterable(metrics, name, "metric names")
    if isinstance(metrics, (set, frozenset)):
        return sorted(metrics)
    return _reiterable(metrics)


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

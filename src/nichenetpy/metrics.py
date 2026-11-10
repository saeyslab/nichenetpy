import warnings
from collections.abc import Callable, Iterable, Sequence
from numbers import Number
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_curve, roc_curve

from nichenetpy.ann_utils import _subset_layer, subset_ann
from nichenetpy.exception import NicheNetError
from nichenetpy.typing import eval_metric_t, nichenet_matrix, validate_matrix_type
from nichenetpy.utils import subset_matrix
from nichenetpy.wilcoxon import wilcoxon_rank_sum_test, wilcoxon_rank_sum_test_with_correlation


def _auc_reverse(
    x: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    y: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    """
    Calculates the area under the curve using the trapezoid rule. The points should be specified in descending order of the x-values.

    Parameters
    ----------
    x : list or tuple or numpy.ndarray of float
        list of x-values on the curve
    y : list or tuple or numpy.ndarray of float
        list of y-values on the curve

    Returns
    -------
    float
        the area under the curve

    Raises
    ------
    ValueError
        if x and y do not have matching length of at least 2

    Examples
    --------
    >>> _auc_reverse(
        (1, 1, 1, 1, 1, 1, 0.85714286, 0.85714286, 0.57142857, 0.42857143, 0.42857143, 0.28571429, 0.14285714, 0),
        (0.5, 0.53846154, 0.58333333, 0.63636364, 0.7, 0.77777778, 0.75, 0.85714286, 0.8, 0.75, 1, 1, 1, 1)
    )
    0.8851473922902493
    """
    if len(x) != len(y):
        raise ValueError("x and y should have the same length")
    if len(x) < 2:
        raise ValueError("x and y should have a length of at least 2")
    if type(x) is np.ndarray and type(y) is np.ndarray:
        return ((x[:-1] - x[1:]) * (y[1:] + y[:-1])).sum() / 2
    return sum((x[i - 1] - x[i]) * (y[i] + y[i - 1]) for i in range(1, len(x))) / 2


def calculate_aupr(
    response: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    prediction: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    """
    Calculates the area under the precision-recall curve using the trapezoid rule.

    Parameters
    ----------
    response : list or tuple or numpy.ndarray of float
        vector indicating whether a target is a True (1) target of the possibly active ligand(s) or a False (0)
    prediction : list or tuple or numpy.ndarray of float
        vector which contains probability scores for each target gene (for one particular ligand)

    Returns
    -------
    float
        the area under the precision-recall curve

    Raises
    ------
    TypeError
        if the arguments have the wrong type

    Examples
    --------
    >>> calculate_aupr(
        (1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
        (0.5, 0.12, 0.47, 0.36, 0.0, 1.0, 0.78, 0.66, 0.24, 0.42, 0.95, 0.47, 0.39, 0.07)
    )
    0.8851473922902493
    """
    if not isinstance(response, (Sequence, np.ndarray)):
        raise TypeError(f"response should be a Sequence or numpy.ndarray of floats, had type {type(response)}")
    if not isinstance(prediction, (Sequence, np.ndarray)):
        raise TypeError(f"prediction should be a Sequence or numpy.ndarray of floats, had type {type(prediction)}")
    if sum(response) == 0:
        warnings.warn("There are no true samples in response, AUPR is undefined", stacklevel=2)
        return np.nan
    precision, recall, _ = precision_recall_curve(response, prediction)
    return _auc_reverse(recall, precision)


def calculate_auroc(
    response: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    prediction: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    """
    Calculates the area under the roc-curve using the trapezoid rule.

    Parameters
    ----------
    response : Sequence or numpy.ndarray of float
        vector indicating whether a target is a True (1) target of the possibly active ligand(s) or a False (0)
    prediction : Sequence or numpy.ndarray of float
        vector which contains probability scores for each target gene (for one particular ligand)

    Returns
    -------
    float
        the area under the roc-curve

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if not isinstance(response, (Sequence, np.ndarray)):
        raise TypeError(f"response should be a Sequence or numpy.ndarray of floats, had type {type(response)}")
    if not isinstance(prediction, (Sequence, np.ndarray)):
        raise TypeError(f"prediction should be a Sequence or numpy.ndarray of floats, had type {type(prediction)}")
    fp, tp, _ = roc_curve(response, prediction)
    fp, tp = zip(*sorted(zip(fp, tp, strict=True), key=lambda x: x[0]), strict=True)
    return -_auc_reverse(fp, tp)


def precision_at_k(
    ranking: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    response: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    k: int,
) -> float:
    """
    Calculates the precision at index k in the ranking.

    Parameters
    ----------
    ranking : list or tuple or numpy.ndarray of float
        the ranking of response indices from best to worst
    response : list or tuple or numpy.ndarray of float
        vector indicating whether a target is a True (1) target of the possibly active ligand(s) or a False (0)
    k : int
        the index at which to compute the precision

    Returns
    -------
    float
        the precision at k
    """
    if not isinstance(ranking, (Sequence, np.ndarray)):
        raise TypeError(f"ranking should be a Sequence or numpy.ndarray of floats, had type {type(ranking)}")
    if not isinstance(response, (Sequence, np.ndarray)):
        raise TypeError(f"response should be a Sequence or numpy.ndarray of floats, had type {type(response)}")
    if type(k) is not int:
        raise TypeError(f"k should be of type int, had type {type(k)}")
    return sum(response[e] for e in ranking[: k + 1]) / (k + 1)  # type: ignore # e is float


def average_precision_at_k(
    response: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    prediction: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    k: int | None = None,
) -> float:
    """
    Calculates the average precision at index k in the ranking.

    Parameters
    ----------
    response : list or tuple or numpy.ndarray of float
        vector indicating whether a target is a True (1) target of the possibly active ligand(s) or a False (0)
    prediction : list or tuple or numpy.ndarray of float
        vector which contains probability scores for each target gene (for one particular ligand)
    k : int or None
        the index at which to compute the precision or None to compute the precision over everything

    Returns
    -------
    float
        the precision at k
    """
    if not isinstance(response, (Sequence, np.ndarray)):
        raise TypeError(f"response should be a Sequence or numpy.ndarray of floats, had type {type(response)}")
    if not isinstance(prediction, (Sequence, np.ndarray)):
        raise TypeError(f"prediction should be a Sequence or numpy.ndarray of floats, had type {type(prediction)}")
    if k is None:
        k = len(response) - 1
    elif type(k) is not int:
        raise TypeError(f"k should be of type int, had type {type(k)}")
    ranking = [e[0] for e in sorted(enumerate(prediction), key=lambda x: x[1], reverse=True)]
    n = sum(response)
    if n == 1:
        i = 0
        while response[ranking[i]] != 1:
            i += 1
        return 1 / (i + 1) if i <= k else 0
    return sum(precision_at_k(ranking, response, i) if response[ranking[i]] else 0 for i in range(k + 1)) / n


def dcg_at_k(
    ranking: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    response: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    k: int,
) -> float:
    """
    Calculates the discounted cumulative gain at index k in the ranking.

    Parameters
    ----------
    ranking : list or tuple or numpy.ndarray of float
        the ranking of response indices from best to worst
    response : list or tuple or numpy.ndarray of float
        vector indicating whether a target is a True (1) target of the possibly active ligand(s) or a False (0)
    k : int
        the index at which to compute the dcg

    Returns
    -------
    float
        the discounted cumulative gain at k
    """
    if not isinstance(response, (Sequence, np.ndarray)):
        raise TypeError(f"response should be a Sequence or numpy.ndarray of floats, had type {type(response)}")
    if not isinstance(ranking, (Sequence, np.ndarray)):
        raise TypeError(f"ranking should be a Sequence or numpy.ndarray of floats, had type {type(ranking)}")
    return sum(response[e] / np.log2(i + 2) for i, e in enumerate(ranking[: k + 1]))  # type: ignore # e is float


def ndcg_at_k(
    response: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    prediction: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    k: int | None = None,
):
    """
    Calculates the normalized discounted cumulative gain at index k in the ranking.

    Parameters
    ----------
    response : Sequence or numpy.ndarray of float
        vector indicating whether a target is a True (1) target of the possibly active ligand(s) or a False (0)
    prediction : Sequence or numpy.ndarray of float
        the ranking of response indices from best to worst
    k : int
        the index at which to compute the ndcg

    Returns
    -------
    float
        the normalized discounted cumulative gain at k
    """
    if not isinstance(response, (Sequence, np.ndarray)):
        raise TypeError(f"response should be a Sequence or numpy.ndarray of floats, had type {type(response)}")
    if not isinstance(prediction, (Sequence, np.ndarray)):
        raise TypeError(f"prediction should be a Sequence or numpy.ndarray of floats, had type {type(prediction)}")
    if k is None:
        k = len(response) - 1
    elif type(k) is not int:
        raise TypeError(f"k should be of type int, had type {type(k)}")
    ranking = [e[0] for e in sorted(enumerate(prediction), key=lambda x: x[1], reverse=True)]
    n = sum(response)
    gs_ranking = [e[0] for e in sorted(enumerate(response), key=lambda x: x[1], reverse=True)]
    # if n is 1 then ideal dcg@k is 1, so let's take a shortcut
    return dcg_at_k(ranking, response, k) / (1 if n == 1 else dcg_at_k(gs_ranking, response, k))


_metric_dct: dict[
    eval_metric_t,
    Callable[
        [
            Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
            Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
        ],
        float,
    ],
] = {
    "aupr": calculate_aupr,
    "aupr_corrected": lambda response, prediction: calculate_aupr(response, prediction) - sum(response) / len(response),
    "auroc": calculate_auroc,
    "pearson": lambda response, prediction: pearsonr(response, prediction).statistic,
    "map": average_precision_at_k,
    "ndcg": ndcg_at_k,
}


def calculate_evaluation_metrics[EvalMetric](
    prediction: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    response: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    metrics: Iterable[EvalMetric],
    allow_nan: bool = False,
) -> dict[EvalMetric, float]:
    """
    Calculates evaluation metrics that can be used to rank ligands or to evaluate the ligand ranking.

    Parameters
    ----------
    prediction : list or tuple or numpy.ndarray of float
        vector which contains probability scores for each target gene (for one particular ligand)
    response : list or tuple or numpy.ndarray of float
        vector indicating whether a target is a True (1) target of the possibly active ligand(s) or a False (0)
    metrics : Iterable of str
        the metrics to compute, must be a subset of ("aupr", "aupr_corrected", "auroc", "pearson", "map", "ndcg")
    allow_nan : bool
        if True, return nan values in case a specific metric is undefined, if False the errors are not caught

    Returns
    -------
    dict[str, float]
        dictionary with as keys the names of the supported metrics and as values the computed metrics

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if response doesn't contain any true samples
    """
    if not isinstance(response, (Sequence, np.ndarray)):
        raise TypeError(f"response should be a list or tuple or numpy.ndarray of floats, had type {type(response)}")
    if not isinstance(prediction, (Sequence, np.ndarray)):
        raise TypeError(f"prediction should be a list or tuple or numpy.ndarray of floats, had type {type(prediction)}")
    if sum(response) == 0:
        raise ValueError(
            "There are no true samples in response. aupr, auroc and pearson correlation coëfficient are undefined."
        )
    output: dict[EvalMetric, float] = {}
    warnings.filterwarnings("ignore")
    for metric in metrics:
        try:
            output[metric] = _metric_dct[metric](response, prediction)  # type: ignore # EvalMetric is always of type eval_metric_t
        except ValueError as ex:
            if allow_nan:
                output[metric] = np.nan
            else:
                raise ex
    warnings.filterwarnings("default")
    return output


def calculate_prediction_evaluation_metrics(
    prediction: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    response: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]],
    allow_nan: bool = False,
) -> dict[eval_metric_t, float]:
    """
    Calculates metrics that can be used to rank ligands.
    currently supported metrics are:

        AUPR
        corrected AUPR
        AUROC
        Pearson correlation

    Parameters
    ----------
    prediction : Sequence or numpy.ndarray of float
        vector which contains probability scores for each target gene (for one particular ligand)
    response : Sequence or numpy.ndarray of float
        vector indicating whether a target is a True (1) target of the possibly active ligand(s) or a False (0)
    allow_nan : bool
        if True, return nan values in case a specific metric is undefined, if False the errors are not caught

    Returns
    -------
    dict[str, float]
        dictionary with as keys the names of the supported metrics and as values the computed metrics

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if response doesn't contain any true samples
    """
    return calculate_evaluation_metrics(
        prediction=prediction,
        response=response,
        metrics=("aupr", "aupr_corrected", "auroc", "pearson"),
        allow_nan=allow_nan,
    )


def _sub_log_fold_change(
    data: nichenet_matrix,
    denormalize: Callable[[nichenet_matrix], nichenet_matrix] | None = np.expm1,  # type: ignore # np.expm1 is valid
    pseudocount: float = 1,
):
    if denormalize is not None:
        data = denormalize(data)
    if data.shape is None:
        raise ValueError("data.shape is None")
    return np.log2((data.sum(axis=0) + pseudocount) / data.shape[0])


def log_fold_change(
    mat1: nichenet_matrix,
    mat2: nichenet_matrix,
    denormalize: Callable[[nichenet_matrix], nichenet_matrix] | None = np.expm1,  # type: ignore # np.expm1 is valid
    pseudocount: float = 1,
) -> np.ndarray:
    """
    Calculates the log fold changes the way seurat does it.

    Parameters
    ----------
    mat1 : numpy.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        the first matrix
    mat2 : numpy.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        the second matrix
    denormalize : Callable
        a denormalization function to apply prior to the calculation
    pseudocount : float
        the pseudocount, to ensure that the log of 0 is never taken
        the pseudocount is divided by the amount of cells

    Returns
    -------
    numpy.ndarray
        the log fold changes

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    validate_matrix_type(mat1, f"mat1 should have type nichenet_matrix, was {type(mat1)}")
    validate_matrix_type(mat2, f"mat2 should have type nichenet_matrix, was {type(mat2)}")
    if denormalize is not None and not isinstance(denormalize, Callable):  # type: ignore
        raise TypeError(f"denormalize should be a Callable or None, had type {type(denormalize)}")
    if not isinstance(pseudocount, Number):
        raise TypeError(f"pseudocount should have type float, was {type(pseudocount)}")
    # make sure the output is a 1D array
    return np.asarray(
        _sub_log_fold_change(mat1, denormalize, pseudocount) - _sub_log_fold_change(mat2, denormalize, pseudocount)
    ).ravel()


def _calc_pct(mat: nichenet_matrix, direction: str = "both") -> list[float]:
    if not isinstance(mat, (csc_matrix, csr_matrix, csc_array, csr_array, np.ndarray)):
        raise TypeError(f"mat should be of type nichenet_matrix, was {type(mat)}")
    if direction == "positive":
        hits = mat > 0  # type: ignore
    elif direction == "negative":
        hits = mat < 0  # type: ignore
    else:
        hits = mat != 0
    return (np.ravel(hits.sum(axis=0)) / mat.shape[0]).tolist()  # type: ignore


def gene_expression_pct(mat: nichenet_matrix) -> list[float]:
    """
    For each gene, calculate the percentage of cells that have an expression value not equal to 0.

    Parameters
    ----------
    mat : numpy.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        (#cells X #genes) matrix containing the expression values

    Returns
    -------
    list
        for each gene the percentage of cells that have an expression value not equal to 0

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    return _calc_pct(mat)


def _single_group_metrics(
    ann: AnnData,
    mat: nichenet_matrix,
    row2index: dict[str, int],
    lfc_denormalize: Callable[[nichenet_matrix], nichenet_matrix] | None,
    lfc_pseudocount: float,
    groupby: str,
    group_oi: str,
    group_ref: str | None = None,
):
    cells_oi = ann.obs[ann.obs[groupby] == group_oi].index
    if group_ref is None:
        # reference cells are all cells that are not cells of interest
        cells_ref = ann.obs[ann.obs[groupby] != group_oi].index
    else:
        cells_ref = ann.obs[ann.obs[groupby] == group_ref].index
    if len(cells_oi) == 0:
        raise NicheNetError("There are no cells in the group of interest")
    if len(cells_ref) == 0:
        raise NicheNetError("There are no cells in the reference group")
    mat1 = subset_matrix(mat, rows=[row2index[cell] for cell in cells_oi])
    mat2 = subset_matrix(mat, rows=[row2index[cell] for cell in cells_ref])
    return (
        log_fold_change(mat1, mat2, lfc_denormalize, lfc_pseudocount),
        gene_expression_pct(mat1),
        gene_expression_pct(mat2),
    )


def group_metrics[GeneSymbol](
    ann: AnnData,
    groupby: str,
    group_oi: str | None = None,
    group_ref: str | None = None,
    layer: str = "data",
    lfc_pseudocount: float = 1,
    tie_correction: bool = True,
    features: Iterable[GeneSymbol] | None = None,
    min_abs_lfc: float = 0,
    min_pct: float = 0,
    pval_thresh: float | None = None,  # 0.01 in seurat
    wilcoxon_limma: bool = False,
    lfc_denormalize: Callable[[nichenet_matrix], nichenet_matrix] | None = np.expm1,  # type: ignore # np.expm1 is valid
):
    """
    For each gene, calculate the percentage of cells that have an expression value greater than 0,
    the log fold changes and the p-values / adjusted p-values.
    The result is a pandas dataframe stored in ann.uns["group_metrics"] with the columns "gene", `groupby`, "lfc",
    "pct" (fraction of expressing cells in the group), "pct_ref" (fraction of expressing cells in the reference cells:
    the other groups or `group_ref`), "pval" and "pval_adj".

    Parameters
    ----------
    ann : AnnData
        the AnnData object
    groupby : str
        the column in ann.obs to group by
    group_oi : str or None
        the group of interest
    group_ref : str or None
        the reference group
    layer : str
        the layer in the AnnData object to use
    lfc_pseudocount : float
        the pseudocount to use in the computation of the log fold changes
    tie_correction : bool
        if True, tie correction is performed through averaging
    features : Iterable of str or None
        the genes to consider
    min_abs_lfc : float
        a gene is only tested (and reported) in a comparison if its absolute log fold change is at least this value
        (as logfc.threshold in Seurat's FindMarkers)
    min_pct : float
        a gene is only tested (and reported) in a comparison if it's expressed in at least this fraction of the cells of
        the group or of the reference cells (as min.pct in Seurat's FindMarkers)
    pval_thresh : float or None
        upper bound for the p-values (if p_values for a gene is smaller than this threshold, it is excluded)
    wilcoxon_limma : bool
        use wilcoxon-limma (reproduces results from seuratv4)
    lfc_denormalize : Callable or None
        a denormalization function to apply prior to the calculation of the log fold changes

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    NicheNetError
        if there are no cells in the group of interest
        if there are no cells in the reference group
    ValueError
        if the arguments are invalid
    RuntimeError
        if the Anndata object could not be subsetted correctly

    Notes
    -----
    The result is the same as seurat's FindMarkers function.
    """
    if type(ann) is not AnnData:
        raise TypeError(f"ann should be of type AnnData, was {type(ann)}")
    if type(groupby) is not str:
        raise TypeError(f"groupby should be of type str, was {type(groupby)}")
    if group_oi is not None and type(group_oi) is not str:
        raise TypeError(f"group_oi should be of type str, was {type(group_oi)}")
    if group_ref is not None and type(group_ref) is not str:
        raise TypeError(f"group_ref should be of type str, was {type(group_ref)}")
    if type(layer) is not str:
        raise TypeError(f"layer should be of type str, was {type(layer)}")
    if not isinstance(lfc_pseudocount, Number):
        raise TypeError(f"lfc_pseudocount should be of type float, was {type(lfc_pseudocount)}")
    if type(tie_correction) is not bool:
        raise TypeError(f"tie_correction should be of type bool, was {type(tie_correction)}")
    if features is not None and not isinstance(features, Iterable):
        raise TypeError(f"features should be an Iterable of strings, had type {type(features)}")
    if not isinstance(min_abs_lfc, Number):
        raise TypeError(f"min_abs_lfc should be of type float, was {type(min_abs_lfc)}")
    if not isinstance(min_pct, Number):
        raise TypeError(f"min_pct should be of type float, was {type(min_pct)}")
    if pval_thresh is not None and not isinstance(pval_thresh, Number):
        raise TypeError(f"pval_thresh should be of type float, was {type(pval_thresh)}")
    if lfc_denormalize is not None and not isinstance(lfc_denormalize, Callable):  # type: ignore
        raise TypeError(f"lfc_denormalize should be a Callable or None, had type {type(lfc_denormalize)}")
    if features is None:
        mat = ann.layers[layer]
        genes = ann.var_names
    else:
        # select genes
        mat, genes = _subset_layer(ann, layer, features)
    validate_matrix_type(mat, f"the selected layer should be of type nichenet_matrix, was {type(mat)}")
    row2index = dict(zip(ann.obs.index, range(len(ann.obs.index)), strict=True))
    try:
        groups = sorted(set(ann.obs[groupby]))
    except KeyError as error:
        raise ValueError(f"can't group by {groupby} as it is not present in the AnnData object") from error
    lfc = []
    pct = []
    pct_ref = []
    # start by computing lfc and pct (these are also used for filtering before computing the p-values)
    if group_oi is None:
        # compare each group with the reference group
        for group in groups:
            x, y, z = _single_group_metrics(
                ann,
                mat,  # type: ignore # mat is type checked
                row2index,
                lfc_denormalize,
                lfc_pseudocount,
                groupby,
                group,
                group_ref,
            )
            lfc.append(x)
            pct.append(y)
            pct_ref.append(z)
    else:
        x, y, z = _single_group_metrics(
            ann,
            mat,  # type: ignore # mat is type checked
            row2index,
            lfc_denormalize,
            lfc_pseudocount,
            groupby,
            group_oi,
            group_ref,
        )
        lfc.append(x)
        pct.append(y)
        pct_ref.append(z)
    output = pd.melt(
        pd.DataFrame(np.column_stack(lfc), index=genes, columns=(groups if group_oi is None else [group_oi])),  # type: ignore # error at runtime
        var_name=groupby,
        value_name="lfc",
        ignore_index=False,
    )
    output.index.name = "gene"
    output.reset_index(inplace=True)
    # pct: fraction of expressing cells in the group of interest, pct_ref: in the reference cells (the other groups or
    # group_ref); both are needed for the prefilter, otherwise genes lost in the group of interest
    # (low pct, high pct_ref) would never be tested
    pct_groups = groups if group_oi is None else [group_oi]
    pct_df = pd.DataFrame(
        {
            groupby: np.repeat(pct_groups, len(genes)),
            "gene": np.tile(np.asarray(genes), len(pct_groups)),
            "pct": np.concatenate([np.asarray(e, dtype=float) for e in pct]),
            "pct_ref": np.concatenate([np.asarray(e, dtype=float) for e in pct_ref]),
        }
    )
    output = output.merge(pct_df, on=["gene", groupby], how="inner")  # type: ignore
    # prefilter as in Seurat's FindMarkers: a gene is tested in a comparison if it's expressed in at least min_pct of
    # the cells of either group and its absolute log fold change is at least min_abs_lfc
    output = output[
        (np.maximum(output["pct"], output["pct_ref"]) >= min_pct) & (abs(output["lfc"]) >= min_abs_lfc)
    ]
    ann_orig = ann
    if group_oi is not None and group_ref is not None:
        # select only the groups the group of interest and the reference group
        ann_sub = subset_ann(ann, val=(group_oi, group_ref), val_col=groupby)
        if ann_sub is None:
            raise RuntimeError("Failed to correctly subset the AnnData object. ")
        ann = ann_sub
    # compute the p-values
    # genes passing the pct / lfc filters in at least one comparison (rows that didn't pass are already removed)
    test_genes = list(set(output["gene"]))
    if wilcoxon_limma:
        gene2index = dict(zip(ann.var_names, range(len(ann.var_names)), strict=True))
        mat = subset_matrix(ann.layers[layer], cols=[gene2index[gene] for gene in test_genes])  # type: ignore # the layer is type checked
        groups = set(ann.obs[groupby])
        group_masks = {group: (ann.obs[groupby] == group).to_numpy() for group in groups}
        pvals = {group: [] for group in groups}
        for i in range(len(test_genes)):
            col = mat[:, [i]].toarray().ravel() if hasattr(mat, "toarray") else np.asarray(mat[:, i])
            for group in groups:
                pvals[group].append(min(2 * min(wilcoxon_rank_sum_test_with_correlation(group_masks[group], col)), 1))
        pvals = pd.DataFrame(pvals, index=test_genes)
        pvals.index.name = "gene"
    else:
        pvals = wilcoxon_rank_sum_test(  # type: ignore
            ann,
            groupby=groupby,
            as_dataframe=True,
            tie_correction=tie_correction,
            layer=layer,
            genes=test_genes,
        )
    pvals = pd.melt(pvals, var_name=groupby, value_name="pval", ignore_index=False)  # type: ignore
    if pval_thresh is not None:
        pvals = pvals[pvals["pval"] < pval_thresh]  # type: ignore
    pvals.reset_index(inplace=True)  # type: ignore
    output = output.merge(pvals, on=["gene", groupby], how="inner")  # type: ignore
    # divide by amount of genes in AnnData object (not just features)
    output["pval_adj"] = np.clip(output["pval"] * len(ann.var_names), 0, 1)  # type: ignore # operation behaves as expected
    ann_orig.uns["group_metrics"] = output

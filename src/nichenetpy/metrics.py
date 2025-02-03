from nichenetpy.utils import subset_matrix
from nichenetpy.wilcoxon import wilcoxon_rank_sum_test

from anndata import AnnData
from collections.abc import Callable
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.metrics import precision_recall_curve

import numpy as np
import pandas as pd


def _auc_reverse(x:list[float], y:list[float]) -> float:
    '''
    Calculates the area under the curve using the trapezoid rule. The points should be specified in descending order of the x-values. 

    Parameters
    ----------
    x : list or tuple of float
        list of x-values on the curve
    y : list or tuple of float
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
    '''
    if len(x) != len(y):
        raise ValueError('x and y should have the same length')
    if len(x) < 2:
        raise ValueError('x and y should have a length of at least 2')
    return sum((x[i-1] - x[i])*(y[i] + y[i-1]) for i in range(1, len(x))) / 2

def calculate_aupr(response:list[float], prediction:list[int]) -> float:
    '''
    Calculates the area under the precision-recall curve using the trapezoid rule. 

    Parameters
    ----------
    response : list or tuple of float
        vector indicating whether a target is a True (1) target of the possibly active ligand(s) or a False (0)
    prediction : list or tuple of float
        vector which contains probability scores for each target gene (for one particular ligand)

    Returns
    -------
    float
        the area under the precision-recall curve

    Examples
    --------
    >>> calculate_aupr(
        (1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
        (0.5, 0.12, 0.47, 0.36, 0.0, 1.0, 0.78, 0.66, 0.24, 0.42, 0.95, 0.47, 0.39, 0.07)
    )
    0.8851473922902493
    '''
    precision, recall, _ = precision_recall_curve(response, prediction)
    return _auc_reverse(recall, precision)

def calculate_metrics(
    prediction:list[float],
    response:list[int]
) -> dict[str, float]:
    '''
    Calculates metrics that can be used to rank ligands. 
    currently supported metrics are:
        AUPR
        corrected AUPR

    Parameters
    ----------
    prediction : list or tuple of float
        vector which contains probability scores for each target gene (for one particular ligand)
    response : list or tuple of float
        vector indicating whether a target is a True (1) target of the possibly active ligand(s) or a False (0)

    Returns
    -------
    dict[float]
        dictionary with as keys the names of the supported metrics and as values the computed metrics
    '''
    aupr = calculate_aupr(response, prediction)
    return {
        "aupr": aupr,
        "aupr_corrected": aupr - sum(response)/len(response)
    }

def _sub_log_fold_change(
    data:csc_matrix|csr_matrix,
    denormalize:Callable=np.expm1,
    pseudocount:int=1
):
    if denormalize is not None:
        data = denormalize(data)
    return np.log2((data.sum(axis=0) + pseudocount) / data.shape[0])

def log_fold_change(
    mat1:np.ndarray|csc_matrix|csr_matrix,
    mat2:np.ndarray|csc_matrix|csr_matrix,
    denormalize:Callable=np.expm1,
    pseudocount:int=1
):
    return (
        _sub_log_fold_change(mat1, denormalize, pseudocount) - 
        _sub_log_fold_change(mat2, denormalize, pseudocount)
    ).transpose()

def gene_expression_pct(
    mat=np.ndarray|csc_matrix|csr_matrix
) -> list[float]:
    if type(mat) is csc_matrix or type(mat) is csr_matrix:
        nrows, ncols = mat.get_shape()
    elif type(mat) is np.ndarray:
        nrows, ncols = mat.shape
    else:
        raise TypeError(f"mat should be of type np.ndarray, scipy.csc_matrix or scipy.csr_matrix, not {type(mat)}")
    # set all non-zero elements to 1
    for i in range(len(mat.data)):
        mat.data[i] = 1
    output = mat.sum(axis=0) / nrows
    return [output[0, i] for i in range(ncols)]

def group_metrics(
    ann:AnnData,
    groupby:str,
    layer:str="data",
    lfc_pseudocount:int=1,
    tie_correction=True,
    min_pct=0.05
):
    if layer == "data":
        lfc_denormalize = np.expm1
    else:
        lfc_denormalize = None
    mat = ann.layers[layer]
    row2index = dict(zip(ann.obs.index, range(len(ann.obs.index))))
    groups = sorted(set(ann.obs[groupby]))
    lfc = []
    pct = []
    for group in groups:
        cells_oi = ann.obs[ann.obs[groupby] == group].index
        rest = ann.obs[ann.obs[groupby] != group].index
        mat1 = subset_matrix(mat, rows=[row2index[cell] for cell in cells_oi])
        mat2 = subset_matrix(mat, rows=[row2index[cell] for cell in rest])
        lfc.append(log_fold_change(mat1, mat2, lfc_denormalize, lfc_pseudocount))
        pct.append(gene_expression_pct(mat1))
    output = pd.melt(
        pd.DataFrame(np.concatenate(lfc, axis=1), index=ann.var_names, columns=groups),
        var_name=groupby,
        value_name="lfc",
        ignore_index=False
    )
    output.reset_index(inplace=True)
    pct = pd.melt(pd.DataFrame(pct, index=groups, columns=ann.var_names), value_name="pct", ignore_index=False)
    pct.index.name = groupby
    pct.reset_index(inplace=True)
    output = output.merge(pct, on=["gene", groupby], how="inner")
    pvals = wilcoxon_rank_sum_test(
        ann,
        groupby,
        as_dataframe=True,
        tie_correction=tie_correction,
        layer=layer,
        genes=pct[pct["pct"] >= min_pct]["gene"]
    )
    pvals = pd.melt(pvals, var_name=groupby, value_name="pval", ignore_index=False)
    pvals.reset_index(inplace=True)
    output = output.merge(pvals, on=["gene", groupby], how="inner")
    output["pval_adj"] = np.clip(output["pval"]*len(ann.var_names), 0, 1)
    ann.uns["group_metrics"] = output
    print(output)
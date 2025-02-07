from nichenetpy.utils import subset_matrix
from nichenetpy.wilcoxon import wilcoxon_rank_sum_test
from nichenetpy.ann_utils import _subset_layer

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
    pseudocount:float=1
) -> np.ndarray:
    '''
    Calculates the log fold changes the way seurat does it.

    Parameters
    ----------
    mat1 : numpy.ndarray or scipy.csc_matrix or scipy.csr_matrix
        the first matrix
    mat2 : numpy.ndarray or scipy.csc_matrix or scipy.csr_matrix
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
    '''
    return (
        _sub_log_fold_change(mat1, denormalize, pseudocount) - 
        _sub_log_fold_change(mat2, denormalize, pseudocount)
    ).transpose()

def gene_expression_pct(
    mat=np.ndarray|csc_matrix|csr_matrix
) -> list[float]:
    '''
    For each gene, calculate the percentage of cells that have an expression value greater than 0. 

    Parameters
    ----------
    mat : numpy.ndarray or scipy.csc_matrix or scipy.csr_matrix
        (#cells X #genes) matrix containing the expression values

    Returns
    -------
    list
        for each gene the percentage of cells that have an expression value greater than 0
    '''
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

def _single_group_metrics(
    ann,
    mat,
    row2index,
    lfc_denormalize,
    lfc_pseudocount,
    groupby,
    group_oi,
    group_ref=None
):
    cells_oi = ann.obs[ann.obs[groupby] == group_oi].index
    if group_ref is None:
        cells_ref = ann.obs[ann.obs[groupby] != group_oi].index
    else:
        cells_ref = ann.obs[ann.obs[groupby] == group_ref].index
    mat1 = subset_matrix(mat, rows=[row2index[cell] for cell in cells_oi])
    mat2 = subset_matrix(mat, rows=[row2index[cell] for cell in cells_ref])
    return (log_fold_change(mat1, mat2, lfc_denormalize, lfc_pseudocount), gene_expression_pct(mat1))

def group_metrics(
    ann:AnnData,
    groupby:str,
    group_oi:str=None,
    group_ref:str=None,
    layer:str="data",
    lfc_pseudocount:float=1,
    tie_correction:bool=True,
    features:list[str]=None,
    min_abs_lfc:float=0,
    min_pct:float=0,
    pval_thresh:float=None # 0.01 in seurat
):
    '''
    For each gene, calculate the percentage of cells that have an expression value greater than 0,
    the log fold changes and the p-values / adjusted p-values. 
    The result is a pandas dataframe stored in ann.uns["group_metrics"]

    Parameters
    ----------
    ann : AnnData
        the AnnData object
    groupby : str
        the column in ann.obs to group by
    group_oi : str
        the group of interest
    group_ref : str
        the reference group
    layer : str
        the layer in the AnnData object to use
    lfc_pseudocount : float
        the pseudocount to use in the computation of the log fold changes
    tie_correction : bool
        if True, tie correction is performed through averaging
    features : list of str
        the genes to consider
    min_lfc : float
        genes with a lfc lower than this value will be excluded from the wilcoxon rank sum test
    min_pct : float
        genes with a pct lower than this value will be excluded from the wilcoxon rank sum test
    pval_thresh : float
        upper bound for the p-values (if p_values for a gene is smaller than this threshold, it is excluded)
    
    Notes
    -----
    The result is the same as seurat's FindMarkers function.
    '''
    if layer == "data":
        lfc_denormalize = np.expm1
    else:
        lfc_denormalize = None
    if features is None:
        mat = ann.layers[layer]
        genes = ann.var_names
    else:
        mat, genes = _subset_layer(ann, layer, features)
    row2index = dict(zip(ann.obs.index, range(len(ann.obs.index))))
    groups = sorted(set(ann.obs[groupby]))
    lfc = []
    pct = []
    if group_oi is None:
        for group in groups:
            x, y = _single_group_metrics(
                ann,
                mat,
                row2index,
                lfc_denormalize,
                lfc_pseudocount,
                groupby,
                group,
                group_ref
            )
            lfc.append(x)
            pct.append(y)
    else:
        x, y = _single_group_metrics(
            ann,
            mat,
            row2index,
            lfc_denormalize,
            lfc_pseudocount,
            groupby,
            group_oi,
            group_ref
        )
        lfc.append(x)
        pct.append(y)
    output = pd.melt(
        pd.DataFrame(np.concatenate(lfc, axis=1), index=genes, columns=(groups if group_oi is None else [group_oi])),
        var_name=groupby,
        value_name="lfc",
        ignore_index=False
    )
    output.index.name = "gene"
    output.reset_index(inplace=True)
    pct = pd.melt(
        pd.DataFrame(pct, index=groups, columns=genes),
        value_name="pct",
        var_name="gene",
        ignore_index=False
    )
    pct.index.name = groupby
    pct.reset_index(inplace=True)
    output = output.merge(pct, on=["gene", groupby], how="inner")
    pvals = wilcoxon_rank_sum_test(
        ann,
        groupby=groupby,
        as_dataframe=True,
        tie_correction=tie_correction,
        layer=layer,
        genes=list(set(output[(output["pct"] >= min_pct) & (abs(output["lfc"]) >= min_abs_lfc)]["gene"]))
    )
    pvals = pd.melt(pvals, var_name=groupby, value_name="pval", ignore_index=False)
    if pval_thresh is not None:
        pvals = pvals[pvals["pval"] < pval_thresh]
    pvals.reset_index(inplace=True)
    output = output.merge(pvals, on=["gene", groupby], how="inner")
    # divide by amount of genes in AnnData object (not just features)
    output["pval_adj"] = np.clip(output["pval"]*len(ann.var_names), 0, 1)
    ann.uns["group_metrics"] = output
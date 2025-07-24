from nichenetpy.utils import subset_matrix
from nichenetpy.wilcoxon import (
    wilcoxon_rank_sum_test,
    wilcoxon_rank_sum_test_with_correlation
)
from nichenetpy.ann_utils import _subset_layer, subset_ann
from nichenetpy.typing import nichenet_matrix

from anndata import AnnData
from mudata import MuData
from collections.abc import Callable, Iterable
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_curve, roc_curve
from numbers import Number

import numpy as np
import pandas as pd
import warnings


def _auc_reverse(
    x:list[float]|tuple[float]|np.ndarray[float],
    y:list[float]|tuple[float]|np.ndarray[float]
) -> float:
    '''
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
    '''
    if len(x) != len(y):
        raise ValueError('x and y should have the same length')
    if len(x) < 2:
        raise ValueError('x and y should have a length of at least 2')
    if type(x) is np.ndarray and type(y) is np.ndarray:
        return ((x[:-1] - x[1:]) * (y[1:] + y[:-1])).sum() / 2
    return sum((x[i-1] - x[i])*(y[i] + y[i-1]) for i in range(1, len(x))) / 2

def calculate_aupr(
    response:list[float]|tuple[float]|np.ndarray[float],
    prediction:list[float]|tuple[float]|np.ndarray[float]
) -> float:
    '''
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
    '''
    if type(response) is not list and type(response) is not tuple and type(response) is not np.ndarray:
        raise TypeError(f"response should be a list or tuple or numpy.ndarray of floats, had type {type(response)}")
    if type(prediction) is not list and type(prediction) is not tuple and type(response) is not np.ndarray:
        raise TypeError(f"prediction should be a list or tuple or numpy.ndarray of floats, had type {type(prediction)}")
    if sum(response) == 0:
        warnings.warn("There are no true samples in response, AUPR is undefined")
        return np.nan
    precision, recall, _ = precision_recall_curve(response, prediction)
    return _auc_reverse(recall, precision)

def calculate_auroc(
    response:list[float]|tuple[float]|np.ndarray[float],
    prediction:list[float]|tuple[float]|np.ndarray[float]
) -> float:
    '''
    Calculates the area under the roc-curve using the trapezoid rule. 

    Parameters
    ----------
    response : list or tuple  or numpy.ndarray of float
        vector indicating whether a target is a True (1) target of the possibly active ligand(s) or a False (0)
    prediction : list or tuple  or numpy.ndarray of float
        vector which contains probability scores for each target gene (for one particular ligand)

    Returns
    -------
    float
        the area under the roc-curve
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(response) is not list and type(response) is not tuple and type(response) is not np.ndarray:
        raise TypeError(f"response should be a list or tuple or numpy.ndarray of floats, had type {type(response)}")
    if type(prediction) is not list and type(prediction) is not tuple and type(response) is not np.ndarray:
        raise TypeError(f"prediction should be a list or tuple or numpy.ndarray of floats, had type {type(prediction)}")
    fp, tp, _ = roc_curve(response, prediction)
    fp, tp = zip(*sorted(zip(fp, tp), key=lambda x : x[0]))
    return -_auc_reverse(fp, tp)

def calculate_prediction_evaluation_metrics(
    prediction:list[float]|tuple[float]|np.ndarray[float],
    response:list[float]|tuple[float]|np.ndarray[float]
) -> dict[str, float]:
    '''
    Calculates metrics that can be used to rank ligands. 
    currently supported metrics are:

        AUPR
        corrected AUPR
        AUROC
        Pearson correlation

    Parameters
    ----------
    prediction : list or tuple or numpy.ndarray of float
        vector which contains probability scores for each target gene (for one particular ligand)
    response : list or tuple or numpy.ndarray of float
        vector indicating whether a target is a True (1) target of the possibly active ligand(s) or a False (0)

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
    '''
    if type(response) is not list and type(response) is not tuple and type(response) is not np.ndarray:
        raise TypeError(f"response should be a list or tuple or numpy.ndarray of floats, had type {type(response)}")
    if type(prediction) is not list and type(prediction) is not tuple and type(response) is not np.ndarray:
        raise TypeError(f"prediction should be a list or tuple or numpy.ndarray of floats, had type {type(prediction)}")
    if sum(response) == 0:
        raise ValueError("There are no true samples in response. aupr, auroc and pearson correlation coëfficient are undefined.")
    aupr = calculate_aupr(response, prediction)
    auroc = calculate_auroc(response, prediction)
    pcc = pearsonr(response, prediction).statistic
    return {
        "auroc": auroc,
        "pearson": pcc,
        "aupr": aupr,
        "aupr_corrected": aupr - sum(response)/len(response)
    }

def _sub_log_fold_change(
    data:csc_matrix|csr_matrix,
    denormalize:Callable[[nichenet_matrix], nichenet_matrix]=np.expm1,
    pseudocount:int=1
):
    if denormalize is not None:
        data = denormalize(data)
    return np.log2((data.sum(axis=0) + pseudocount) / data.shape[0])

def log_fold_change(
    mat1:nichenet_matrix,
    mat2:nichenet_matrix,
    denormalize:Callable[[nichenet_matrix], nichenet_matrix]=np.expm1,
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

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if not isinstance(mat1, nichenet_matrix):
        raise TypeError(f"mat1 should have type numpy.ndarray, scipy.csc_matrix or scipy.csr_matrix, was {type(mat1)}")
    if not isinstance(mat2, nichenet_matrix):
        raise TypeError(f"mat2 should have type numpy.ndarray, scipy.csc_matrix or scipy.csr_matrix, was {type(mat2)}")
    if denormalize is not None and not isinstance(denormalize, Callable):
        raise TypeError(f"denormalize should be a Callable or None, had type {type(denormalize)}")
    if not isinstance(pseudocount, Number):
        raise TypeError(f"pseudocount should have type float, was {type(pseudocount)}")
    return (
        _sub_log_fold_change(mat1, denormalize, pseudocount) - 
        _sub_log_fold_change(mat2, denormalize, pseudocount)
    ).transpose()

def gene_expression_pct(
    mat=nichenet_matrix
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
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
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
    if len(cells_oi) == 0:
        raise RuntimeError("There are no cells in the group of interest")
    if len(cells_ref) == 0:
        raise RuntimeError("There are no cells in the reference group")
    mat1 = subset_matrix(mat, rows=[row2index[cell] for cell in cells_oi])
    mat2 = subset_matrix(mat, rows=[row2index[cell] for cell in cells_ref])
    return (log_fold_change(mat1, mat2, lfc_denormalize, lfc_pseudocount), gene_expression_pct(mat1))

def group_metrics(
    data:AnnData|MuData,
    groupby:str,
    group_oi:str|None=None,
    group_ref:str|None=None,
    layer:str="data",
    lfc_pseudocount:float=1,
    tie_correction:bool=True,
    features:Iterable[str]|None=None,
    min_abs_lfc:float=0,
    min_pct:float=0,
    pval_thresh:float|None=None, # 0.01 in seurat
    wilcoxon_limma:bool=False,
    lfc_denormalize:Callable[[nichenet_matrix], nichenet_matrix]|None=np.expm1,
    modality:str=None
):
    '''
    For each gene, calculate the percentage of cells that have an expression value greater than 0,
    the log fold changes and the p-values / adjusted p-values. 
    The result is a pandas dataframe stored in ann.uns["group_metrics"]

    Parameters
    ----------
    data : AnnData or MuData
        the AnnData or MuData object
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
        genes with a lfc lower than this value will be excluded from the wilcoxon rank sum test
    min_pct : float
        genes with a pct lower than this value will be excluded from the wilcoxon rank sum test
    pval_thresh : float or None
        upper bound for the p-values (if p_values for a gene is smaller than this threshold, it is excluded)
    wilcoxon_limma : bool
        use wilcoxon-limma (reproduces results from seuratv4)
    lfc_denormalize : Callable or None
        a denormalization function to apply prior to the calculation of the log fold changes
    modality : str
        the modality of the MuData object to use
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    RuntimeError
        if there are no cells in the group of interest
        if there are no cells in the reference group
    
    Notes
    -----
    The result is the same as seurat's FindMarkers function.
    '''
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
    if lfc_denormalize is not None and not isinstance(lfc_denormalize, Callable):
        raise TypeError(f"lfc_denormalize should be a Callable or None, had type {type(lfc_denormalize)}")
    if type(data) is MuData:
        if modality is None:
            raise ValueError("if MuData is used, modality needs to be provided")
        data = data.mod[modality]
    elif type(data) is not AnnData:
        raise TypeError(f"data should be of type AnnData or Mudata, was {type(data)}")
    if features is None:
        mat = data.layers[layer]
        genes = data.var_names
    else:
        mat, genes = _subset_layer(data, layer, features)
    row2index = dict(zip(data.obs.index, range(len(data.obs.index))))
    groups = sorted(set(data.obs[groupby]))
    lfc = []
    pct = []
    if group_oi is None:
        for group in groups:
            x, y = _single_group_metrics(
                data,
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
            data,
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
    ann_orig = data
    if group_oi is not None and group_ref is not None:
        data = subset_ann(data, val=(group_oi, group_ref), val_col=groupby)
    if wilcoxon_limma: # TODO: this will need serious optimization after verification that it works
        mat = data.layers[layer]
        groups = set(data.obs[groupby])
        pvals = {group: [] for group in groups}
        # for each gene
        for i in range(mat.shape[1]):
            col = mat[:, i].todense()
            for group in groups:
                pvals[group].append(
                    min(
                        2 * min(
                            wilcoxon_rank_sum_test_with_correlation(
                                data.obs[groupby] == group,
                                col
                            )
                        ),
                        1
                    )
                )
        pvals = pd.DataFrame(pvals, index=genes)
        pvals.index.name = "gene"
    else:
        pvals = wilcoxon_rank_sum_test(
            data,
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
    output["pval_adj"] = np.clip(output["pval"]*len(data.var_names), 0, 1)
    ann_orig.uns["group_metrics"] = output
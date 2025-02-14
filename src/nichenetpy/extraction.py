from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork
from nichenetpy.utils import subset_matrix
from nichenetpy.metrics import gene_expression_pct, group_metrics
from nichenetpy.ann_utils import subset_ann

from anndata import AnnData
from collections.abc import Iterable, Callable
from numbers import Number

import numpy as np
import pandas as pd


def get_expressed_genes(
    celltype:str|Iterable[str],
    ann:AnnData,
    pct:float=0.1,
    celltype_col:str="celltype",
    layer:str="data"
) -> list[str]:
    '''
    Gets the expressed genes from an AnnData object. 

    Parameters
    ----------
    celltype : str or Iterable of str
        the cell types to consider
    ann : AnnData
        the AnnData object to extract expressed genes from
    pct : float
        We consider genes expressed if they are expressed in at least a specific fraction of cells of the given cluster(s). 
        This number indicates this fraction. 
    celltype_col : str
        the name of the column in obs which contains the celltypes
    layer : str
        the name of the layer which contains the data matrix
    
    Returns
    -------
    list
        list of expressed genes
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(celltype) is str:
        celltype = [celltype]
    elif not isinstance(celltype, Iterable):
        raise TypeError(f"celltype should be a string or an Iterable of strings, was {type(celltype)}")
    if type(ann) is not AnnData:
        raise TypeError(f"ann should be of type AnnData, was {type(ann)}")
    if not isinstance(pct, Number):
        raise TypeError(f"pct should be of type float, was {type(pct)}")
    if type(celltype_col) is not str:
        raise TypeError(f"celltype_col should be of type str, was {type(celltype_col)}")
    if type(layer) is not str:
        raise TypeError(f"layer should be of type str, was {type(layer)}")
    cells_oi = list(ann.obs.loc[[ct in celltype for ct in ann.obs[celltype_col]]].index)
    # ncells x ngenes
    mat = ann.layers[layer]
    # select rows corresponding to cells of interest
    row2index = dict(zip(ann.obs.index, range(len(ann.obs.index))))
    exprs_m = subset_matrix(mat, rows=[row2index[name] for name in cells_oi])
    exps = gene_expression_pct(exprs_m)
    return [ann.var_names[gene] for gene, val in enumerate(exps) if val > pct]

def get_weighted_ligand_receptor_links(
    best_upstream_ligands:Iterable[str],
    expressed_receptors:Iterable[str],
    lr_network:LigandReceptorNetwork,
    lr_sig:WeightedNetwork
) -> WeightedNetwork:
    '''
    Get the weighted ligand-receptor links between a possible ligand and its receptors. 

    Parameters
    ----------
    best_upstream_ligands : Iterable of str
        the ligands of interest
    expressed_receptors : Iterable of str
        the receptors expressed in the cell type of interest
    lr_network : LigandReceptorNetwork
        the ligand-receptor network containing the ligand-receptor interactions
    lr_sig : WeightedNetwork
        a weighted network containing the ligand-receptor interactions and their weights
    
    Returns
    -------
    WeightedNetwork
        the weighted ligand-receptor links
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if not isinstance(best_upstream_ligands, Iterable):
        raise TypeError(f"best_upstream_ligands should be an Iterable of strings, was {type(best_upstream_ligands)}")
    elif type(best_upstream_ligands) is not set:
        best_upstream_ligands = set(best_upstream_ligands)
    if not isinstance(expressed_receptors, Iterable):
        raise TypeError(f"expressed_receptors should be an Iterable of strings, was {type(expressed_receptors)}")
    elif type(expressed_receptors) is not set:
        expressed_receptors = set(expressed_receptors)
    if type(lr_network) is not LigandReceptorNetwork:
        raise TypeError(f"lr_network should have type LigandReceptorNetwork, was {type(lr_network)}")
    if type(lr_sig) is not WeightedNetwork:
        raise TypeError(f"lr_sig should have type WeightedNetwork, was {type(lr_sig)}")
    lr_sig = lr_sig.subset(set(lr_network))
    best_upstream_receptors = set(t for f, t in lr_network if f in best_upstream_ligands and t in expressed_receptors)
    return lr_sig.subset_sep(best_upstream_ligands.intersection(set(e[0] for e in lr_network)), best_upstream_receptors)

def get_lfc_celltype(
    ann:AnnData,
    celltype:str,
    condition_col:str,
    condition_oi:str,
    layer:str,
    celltype_col:str="celltype",
    features:Iterable[str]=None
) -> tuple[list[str], list[float]]:
    '''
    Get log fold change of genes between two conditions in cell type of interest from an AnnData object.

    Parameters
    ----------
    ann : AnnData
        the AnnData object
    celltype : str
        the cell type of interest
    condition_col : str
        the name of the column in obs that contains the condition
    condition_oi : str
        the condition of interest
    layer : str
        the name of the data layer
    celltype_col : str
        the name of the column in obs that contains the cell types
    features : Iterable of str or None
        the genes to consider, consider all genes if None
    
    Returns
    -------
    list
        list of genes
    list
        list of log fold changes
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(ann) is not AnnData:
        raise TypeError(f"ann should be of type AnnData, was {type(ann)}")
    if type(celltype) is not str:
        raise TypeError(f"celltype should be of type str, was {type(celltype)}")
    if type(condition_col) is not str:
        raise TypeError(f"condition_col should be of type str, was {type(condition_col)}")
    if type(condition_oi) is not str:
        raise TypeError(f"condition_oi should be of type str, was {type(condition_oi)}")
    if type(layer) is not str:
        raise TypeError(f"layer should be of type str, was {type(layer)}")
    if type(celltype_col) is not str:
        raise TypeError(f"celltype_col should be of type str, was {type(celltype_col)}")
    if features is not None and not isinstance(features, Iterable):
        raise TypeError(f"features should be an Iterable of strings, was {type(features)}")
    ann_sender = subset_ann(
        ann,
        celltype,
        layers=[layer],
        val_col=celltype_col,
        genes=features
    )
    group_metrics(
        ann_sender,
        groupby=condition_col,
        layer=layer
    )
    res = ann_sender.uns["group_metrics"]
    pd.options.mode.chained_assignment = None # false positive warnings removal
    res = res[res[condition_col] == condition_oi]
    res.drop(columns={condition_col}, inplace=True)
    res.drop_duplicates(inplace=True)
    return (
        list(res["gene"]),
        list(res["lfc"])
    )

def average_expression(
    ann:AnnData,
    groupby:str,
    keys:Iterable[str]=None,
    layer:str="counts",
    norm_f:Callable=None
):
    '''
    Computes averaged expression values for each group. Similar to seurat's AverageExpression. 

    Parameters
    ----------
    ann : AnnData
        the AnnData object for which to compute averaged expression values
    groupby : str
        the column in ann.obs to group by
    keys : Iterable of str
        the values to group by, all values in the groupby column by default
    layer : str
        the layer to compute average expression values from, this layer should contain counts
    norm_f : Callable
        the normalization function (normalization prior to the computation)
    
    Returns
    -------
    dict
        a dictionary with the groups as keys and the lists of average expressions for each gene as values
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(ann) is not AnnData:
        raise TypeError(f"ann should be of type AnnData, was {type(ann)}")
    if type(groupby) is not str:
        raise TypeError(f"groupby should be of type str, was {type(groupby)}")
    if type(layer) is not str:
        raise TypeError(f"layer should be of type str, was {type(layer)}")
    if norm_f is not None:
        if not isinstance(norm_f, Callable):
            raise TypeError(f"norm_f should be of type Callable, was {type(norm_f)}")
        data = norm_f(ann.layers[layer])
    if keys is None:
        keys = set(ann.obs[groupby])
    elif not isinstance(keys, Iterable):
        raise TypeError(f"keys should be an Iterable of strings, was {type(keys)}")
    cell2id = dict(zip(ann.obs.index, range(len(ann.obs.index))))
    col_names, cols = zip(*(
        (key, data[[cell2id[cell] for cell in ann.obs.index if ann.obs.loc[cell][groupby] == key], :].mean(axis=0))
        for key in keys
    ))
    df = pd.DataFrame(np.column_stack([col.T for col in cols]))
    df.index = ann.var_names
    df.columns = col_names
    return df
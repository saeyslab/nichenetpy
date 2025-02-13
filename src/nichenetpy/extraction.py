from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork
from nichenetpy.utils import subset_matrix
from nichenetpy.metrics import gene_expression_pct, group_metrics
from nichenetpy.ann_utils import subset_ann, subset_ann_layer

from anndata import AnnData

import numpy as np
import pandas as pd


def get_expressed_genes(
    celltype:str|list[str],
    ann:AnnData,
    pct:float=0.1,
    celltype_col:str="celltype",
    layer:str="data"
) -> list[str]:
    '''
    Gets the expressed genes from an AnnData object. 

    Parameters
    ----------
    celltype : str or list of str
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
    '''
    if type(celltype) is str:
        celltype = [celltype]
    cells_oi = list(ann.obs.loc[[ct in celltype for ct in ann.obs[celltype_col]]].index)
    # ncells x ngenes
    mat = ann.layers[layer]
    # select rows corresponding to cells of interest
    row2index = dict(zip(ann.obs.index, range(len(ann.obs.index))))
    exprs_m = subset_matrix(mat, rows=[row2index[name] for name in cells_oi])
    exps = gene_expression_pct(exprs_m)
    return [ann.var_names[gene] for gene, val in enumerate(exps) if val > pct]

def get_weighted_ligand_receptor_links(
    best_upstream_ligands:list[str],
    expressed_receptors:list[str],
    lr_network:LigandReceptorNetwork,
    lr_sig:WeightedNetwork
) -> WeightedNetwork:
    '''
    Get the weighted ligand-receptor links between a possible ligand and its receptors. 

    Parameters
    ----------
    best_upstream_ligands : list of str
        the ligands of interest
    expressed_receptors : list of str
        the receptors expressed in the cell type of interest
    lr_network : LigandReceptorNetwork
        the ligand-receptor network containing the ligand-receptor interactions
    lr_sig : WeightedNetwork
        a weighted network containing the ligand-receptor interactions and their weights
    
    Returns
    -------
    WeightedNetwork
        the weighted ligand-receptor links
    '''
    lr_sig = lr_sig.subset(set(lr_network))
    best_upstream_ligands = set(best_upstream_ligands)
    expressed_receptors = set(expressed_receptors)
    best_upstream_receptors = set(t for f, t in lr_network if f in best_upstream_ligands and t in expressed_receptors)
    return lr_sig.subset_sep(best_upstream_ligands.intersection(set(e[0] for e in lr_network)), best_upstream_receptors)

def get_lfc_celltype(
    ann:AnnData,
    celltype:str,
    condition_col:str,
    condition_oi:str,
    layer:str,
    celltype_col:str="celltype",
    features:list[str]=None
) -> tuple[list[str], list[float]]:
    '''
    Get log fold change of genes between two conditions in cell type of interest from an AnnData object.

    Parameters
    ----------
    ann : AnnData
        the AnnData object
    celltype : str
        the cell type of interest
    condition_colname : str
        the name of the column in obs that contains the condition
    condition_oi : str
        the condition of interest
    layer : str
        the name of the data layer
    celltype_col : str
        the name of the column in obs that contains the cell types
    features : list of str or None
        the genes to consider, consider all genes if None
    
    Returns
    -------
    list
        list of genes
    list
        list of log fold changes
    '''
    ann_sender = subset_ann(ann, celltype, layers=[layer], val_col=celltype_col)
    if features is None:
        ann_sender.var_names = ann.var_names
    else:
        ann_sender = subset_ann_layer(ann_sender, layer, features)
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
    keys:list[str]=None,
    layer:str="counts",
    norm_f=None
):
    '''
    Computes averaged expression values for each group. Similar to seurat's AverageExpression. 

    Parameters
    ----------
    ann : AnnData
        the AnnData object for which to compute averaged expression values
    groupby : str
        the column in ann.obs to group by
    keys : list of str
        the values to group by, all values in the groupby column by default
    layer : str
        the layer to compute average expression values from, this layer should contain counts
    norm_f : function
        the normalization function (normalization prior to the computation)
    
    Returns
    -------
    dict
        a dictionary with the groups as keys and the lists of average expressions for each gene as values
    '''
    if norm_f is not None:
        data = norm_f(ann.layers[layer])
    if keys is None:
        keys = set(ann.obs[groupby])
    cell2id = dict(zip(ann.obs.index, range(len(ann.obs.index))))
    col_names, cols = zip(*(
        (key, data[[cell2id[cell] for cell in ann.obs.index if ann.obs.loc[cell][groupby] == key], :].mean(axis=0))
        for key in keys
    ))
    df = pd.DataFrame(np.column_stack([col.T for col in cols]))
    df.index = ann.var_names
    df.columns = col_names
    return df
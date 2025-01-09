from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork

from scipy.sparse import hstack, vstack, csc_matrix, csr_matrix
from anndata import AnnData

import scanpy as sc
import numpy as np
import pandas as pd


def get_expressed_genes(
    celltype:str|list[str],
    ann:AnnData,
    pct:float=0.1,
    celltype_col:str="celltype",
    layer:str="data",
    gene_field:str="gene"
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
        the minimum percent difference between the percent of cells expressing the gene in the cluster and the percent of cells expressing the gene in all other clusters combined. 
    celltype_col : str
        the name of the column in obs which contains the celltypes
    layer : str
        the name of the layer which contains the data matrix
    gene_field : str
        the name of the column in var which contains the gene symbols
    
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
    ids = [row2index[name] for name in cells_oi]
    exprs_m = vstack([mat[id, :] for id in ids])
    nrows = exprs_m.get_shape()[0]
    # set all non-zero elements to 1
    for i in range(len(exprs_m.data)):
        exprs_m.data[i] = 1
    rowsum = exprs_m.sum(axis=0)/nrows
    return [
        ann.var[gene_field].iloc[gene]
        for gene, val in enumerate(
            rowsum[0, i]
            for i in range(len(ann.var[gene_field]))
        )
        if val > pct
    ]

def subset_ann(
        ann:AnnData,
        val:str|list[str],
        layers:list[str]=None,
        val_col:str="celltype"
    ) -> AnnData|None:
    '''
    Subsets an AnnData object. 

    Parameters
    ----------
    ann : AnnData
        the AnnData object to subset
    val : str or list of str
        the values to subset by
    layers : list of str or None
        the layers to subset (additionally to subsetting obs), if None all layers are subsetted
        layers that aren't subsetted won't be present in the output
    val_col : str
        the name of the column in obs that contains the values
    
    Returns
    -------
    AnnData
        the subsetted AnnData object
    '''
    if layers is None:
        layers = list(ann.layers.keys())
    if type(val) is str:
        val = [val]
    cells_oi = ann.obs.loc[[ct in val for ct in ann.obs[val_col]]]
    if len(cells_oi) == 0:
        return None
    col2index = dict(zip(ann.obs.index, range(len(ann.obs.index))))
    ids = [col2index[name] for name in cells_oi.index]
    new_layers = dict(
        (
            layer,
            vstack([ann.layers[layer][id, :] for id in ids])
            if type(ann.layers[layer]) is csc_matrix or type(ann.layers[layer]) is csr_matrix
            else np.concatenate([[ann.layers[layer][id, :]] for id in ids])
        ) for layer in layers
    )
    output =  AnnData(
        obs=cells_oi,
        layers=new_layers,
        shape=new_layers[layers[0]].shape,
        var=ann.var,
        varm=ann.varm
    )
    output.var_names = ann.var_names
    return output

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
    condition_ref:str,
    layer:str,
    celltype_col:str="celltype",
    gene_field:str="gene",
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
    condition_ref : str
        the reference condition
    layer : str
        the name of the data layer
    celltype_col : str
        the name of the column in obs that contains the cell types
    gene_field : str
        the name of the column in var which contains the gene symbols
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
    if features is not None:
        gene2index = dict(zip(ann.var[gene_field], range(len(ann.var[gene_field]))))
        ids = [gene2index[gene] for gene in features]
        mat = ann_sender.layers[layer]
        mat = hstack([mat[:, id] for id in ids])
        ann_sender = AnnData(
            obs=ann_sender.obs,
            layers={"data": mat},
            shape=(ann_sender.obs.shape[0], len(features))
        )
        ann_sender.var_names = features
    else:
        ann_sender.var_names = ann.var[gene_field]
    sc.pp.log1p(ann_sender, layer=layer)
    sc.tl.rank_genes_groups(
        ann_sender,
        groupby=condition_col,
        method="wilcoxon",
        layer=layer,
        groups=[condition_oi],
        reference=condition_ref
    )
    return (
        [e[0] for e in ann_sender.uns["rank_genes_groups"]["names"]],
        [e[0] for e in ann_sender.uns["rank_genes_groups"]["logfoldchanges"]]
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
    df.index = ann.var["gene"]
    df.columns = col_names
    return df
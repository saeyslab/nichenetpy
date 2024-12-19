from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork

from scipy.sparse import hstack, vstack
from anndata import AnnData

import scanpy as sc


def get_expressed_genes(celltype:str|list[str], ann:AnnData, pct:float=0.1) -> list[int]:
    if type(celltype) is str:
        celltype = [celltype]
    cells_oi = list(ann.obs.loc[[ct in celltype for ct in ann.obs["celltype"]]].index)
    # ncells x ngenes
    mat = ann.layers["data"]
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
        ann.var["gene"].iloc[gene]
        for gene, val in enumerate(
            rowsum[0, i]
            for i in range(len(ann.var["gene"]))
        )
        if val > pct
    ]

def subset_ann_celltype(ann:AnnData, celltype:str|list[str], layers:list[str]=None, celltype_col:str="celltype"):
    if layers is None:
        layers = ann.layers.keys()
    cells_oi = ann.obs.loc[[ct in celltype for ct in ann.obs[celltype_col]]]
    col2index = dict(zip(ann.obs.index, range(len(ann.obs.index))))
    ids = [col2index[name] for name in cells_oi.index]
    new_layers = dict((layer, vstack([ann.layers[layer][id, :] for id in ids])) for layer in layers)
    return AnnData(
        obs=cells_oi,
        layers=new_layers,
        shape=new_layers[layers[0]].shape
    )

def get_weighted_ligand_receptor_links(
    best_upstream_ligands:list[str],
    expressed_receptors:list[str],
    lr_network:LigandReceptorNetwork,
    lr_sig:WeightedNetwork
) -> WeightedNetwork:
    lr_sig = lr_sig.subset(set(lr_network))
    best_upstream_ligands = set(best_upstream_ligands)
    expressed_receptors = set(expressed_receptors)
    best_upstream_receptors = set(t for f, t in lr_network if f in best_upstream_ligands and t in expressed_receptors)
    return lr_sig.subset_sep(best_upstream_ligands.intersection(set(e[0] for e in lr_network)), best_upstream_receptors)

def get_lfc_celltype(
    ann:AnnData,
    celltype:str,
    condition_colname:str,
    condition_oi:str,
    condition_ref:str,
    layer:str,
    celltype_coll:str="celltype",
    features:list[str]=None
) -> list[float]:
    ann_sender = subset_ann_celltype(ann, celltype, layers=[layer], celltype_col=celltype_coll)
    if features is not None:
        gene2index = dict(zip(ann.var["gene"], range(len(ann.var["gene"]))))
        ids = sorted(gene2index[gene] for gene in features)
        mat = ann_sender.layers[layer]
        mat = hstack([mat[:, id] for id in ids])
        ann_sender = AnnData(
            obs=ann_sender.obs,
            layers={"data": mat},
            shape=(ann_sender.obs.shape[0], len(features))
        )
        ann_sender.var_names = features
    else:
        ann_sender.var_names = ann.var["gene"]
    sc.pp.log1p(ann_sender, layer=layer)
    sc.tl.rank_genes_groups(
        ann_sender,
        groupby=condition_colname,
        method="wilcoxon",
        layer=layer,
        groups=[condition_oi],
        reference=condition_ref
    )
    return (ann_sender.uns["rank_genes_groups"]["names"], ann_sender.uns["rank_genes_groups"]["logfoldchanges"])
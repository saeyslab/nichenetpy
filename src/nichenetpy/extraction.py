from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork

from scipy.sparse import vstack
from anndata import AnnData


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

def subset_ann_celltype(ann:AnnData, celltype:str|list[str], layers:list[str]=None):
    if layers is None:
        layers = ann.layers.keys()
    cells_oi = ann.obs.loc[[ct in celltype for ct in ann.obs["celltype"]]]
    col2index = dict(zip(ann.obs.index, range(len(ann.obs.index))))
    ids = [col2index[name] for name in cells_oi.index]
    new_layers = dict((layer, vstack([ann.layers[layer][id, :] for id in ids])) for layer in layers)
    return AnnData(obs=cells_oi, layers=new_layers, shape=new_layers[layers[0]].shape)

def get_weighted_ligand_receptor_links(
    best_upstream_ligands:list[str],
    expressed_receptors:list[str],
    lr_network:LigandReceptorNetwork,
    lr_sig:WeightedNetwork
) -> WeightedNetwork:
    fr, to = zip(*lr_network)
    best_upstream_ligands = set(best_upstream_ligands)
    expressed_receptors = set(expressed_receptors)
    fr = set(fr).intersection(best_upstream_ligands)
    to = set(to).intersection(expressed_receptors)
    best_upstream_receptors = set(t for f, t, _ in lr_sig if f in fr and t in to)
    return lr_sig.subset(best_upstream_ligands, best_upstream_receptors)
from scipy.sparse import hstack
from anndata import AnnData


def get_expressed_genes(celltype:str|list[str], ann:AnnData, pct:float=0.1) -> list[int]:
    if type(celltype) is str:
        celltype = [celltype]
    cells_oi = list(ann.obs.loc[[ct in celltype for ct in ann.obs["celltype"]]].index)
    # transpose the matrix (remove this once the bug in anndataR is fixed)
    mat = ann.layers["data"].T
    col2index = dict(zip(ann.obs.index, range(len(ann.obs.index))))
    ids = [col2index[name] for name in cells_oi]
    exprs_m = hstack([mat[:, id] for id in ids])
    ncols = exprs_m.get_shape()[1]
    for i in range(len(exprs_m.data)):
        exprs_m.data[i] = 1
    return [ann.var["gene"].iloc[gene] for gene, val in enumerate(exprs_m.sum(axis=1)/ncols) if val > pct]

def subset_ann_celltype(ann:AnnData, celltype:str|list[str], layers:list[str]=None):
    if layers is None:
        layers = ann.layers.keys()
    cells_oi = ann.obs.loc[[ct in celltype for ct in ann.obs["celltype"]]]
    col2index = dict(zip(ann.obs.index, range(len(ann.obs.index))))
    ids = [col2index[name] for name in cells_oi.index]
    new_layers = dict()
    for layer in layers:
        # transpose the matrix (remove this once the bug in anndataR is fixed)
        mat = ann.layers[layer].T
        mat = hstack([mat[:, id] for id in ids])
        new_layers[layer] = mat.T
    return AnnData(obs=cells_oi, layers=new_layers, shape=new_layers[layers[0]].shape)
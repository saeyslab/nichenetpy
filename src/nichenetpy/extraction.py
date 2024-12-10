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
    enumerate((exprs_m.sum(axis=0)/nrows)[0, i] for i in range(len(ann.var["gene"])))
    return [
        ann.var["gene"].iloc[gene]
        for gene, val in enumerate(
            (exprs_m.sum(axis=0)/nrows)[0, i]
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
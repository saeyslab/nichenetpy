from scipy.sparse import hstack
from anndata import AnnData


def get_expressed_genes(celltype:str, ann:AnnData, pct:float=0.1) -> list[int]:
    cells_oi = list(ann.obs.loc[ann.obs["celltype"] == celltype].index)
    # transpose the matrix (remove this once the bug in anndataR is fixed)
    mat = ann.layers["data"].T
    col2index = dict(zip(ann.obs.index, range(len(ann.obs.index))))
    ids = [col2index[name] for name in cells_oi]
    exprs_m = hstack([mat[:, id] for id in ids])
    ncols = exprs_m.get_shape()[1]
    for i in range(len(exprs_m.data)):
        exprs_m.data[i] = 1
    return [ann.var["gene"].iloc[gene] for gene, val in enumerate(exprs_m.sum(axis=1)/ncols) if val > pct]
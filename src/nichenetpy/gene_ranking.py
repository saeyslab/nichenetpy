from anndata import AnnData
from collections.abc import Callable
from scipy.sparse import vstack, csc_matrix, csr_matrix

import numpy as np


def _sub_log_fold_change(
    data:csc_matrix|csr_matrix,
    denormalize:Callable=np.expm1,
    pseudocount:int=1
):
    if denormalize is not None:
        data = denormalize(data)
    return np.log2((data.sum(axis=0) + pseudocount) / data.shape[1])

def log_fold_change(
    ann:AnnData,
    groupby:str,
    denormalize:Callable=np.expm1,
    pseudocount:int=1
):
    row2index = dict(zip(ann.obs.index, range(len(ann.obs.index))))
    groups = set(ann.obs[groupby])
    for group in groups:
        cells_oi = ann.obs[ann.obs[groupby] == group].index
        rest = ann.obs[ann.obs[groupby] != group].index
        mat1 = vstack([ann.layers["data"][row2index[cell], :] for cell in cells_oi])
        mat2 = vstack([ann.layers["data"][row2index[cell], :] for cell in rest])
        lfc = _sub_log_fold_change(mat1, denormalize, pseudocount) - _sub_log_fold_change(mat1, denormalize, pseudocount)
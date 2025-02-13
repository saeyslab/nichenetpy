from nichenetpy.utils import subset_matrix

from anndata import AnnData
from collections.abc import Iterable

import numpy as np


def _subset_layer(
    ann:AnnData,
    layer:str,
    features:Iterable[str],
    gene2index:dict[str, int]=None
) -> tuple[np.ndarray, list[str]]:
    if gene2index is None:
        gene2index = dict(zip(ann.var_names, range(len(ann.var_names))))
    if type(features) is set:
        features = sorted(features)
    ids = [gene2index[gene] for gene in features]
    return (subset_matrix(ann.layers[layer], cols=ids), features)

def subset_ann(
    ann:AnnData,
    val:str|Iterable[str]=None,
    genes:Iterable[str]=None,
    layers:Iterable[str]=None,
    val_col:str="celltype"
) -> AnnData|None:
    '''
    Subsets the cells and/or genes of an AnnData object. 

    Parameters
    ----------
    ann : AnnData
        the AnnData object to subset
    val : str or Iterable of str
        the values to subset by
    genes : Iterable of str
        the genes to select
    layers : Iterable of str or None
        the layers to subset (additionally to subsetting obs), if None all layers are subsetted
        layers that aren't subsetted won't be present in the output
    val_col : str
        the name of the column in obs that contains the values to subset by
    
    Returns
    -------
    AnnData
        the subsetted AnnData object
    '''
    if layers is None:
        layers = list(ann.layers.keys())
    elif not isinstance(layers, Iterable):
        raise TypeError(f"layers should be a string or an Iterable of strings, was {type(layers)}")
    if val is None:
        row_ids = None
    else:
        if type(val) is str:
            val = {val}
        elif type(val) is not set:
            val = set(val)
        cells_oi = ann.obs.loc[[ct in val for ct in ann.obs[val_col]]]
        if len(cells_oi) == 0:
            row_ids = None
        else:
            col2index = dict(zip(ann.obs.index, range(len(ann.obs.index))))
            row_ids = [col2index[name] for name in cells_oi.index]
    if genes is None:
        col_ids = None
    else:
        if type(genes) is set:
            genes = sorted(genes)
        gene2index = dict(zip(ann.var_names, range(len(ann.var_names))))
        col_ids = [gene2index[gene] for gene in genes]
    new_layers = dict(
        (
            layer,
            subset_matrix(ann.layers[layer], rows=row_ids, cols=col_ids)
        ) for layer in layers
    )
    output = AnnData(
        obs=ann.obs if row_ids is None else cells_oi,
        layers=new_layers,
        shape=new_layers[layers[0]].shape
    )
    if genes is None:
        output.var_names = ann.var_names
    else:
        output.var_names = ann.var_names.reindex(genes)[0]
    output.var_names.name = "gene"
    return output
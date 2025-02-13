from nichenetpy.utils import subset_matrix

from anndata import AnnData
from collections.abc import Iterable

import numpy as np


def subset_ann(
    ann:AnnData,
    val:str|list[str],
    layers:list[str]=None,
    val_col:str="celltype"
) -> AnnData|None:
    '''
    Subsets the cells of an AnnData object. 

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
            subset_matrix(ann.layers[layer], rows=ids)
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

def _subset_layer(
    ann:AnnData,
    layer:str,
    features:list[str]
) -> tuple[np.ndarray, list[str]]:
    if type(features) is set:
        features = sorted(features)
    gene2index = dict(zip(ann.var_names, range(len(ann.var_names))))
    ids = [gene2index[gene] for gene in features]
    return (subset_matrix(ann.layers[layer], cols=ids), features)

def subset_ann_layer(
    ann:AnnData,
    layers:str|Iterable[str],
    features:list[str]
) -> AnnData:
    '''
    Subsets genes in a layer of an AnnData object. 

    Parameters
    ----------
    ann : AnnData
        the AnnData object to subset
    layers : str or Iterable of str
        the layers to subset
    features : list of str
        the genes to select
    
    Returns
    -------
    AnnData
        the subsetted AnnData object (only contains the subsetted layers)
    
    Raises
    ------
    TypeError
        if arguments have the wrong type
    '''
    if type(features) is set:
        features = sorted(features)
    if type(layers) is str:
        layers = [layers]
    elif not isinstance(layers, Iterable):
        raise TypeError(f"layers should be a string or an Iterable of strings, was {type(layers)}")
    new_layers = dict(
        (
            layer,
            _subset_layer(ann, layer=layer, features=features)[0]
        ) for layer in layers
    )
    ann = AnnData(
        obs=ann.obs,
        layers=new_layers,
        shape=(ann.obs.shape[0], len(features))
    )
    ann.var_names = ann.var_names.reindex(features)[0]
    ann.var_names.name = "gene"
    return ann
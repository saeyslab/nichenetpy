from nichenetpy.utils import subset_matrix

from anndata import AnnData
from collections.abc import Iterable

import numpy as np


def _subset_layer(
    ann:AnnData,
    layer:str,
    features:Iterable[str],
    gene2index:dict[str, int]|None=None
) -> tuple[np.ndarray, list[str]]:
    if gene2index is None:
        gene2index = dict(zip(ann.var_names, range(len(ann.var_names))))
    if type(features) is set:
        features = sorted(features)
    ids = [gene2index[gene] for gene in features]
    return (subset_matrix(ann.layers[layer], cols=ids), features)

def subset_ann(
    ann:AnnData,
    val:str|Iterable[str]|None=None,
    genes:Iterable[str]|None=None,
    layers:Iterable[str]|None=None,
    val_col:str="celltype"
) -> AnnData|None:
    '''
    Subsets the cells and/or genes of an AnnData object. 

    Parameters
    ----------
    ann : AnnData
        the AnnData object to subset
    val : str or Iterable of str or None
        the values to subset by
    genes : Iterable of str or None
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
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if the arguments are invalid
    '''
    if type(ann) is not AnnData:
        raise TypeError(f"ann should be of type AnnData, was {type(ann)}")
    if val_col is not None and type(val_col) is not str:
        raise TypeError(f"val_col should be of type str, was {type(val_col)}")
    if genes is not None and not isinstance(genes, Iterable):
        raise TypeError(f"genes should be an Iterable of strings, was {type(genes)}")
    if layers is None:
        layers = list(ann.layers.keys())
    elif not isinstance(layers, Iterable):
        raise TypeError(f"layers should be an Iterable of strings, was {type(layers)}")
    if val is None:
        row_ids = None
    else:
        if type(val) is str:
            val = {val}
        elif not isinstance(val, Iterable):
            raise TypeError(f"val should be a string or an Iterable of strings, was {type(val)}")
        elif type(val) is not set:
            val = set(val)
        try:
            cells_oi = ann.obs.loc[[ct in val for ct in ann.obs[val_col]]]
        except KeyError:
            raise ValueError(f"There is no column '{val_col}' in the AnnData object")
        if len(cells_oi) == 0:
            raise ValueError(f"'{val}' not present in the column '{val_col}' of the AnnData object")
        else:
            col2index = dict(zip(ann.obs.index, range(len(ann.obs.index))))
            row_ids = [col2index[name] for name in cells_oi.index]
    if genes is None:
        col_ids = None
    else:
        if type(genes) is set:
            genes = sorted(genes)
        gene2index = dict(zip(ann.var_names, range(len(ann.var_names))))
        try:
            col_ids = [gene2index[gene] for gene in genes]
        except KeyError as error:
            raise ValueError(f"The gene '{error.args[0]}' is not present in the AnnData object")
    if row_ids is None and col_ids is None:
        return None
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

def prepare_ann(
    ann:AnnData
):
    '''
    Makes sure the AnnData object is suitable for a nichenet analysis. 

    Parameters
    ----------
    ann : AnnData
        the AnnData object to subset
    
    Raises
    ------
    TypeError
        if the AnnData object has the wrong type
    ValueError
        if the AnnData object is not suitable for a nichenet analysis and it is not possible to fix the issues
    '''
    if type(ann) is not AnnData:
        raise TypeError(f"ann should have type anndata.AnnData, was {type(ann)}")
    if ann.var_names is None:
        if "gene" in ann.var:
            ann.var_names = ann.var["gene"]
        else:
            raise ValueError("var_names and var['gene'] are both missing from the AnnData object")
    elif "gene" not in ann.var:
        ann.var["gene"] = ann.var_names
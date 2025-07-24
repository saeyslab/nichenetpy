from nichenetpy.utils import subset_matrix

from mudata import MuData
from anndata import AnnData
from collections.abc import Iterable

import numpy as np
import pandas as pd


def subset_mu(
    mdata:MuData,
    val:str|Iterable[str]|None=None,
    modality_layers:dict[str, Iterable[str]|None]|None=None,
    val_col:str="celltype"
) -> MuData|None:
    '''
    Subsets the cells and/or genes of an AnnData object. 

    Parameters
    ----------
    ann : AnnData
        the AnnData object to subset
    val : str or Iterable of str or None
        the values to subset by
    modality_layers : dict of Iterable of str or None
        for each modality: the layers to subset (additionally to subsetting obs), if None all layers of all modalities are subsetted
        layers that aren't subsetted won't be present in the output
    val_col : str
        the name of the column in obs that contains the values to subset by
    
    Returns
    -------
    MuData
        the subsetted MuData object
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if the arguments are invalid
    '''
    if type(mdata) is not MuData:
        raise TypeError(f"ann should be of type MuData, was {type(mdata)}")
    if val_col is not None and type(val_col) is not str:
        raise TypeError(f"val_col should be of type str, was {type(val_col)}")
    if modality_layers is None:
        modality_layers = {
            modality: list(mdata.mod[modality].layers.keys())
            for modality in mdata.mod.keys()
        }
    elif type(modality_layers) is not dict:
        raise TypeError(f"layers should be a dictionary, was {type(modality_layers)}")
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
            cells_oi = mdata.obs.loc[[ct in val for ct in mdata.obs[val_col]]]
        except KeyError:
            raise ValueError(f"There is no column '{val_col}' in the AnnData object")
        if len(cells_oi) == 0:
            raise ValueError(f"'{val}' not present in the column '{val_col}' of the AnnData object")
        else:
            col2index = dict(zip(mdata.obs.index, range(len(mdata.obs.index))))
            row_ids = [col2index[name] for name in cells_oi.index]
    if row_ids is None:
        return None
    modality_new_layers = {
        modality: dict(
            (
                layer,
                subset_matrix(mdata.mod[modality].layers[layer], rows=row_ids)
            ) for layer in layers
        ) if layers is not None and len(layers) > 0 else dict()
        for modality, layers in modality_layers.items()
    }
    anns = {
        modality: AnnData(
            obs=mdata.mod[modality].obs if row_ids is None else mdata.mod[modality].obs.iloc[row_ids],
            layers=new_layers,
            shape=(len(row_ids), mdata.mod[modality].shape[1])
        )
        for modality, new_layers in modality_new_layers.items()
    }
    for modality, ann in anns.items():
        ann.var_names = mdata.mod[modality].var_names
        ann.var_names.name = "gene"
    return MuData(anns)
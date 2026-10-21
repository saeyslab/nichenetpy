from collections.abc import Iterable
from typing import overload

import pandas as pd
import numpy as np
from anndata import AnnData
from scipy.sparse import csc_matrix, csr_matrix

from nichenetpy.typing import gene_t, nichenet_matrix
from nichenetpy.utils import subset_matrix


def _subset_layer(
    ann: AnnData, layer: str, features: Iterable[gene_t], gene2index: dict[gene_t, int] | None = None
) -> tuple[nichenet_matrix, Iterable[gene_t]]:
    sel_layer = ann.layers[layer]
    if (
        not isinstance(sel_layer, np.ndarray)
        and not isinstance(sel_layer, csr_matrix)
        and not isinstance(sel_layer, csc_matrix)
    ):
        raise TypeError(f"the selected layer should be of type nichenet_matrix, was {type(sel_layer)}")
    if gene2index is None:
        gene2index = dict(zip(ann.var_names, range(len(ann.var_names)), strict=True))
    if type(features) is set:
        features = sorted(features)
    ids = [gene2index[gene] for gene in features]
    return (subset_matrix(sel_layer, cols=ids), features)


@overload
def subset_ann(
    ann: AnnData,
    val: gene_t | Iterable[gene_t],
    genes: Iterable[gene_t] | None = None,
    layers: Iterable[str | None] | None = None,
    val_col: str = "celltype",
    subset_X: bool = True,
) -> AnnData: ...


@overload
def subset_ann(
    ann: AnnData,
    val: gene_t | Iterable[gene_t] | None = None,
    *,
    genes: Iterable[gene_t],
    layers: Iterable[str | None] | None = None,
    val_col: str = "celltype",
    subset_X: bool = True,
) -> AnnData: ...


@overload
def subset_ann(  # if val and genes are not provided, it should be this overload
    ann: AnnData,
    val: None = None,
    genes: None = None,
    layers: Iterable[str | None] | None = None,
    val_col: str = "celltype",
    subset_X: bool = True,
) -> None: ...


def subset_ann(
    ann: AnnData,
    val: gene_t | Iterable[gene_t] | None = None,
    genes: Iterable[gene_t] | None = None,
    layers: Iterable[str | None] | None = None,
    val_col: str = "celltype",
    subset_X: bool = True,
) -> AnnData | None:
    """
    Subsets the cells and/or genes of an AnnData object.

    Parameters
    ----------
    ann : AnnData
        the AnnData object to subset
    val : str or Iterable of str or None
        the values to subset by
    genes : Iterable of gene_t or None
        the genes to select
    layers : Iterable of str or None
        the layers to subset (additionally to subsetting obs), if None all layers are subsetted
        layers that aren't subsetted won't be present in the output
    val_col : str
        the name of the column in obs that contains the values to subset by
    subset_X : bool
        whether or not to subset X

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
    """
    if type(ann) is not AnnData:
        raise TypeError(f"ann should be of type AnnData, was {type(ann)}")
    if not isinstance(ann.obs, pd.DataFrame):
        raise TypeError(f"ann.obs should be of type pandas.DataFrame, was {type(ann.obs)}")
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
        if type(val) is gene_t:
            val = {val}  # type: ignore # val has type set[gene_t] which satisfies Iterable[gene_t]
        elif not isinstance(val, Iterable):
            raise TypeError(f"val should be of type gene_t or an Iterable of gene_t, was {type(val)}")
        elif type(val) is not set:
            val = set(val)
        try:
            cells_oi = ann.obs.loc[ann.obs[val_col].isin(val)]  # type: ignore # val has type set[gene_t]
        except KeyError as error:
            raise ValueError(f"There is no column '{val_col}' in the AnnData object") from error
        if len(cells_oi) == 0:
            raise ValueError(f"'{val}' not present in the column '{val_col}' of the AnnData object")
        else:
            row2index = dict(zip(ann.obs.index, range(len(ann.obs.index)), strict=True))
            row_ids = [row2index[name] for name in cells_oi.index]
    if genes is None:
        col_ids = None
    else:
        if type(genes) is set:
            genes = sorted(genes)
        gene2index = dict(zip(ann.var_names, range(len(ann.var_names)), strict=True))
        try:
            col_ids = [gene2index[gene] for gene in genes]
        except KeyError as error:
            raise ValueError(f"The gene '{error.args[0]}' is not present in the AnnData object") from error
    if row_ids is None and col_ids is None:
        return None
    new_layers = {layer: subset_matrix(ann.layers[layer], rows=row_ids, cols=col_ids) for layer in layers}  # type: ignore # let subset_matrix return TypeError
    if ann.X is not None:
        new_X = subset_matrix(ann.X, rows=row_ids, cols=col_ids) if subset_X else ann.X  # type: ignore # let subset_matrix return TypeError
    # subset categories if the column is categorical
    if row_ids is not None and cells_oi[val_col].dtype.name == "category":
        pd.options.mode.chained_assignment = None  # false positive warning removal
        cells_oi[val_col] = cells_oi[val_col].cat.set_categories(val)  # type: ignore # val has type set[gene_t]
    if len(new_layers) > 0:
        new_shape = new_layers[next(iter(layers))].shape
    elif ann.X is not None and subset_X:
        new_shape = new_X.shape
    else:
        raise ValueError("if layers=[] then X must be defined and subset_X must be True")
    output = AnnData(obs=ann.obs if row_ids is None else cells_oi, layers=new_layers, shape=new_shape)  # type: ignore # despite the type warning this behaves as expected
    if genes is None:
        output.var_names = ann.var_names  # type: ignore # despite the type warning this behaves as expected
    else:
        output.var_names = ann.var_names.reindex(genes)[0]  # type: ignore # despite the type warning this behaves as expected
    output.var_names.name = "gene"
    if ann.X is not None:
        output.X = new_X
    return output


def prepare_ann(ann: AnnData):
    """
    Makes sure the AnnData object is suitable for a nichenet analysis.

    Parameters
    ----------
    ann : AnnData
        the AnnData object to subset

    Raises
    ------
    TypeError
        if the AnnData object has the wrong type
    AnnError
        if the AnnData object is not suitable for a nichenet analysis and it is not possible to fix the issues
    """
    if type(ann) is not AnnData:
        raise TypeError(f"ann should have type anndata.AnnData, was {type(ann)}")
    if ann.var_names is None:
        # nichenetpy expects the gene names to be stored here
        if "gene" in ann.var:
            ann.var_names = ann.var["gene"]  # type: ignore # despite the type warning this behaves as expected
        else:
            raise ValueError("var_names and var['gene'] are both missing from the AnnData object")
    elif "gene" not in ann.var:
        ann.var["gene"] = ann.var_names

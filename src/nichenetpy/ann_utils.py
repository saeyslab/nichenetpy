from collections.abc import Iterable, Sequence
from typing import Any, overload

import numpy as np
import pandas as pd
from anndata import AnnData

from nichenetpy.typing import GeneSymbol, _as_gene, _check_iterable, nichenet_matrix, validate_matrix_type
from nichenetpy.utils import subset_matrix


def _as_gene_sequence(genes: Iterable[GeneSymbol]) -> Sequence[GeneSymbol]:
    # the genes are used more than once and as a sequence (their order, len, a pandas index): a set is sorted (it has
    # no fixed order), anything else that isn't a sequence (e.g. a generator) is stored in a list
    if isinstance(genes, (set, frozenset)):
        return sorted(genes)
    if isinstance(genes, (Sequence, np.ndarray, pd.Index)):
        return genes  # type: ignore # a numpy array or pandas Index of genes behaves as a Sequence
    return list(genes)


def _subset_layer(
    ann: AnnData, layer: str, features: Iterable[GeneSymbol], gene2index: dict[GeneSymbol, int] | None = None
) -> tuple[nichenet_matrix, Sequence[GeneSymbol]]:
    sel_layer = ann.layers[layer]
    validate_matrix_type(sel_layer, f"the selected layer should be of type nichenet_matrix, was {type(sel_layer)}")
    if gene2index is None:
        gene2index = dict(zip(ann.var_names, range(len(ann.var_names)), strict=True))
    features = _as_gene_sequence(features)
    ids = [gene2index[gene] for gene in features]
    return (subset_matrix(sel_layer, cols=ids), features)  # type: ignore # sel_layer is type checked


@overload
def subset_ann(
    ann: AnnData,
    val: str | Iterable[str],
    genes: Iterable[GeneSymbol] | None = None,
    layers: Iterable[str | None] | None = None,
    val_col: str = "celltype",
    subset_X: bool = True,
) -> AnnData: ...


@overload
def subset_ann(
    ann: AnnData,
    val: str | Iterable[str] | None = None,
    *,
    genes: Iterable[GeneSymbol],
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
    val: str | Iterable[str] | None = None,
    genes: Iterable[GeneSymbol] | None = None,
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
        the layers to subset, if None all layers are subsetted
        layers that aren't subsetted won't be present in the output
    val_col : str
        the name of the column in obs that contains the values to subset by
    subset_X : bool
        whether or not to subset X; if False, the output has no X (this saves the time and memory of subsetting X when
        only the layers are needed)

    Returns
    -------
    AnnData or None
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
    if genes is not None:
        _check_iterable(genes, "genes", "gene symbols")
    if layers is None:
        layers = [key for key in ann.layers.keys() if key is not None]
    else:
        _check_iterable(layers, "layers", "layer names")
    # the selected cells (None if the cells aren't subsetted) and the selected values of val_col
    cells_oi: pd.DataFrame | None = None
    selected_vals: set[Any] = set()
    if val is None:
        row_ids = None
    else:
        # a single value or an Iterable of values
        if isinstance(val, str) or not isinstance(val, Iterable):
            selected_vals = {_as_gene(val, "val")}
        else:
            selected_vals = set(val)
        try:
            # the positions of the selected cells (by position, a cell name can occur more than once)
            row_ids = np.flatnonzero(ann.obs[val_col].isin(selected_vals).to_numpy())
        except KeyError as error:
            raise ValueError(f"There is no column '{val_col}' in the AnnData object") from error
        if len(row_ids) == 0:
            raise ValueError(f"'{val}' not present in the column '{val_col}' of the AnnData object")
        # copy, the categories of val_col are changed below (on a view this changed ann.obs or gave a warning)
        cells_oi = ann.obs.iloc[row_ids].copy()
    if genes is None:
        col_ids = None
    else:
        # used twice: for the columns and the var_names
        genes = _as_gene_sequence(genes)
        gene2index = dict(zip(ann.var_names, range(len(ann.var_names)), strict=True))
        try:
            col_ids = [gene2index[gene] for gene in genes]
        except KeyError as error:
            raise ValueError(f"The gene '{error.args[0]}' is not present in the AnnData object") from error
    if row_ids is None and col_ids is None:
        return None
    new_layers = {layer: subset_matrix(ann.layers[layer], rows=row_ids, cols=col_ids) for layer in layers}  # type: ignore # let subset_matrix return TypeError
    # X is only kept if it's subsetted (the original X doesn't fit the subsetted shape)
    new_X = None
    if ann.X is not None and subset_X:
        new_X = subset_matrix(ann.X, rows=row_ids, cols=col_ids)  # type: ignore # let subset_matrix return TypeError
    # subset categories if the column is categorical
    if cells_oi is not None and cells_oi[val_col].dtype.name == "category":
        # keep the selected categories in their original order (val is a set, which has no fixed order)
        categories = [cat for cat in cells_oi[val_col].cat.categories.tolist() if cat in selected_vals]
        cells_oi[val_col] = cells_oi[val_col].cat.set_categories(categories)
    if len(new_layers) > 0:
        # the shape of one of the subsetted layers (layers itself is only iterated once, it can be a generator)
        new_shape = next(iter(new_layers.values())).shape
    elif new_X is not None:
        new_shape = new_X.shape
    else:
        raise ValueError("if layers=[] then X must be defined and subset_X must be True")
    output = AnnData(obs=ann.obs if cells_oi is None else cells_oi, layers=new_layers, shape=new_shape)  # type: ignore # despite the type warning this behaves as expected
    if genes is None:
        output.var_names = ann.var_names  # pyright: ignore[reportAttributeAccessIssue] # despite the type warning this behaves as expected
    else:
        output.var_names = ann.var_names.reindex(genes)[0]  # pyright: ignore[reportAttributeAccessIssue] # despite the type warning this behaves as expected
    output.var_names.name = "gene"
    if new_X is not None:
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
            ann.var_names = ann.var["gene"]  # pyright: ignore[reportAttributeAccessIssue] # despite the type warning this behaves as expected
        else:
            raise ValueError("var_names and var['gene'] are both missing from the AnnData object")
    elif "gene" not in ann.var:
        ann.var["gene"] = ann.var_names

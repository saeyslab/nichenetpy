from __future__ import annotations  # Index[T] typehint can throw an Error at runtime

import os
import pickle
from collections.abc import Callable, Iterable, Sequence
from re import Pattern, compile, finditer, search, split
from typing import Any, overload

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from nichenetpy.typing import GeneSymbol, _combination_name, nichenet_matrix

_default_row_name_pattern = compile(r"^\"*([^\"]+)\"*,")
_default_col_name_pattern = compile(r",\"*([^\"]+)\"*")
_row_name_split = lambda s, i: (s[:i], s[i:])


def read_list_from_csv(filename: str) -> list[str]:
    """
    Reads a sequence of strings from a csv file.

    Parameters
    ----------
    filename : str
        the name of the csv file to read from

    Returns
    -------
    list
        list of read strings

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    with open(filename) as file:
        lines = file.readlines()
    return [line.rstrip().strip("\"'") for line in lines[1:]]


def read_matrix_from_csv(
    filename: str,
    extract_row_name: Callable[[str], tuple[str, str]] | Pattern[str] | None = None,
    extract_col_name: Callable[[str], list[str]] | Pattern[str] | None = None,
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], list[str], list[str]]:
    """
    Reads a matrix from a csv file.

    Parameters
    ----------
    filename : str
        the name of the csv file to read from
    extract_row_name : Callable or re.Pattern or None
        function which extracts the row name and the rest of the string from a line or a search pattern; if both
        extract_row_name and extract_col_name are None the file is read with pandas.read_csv (quoted or unquoted csv,
        "NA" and empty values become NaN)
    extract_col_name : Callable or re.Pattern or None
        function which extracts the col names from the first line or a search pattern

    Returns
    -------
    ndarray
        the matrix
    list
        the row labels
    list
        the column labels

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    if extract_row_name is None and extract_col_name is None:
        # pandas handles quoted and unquoted fields, commas in quoted names, a header without a cell for the row names
        # and Windows line endings; everything is read as str so that names stay as they are (e.g. a gene "NA" or "1")
        df = pd.read_csv(filename, index_col=0, dtype=str, keep_default_na=False)
        values = df.to_numpy()
        # missing values (NA in R, empty cells) become NaN
        values = np.where(np.isin(values, ("NA", "")), "nan", values).astype(np.float64)
        return (values, df.index.tolist(), df.columns.tolist())
    if extract_row_name is None:
        extract_row_name = _default_row_name_pattern
    if isinstance(extract_row_name, Pattern):
        row_name_pattern = extract_row_name
        extract_row_name = lambda x: _row_name_split(x, search(row_name_pattern, x).end())  # type: ignore # if search returns None then runtime error
    if not isinstance(extract_row_name, Callable):  # type: ignore
        raise TypeError(f"extract_row_name should have type Callable, re.Pattern or None, was {type(extract_row_name)}")
    if extract_col_name is None:
        extract_col_name = _default_col_name_pattern
    if isinstance(extract_col_name, Pattern):
        col_name_pattern = extract_col_name
        extract_col_name = lambda x: [x[slice(*m.span(1))] for m in finditer(col_name_pattern, x)]
    if not isinstance(extract_col_name, Callable):  # type: ignore
        raise TypeError(f"extract_col_name should have type Callable, re.Pattern or None, was {type(extract_col_name)}")
    with open(filename) as file:
        lines: Sequence[str] = file.readlines()
    col_names = extract_col_name(lines[0].rstrip())
    row_names, lines = zip(*(extract_row_name(line) for line in lines[1:]), strict=True)
    rows = [[float(e.strip("\"'")) for e in line.strip(",").rstrip().split(",")] for line in lines]
    return (np.array(rows, dtype=np.float64), [row_name[:-1].strip("'\"") for row_name in row_names], col_names)


def read_csv_rows(filename: str, sep: str | Pattern = ",") -> tuple[list[str], list[list[str]]]:
    """
    Reads the rows from a csv file.

    Parameters
    ----------
    filename : str
        the name of the csv file to read from
    sep : str or re.Pattern
        the separator

    Returns
    -------
    list of str
        list of column names
    list of list of str
        list of rows

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    with open(filename) as file:
        lines = file.readlines()
    lines = [[word.strip("\"'") for word in split(sep, line.rstrip())] for line in lines]  # type: ignore
    return (lines[0], lines[1:])  # type: ignore


def read_csv_cols(filename: str, sep: str | Pattern = ",") -> dict[str, Sequence[str]]:
    """
    Reads the columns from a csv file.

    Parameters
    ----------
    filename : str
        the name of the csv file to read from
    sep : str or re.Pattern
        the separator

    Returns
    -------
    dict
        mapping of column names to columns (lists of strings)

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    with open(filename) as file:
        lines = file.readlines()
    lines = [[word.strip("\"'") for word in split(sep, line.rstrip())] for line in lines]  # type: ignore
    return dict(zip(lines[0], zip(*lines[1:], strict=True), strict=True))


def read_network_file(filename: str):
    """
    Reads a network (pandas.DataFrame) from a csv or pickle file

    Parameters
    ----------
    filename : str
        the name of the file to read from

    Returns
    -------
    pandas.DataFrame
        the network as a data frame

    Raises
    ------
    ValueError
        if the file name does not have an expected extension
    """
    file_ext = (os.path.split(filename)[1]).split(".")[-1]
    if file_ext == "csv":
        return pd.DataFrame(read_csv_cols(filename))
    elif file_ext == "pkl" or file_ext == "pickle":
        with open(filename, "rb") as file:
            output = pickle.loads(file.read())
        return output
    else:
        raise ValueError(f"filename should have csv, pkl or pickle extension, was {file_ext}")


def _index_array(
    idx: Sequence[int | bool]
    | np.ndarray[Any, np.dtype[np.integer] | np.dtype[np.bool_]]
    | pd.Index[int]
    | pd.Index[bool],
    size: int,
    name: str,
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    # integer indices, from indices or a boolean mask of the axis (of length size)
    arr = np.asarray(idx).reshape(-1)
    if arr.size == 0:
        # np.asarray([]) has a float dtype
        return np.empty(0, dtype=np.intp)
    if arr.dtype == np.bool_:
        if arr.size != size:
            raise IndexError(f"{name} is a boolean mask of length {arr.size}, the axis has length {size}")
        return np.flatnonzero(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} should be integer indices or a boolean mask, had dtype {arr.dtype}")
    return arr.astype(np.intp, copy=False)


@overload
def subset_matrix(
    mat: np.ndarray,
    rows: Sequence[int | bool]
    | np.ndarray[Any, np.dtype[np.integer] | np.dtype[np.bool_]]
    | pd.Index[int]
    | pd.Index[bool]
    | None = None,
    cols: Sequence[int | bool]
    | np.ndarray[Any, np.dtype[np.integer] | np.dtype[np.bool_]]
    | pd.Index[int]
    | pd.Index[bool]
    | None = None,
) -> np.ndarray: ...


@overload
def subset_matrix(
    mat: csr_matrix[np.floating] | csc_matrix[np.floating],
    rows: Sequence[int | bool]
    | np.ndarray[Any, np.dtype[np.integer] | np.dtype[np.bool_]]
    | pd.Index[int]
    | pd.Index[bool]
    | None = None,
    cols: Sequence[int | bool]
    | np.ndarray[Any, np.dtype[np.integer] | np.dtype[np.bool_]]
    | pd.Index[int]
    | pd.Index[bool]
    | None = None,
) -> csr_matrix[np.floating] | csc_matrix[np.floating]: ...


@overload
def subset_matrix(
    mat: csr_array | csc_array,
    rows: Sequence[int | bool]
    | np.ndarray[Any, np.dtype[np.integer] | np.dtype[np.bool_]]
    | pd.Index[int]
    | pd.Index[bool]
    | None = None,
    cols: Sequence[int | bool]
    | np.ndarray[Any, np.dtype[np.integer] | np.dtype[np.bool_]]
    | pd.Index[int]
    | pd.Index[bool]
    | None = None,
) -> csr_array | csc_array: ...


def subset_matrix(
    mat: nichenet_matrix,
    rows: Sequence[int | bool]
    | np.ndarray[Any, np.dtype[np.integer] | np.dtype[np.bool_]]
    | pd.Index[int]
    | pd.Index[bool]
    | None = None,
    cols: Sequence[int | bool]
    | np.ndarray[Any, np.dtype[np.integer] | np.dtype[np.bool_]]
    | pd.Index[int]
    | pd.Index[bool]
    | None = None,
) -> nichenet_matrix:
    """
    Subsets a matrix.

    Parameters
    ----------
    mat : numpy.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        the matrix to subset
    rows : None or Sequence, numpy.ndarray or pandas.Index of int or bool
        row indices to keep or booleans indicating which rows to keep
    cols : None or Sequence, numpy.ndarray or pandas.Index of int or bool
        column indices to keep or booleans indicating which columns to keep

    Returns
    -------
    numpy.ndarray, scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
        the subsetted matrix

    Raises
    ------
    TypeError
        if the arguments have the wrong type, or the indices aren't integers or booleans
    IndexError
        if a boolean mask doesn't have the length of its axis

    Note
    ----
    if the rows of a numpy.ndarray are subsetted, the output is row-major
    if the columns of a numpy.ndarray are subsetted and the rows are not being subsetted, the output is column-major
    """
    if rows is not None and (isinstance(rows, str) or not isinstance(rows, (Sequence, np.ndarray, pd.Index))):
        raise TypeError(f"rows should be of type Sequence, numpy.ndarray or pandas.Index, was {type(rows)}")
    if cols is not None and (isinstance(cols, str) or not isinstance(cols, (Sequence, np.ndarray, pd.Index))):
        raise TypeError(f"cols should be of type Sequence, numpy.ndarray or pandas.Index, was {type(cols)}")
    if rows is None and cols is None:
        return mat
    if rows is not None:
        rows = _index_array(rows, mat.shape[0], "rows")
    if cols is not None:
        cols = _index_array(cols, mat.shape[1], "cols")
    if isinstance(mat, np.ndarray):
        if rows is None:
            output = mat[:, cols]
        else:
            output = mat[np.ix_(rows, np.arange(mat.shape[1]) if cols is None else cols)]
    elif isinstance(mat, (csc_matrix, csc_array, csr_matrix, csr_array)):
        output = mat
        if rows is not None:
            output = output[rows, :]
        if cols is not None:
            output = output[:, cols]
    else:
        raise TypeError(f"mat needs to be of type nichenet_matrix, not {type(mat)}")
    return output


def remove_zero_rows_cols(mat: np.ndarray[Any, np.dtype[np.floating]]) -> np.ndarray[Any, np.dtype[np.floating]]:
    """
    Remove all 0-rows and 0-columns from a matrix.

    Parameters
    ----------
    mat : numpy.ndarray
        the matrix to remove all 0-rows and 0-columns from

    Returns
    -------
    numpy.ndarray
        the subsetted matrix
    """
    return subset_matrix(mat, mat.any(axis=1), mat.any(axis=0))


def combine_by_key[T1, T2](*args: tuple[list[T1], list[T2]]) -> dict[T1, list[T2]]:
    """
    Combine tuples of lists matched by the keys in the first list of each passed tuple.

    Parameters
    ----------
    *args : tuple
        tuples of equally large lists to combine, where the first list of a tuple contains the keys

    Returns
    -------
    dict
        a dictionary where each key is mapped to a list containing one element per input tuple in the order the tuples were passed to the function

    Examples
    --------
    >>> combine_by_key(
        (["a", "b", "c", "d"], [4, 1, 8, 5]),
        (["b", "c", "d", "a"], [7, 3, 2, 1]),
        (["b", "a", "c", "d"], [9, 6, 5, 3])
    )
    {
        "a": [4, 1, 6],
        "b": [1, 7, 9],
        "c": [8, 3, 5],
        "d": [5, 2, 3]
    }
    """
    output: dict[T1, list[T2]] = {}
    for arg in args:
        for key, val in zip(arg[0], arg[1], strict=True):
            if key in output:
                output[key].append(val)
            else:
                output[key] = [val]
    return output


def combine_dicts(dict1: dict, dict2: dict, func: Callable | None = None) -> dict:
    """
    Combine two dictionaries by their mutual keys.

    Parameters
    ----------
    dict1 : dict
        one of the dictionaries to combine
    dict2 : dict
        one of the dictionaries to combine
    func : Callable or None
        a binary function that computes the new value from the old values
        if None, the values are combined into a tuple

    Returns
    -------
    dict
        a dictionary where each key is mapped to a tuple containing the mappings of the input dictionaries for that key,
        or alternatively a function is applied which combines the mappings

    Raises
    ------
    TypeError
        if the arguments have the wrong type

    Examples
    --------
    >>> combine_dicts(
        {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3
        },
        {
            "a": 4,
            "b": 5,
            "c": 6,
            "d": 7
        }
    )
    {
        "a": (0, 4),
        "b": (1, 5),
        "c": (2, 6),
        "d": (3, 7)
    }

    >>> combine_dicts(
        {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3
        },
        {
            "a": 4,
            "b": 5,
            "c": 6,
            "e": 7
        }
    )
    {
        "a": (0, 4),
        "b": (1, 5),
        "c": (2, 6)
    }
    """
    if func is None:
        func = lambda x, y: (x, y)
    if type(dict1) is not dict:
        raise TypeError(f"dict1 should have type dict, was {type(dict1)}")
    if type(dict2) is not dict:
        raise TypeError(f"dict2 should have type dict, was {type(dict2)}")
    return {key: func(dict1[key], dict2[key]) for key in set(dict1.keys()).intersection(set(dict2.keys()))}


def ligand_activities_df(
    ligand_activities: dict[GeneSymbol, dict[str, float]] | Iterable[tuple[GeneSymbol, dict[str, float]]],
) -> pd.DataFrame:
    """
    Convert ligand activities to a pandas DataFrame

    Parameters
    ----------
    ligand_activities : dict or Iterable
        the ligand activities to convert

    Returns
    -------
    pandas.DataFrame
        a pandas DataFrame containing the ligand activities

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if type(ligand_activities) is dict:
        ligands, activities = zip(*ligand_activities.items(), strict=True)
    elif isinstance(ligand_activities, Iterable):
        ligands, activities = zip(*ligand_activities, strict=True)
    else:
        raise TypeError(f"ligand_activities should be of type dict or Iterable, was {type(ligand_activities)}")
    columns = tuple(activities[0].keys())
    data = [tuple(act.values()) for act in activities]
    df = pd.DataFrame(data=data, index=ligands, columns=columns)
    df.index.name = "ligand"
    return df


def df_grouped_apply(df: pd.DataFrame, groupby: str, func: Callable, dest: str) -> pd.DataFrame:
    """
    Apply a function to groups of a dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        the dataframe
    groupby : str
        the column to group by
    func : Callable
        the function to apply
    dest : str
        the column to store the output in

    Returns
    -------
    pandas.DataFrame
        the result of the function application

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if type(df) is not pd.DataFrame:
        raise TypeError(f"df should have type pandas.DataFrame, was {type(df)}")
    if type(groupby) is not str:
        raise TypeError(f"groupby should have type str, was {type(groupby)}")
    if not isinstance(func, Callable):  # type: ignore
        raise TypeError(f"func should have type Callable, was {type(func)}")
    if type(dest) is not str:
        raise TypeError(f"dest should have type str, was {type(dest)}")
    vals = sorted(set(df[groupby]))
    groups = []
    for val in vals:
        # copy, so the new column is added to a dataframe of its own and not to a view of df
        group = df[df[groupby] == val].copy()
        group[dest] = func(group)
        groups.append(group)
    return pd.concat(groups)


def ncycle(it: Iterable, n: int) -> Iterable:
    """
    A combination of itertools.repeat and itertools.cycle,
    iterate over it n times.

    Parameters
    ----------
    it : Iterable
        the iterable
    n : int
        the amount of times to cycle through the iterable

    Returns
    -------
    Iterable
        it chained n times

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if not isinstance(it, Iterable):
        raise TypeError(f"it should have type Iterable, was {it}")
    if type(n) is not int:
        raise TypeError(f"n should have type int, was {type(n)}")
    for _ in range(n):
        yield from it


def decomplexify(df: pd.DataFrame, from_col: str = "ligand", to_col: str = "receptor") -> pd.DataFrame:
    """
    Helper Function to 'decomplexify' ligands and receptors into individual subunits. (function from LIANA R)

    Splits "from" and "to" in subunits and takes all combinations.

    Parameters
    ----------
    df : pandas.DataFrame
        data frame which has columns from_col and to_col
    from_col : str
        the "from" column
    to_col : str
        the "to" column

    Returns
    -------
    pandas.DataFrame
        data frame which has columns from_col and to_col after which contain all combinations of subunits

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if the arguments are invalid
    """
    if type(df) is not pd.DataFrame:
        raise TypeError(f"df should have type pandas.DataFrame, was {type(df)}")
    if type(from_col) is not str:
        raise TypeError(f"from_col should have type str, was {type(from_col)}")
    if type(to_col) is not str:
        raise TypeError(f"to_col should have type str, was {type(to_col)}")
    if from_col not in df.columns:
        raise ValueError(f"There is no column '{from_col}' in the data frame")
    if to_col not in df.columns:
        raise ValueError(f"There is no column '{to_col}' in the data frame")
    frs = []
    tos = []
    for fr, to in zip(df[from_col].tolist(), df[to_col].tolist(), strict=True):
        for nfr in fr.split("_"):
            for nto in to.split("_"):
                frs.append(nfr)
                tos.append(nto)
    return pd.DataFrame({from_col: frs, to_col: tos})


def rank_genes_groups_to_dataframe(ann: AnnData, groupby: str) -> pd.DataFrame:
    """
    Converts the output of scanpy.tl.rank_genes_groups to a pandas data frame (ann.uns["rank_genes_groups"] is not
    changed).

    Parameters
    ----------
    ann : AnnData
        the AnnData object (ann.uns["rank_genes_groups"] needs to be defined by using scanpy.tl.rank_genes_groups
        with pts=True)
    groupby : str
        the "groupby" argument that was passed to scanpy.tl.rank_genes_groups

    Returns
    -------
    pandas.DataFrame
        the data frame which contains the output of scanpy.tl.rank_genes_groups: one row per group and gene, with
        the columns groupby, "gene", "pval", "pval_adj", "lfc", "pct" (the fraction of expressing cells in the
        group) and "pct_ref" (the fraction of expressing cells in the reference cells: the other cells, or the
        reference group)

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if the arguments are invalid
    """
    if type(ann) is not AnnData:
        raise TypeError(f"ann should have type anndata.AnnData, was {type(ann)}")
    if groupby not in ann.obs.columns:
        raise ValueError(f"There is no column '{groupby}' in the AnnData object")
    res = ann.uns["rank_genes_groups"]
    output = pd.melt(pd.DataFrame(res["names"]), var_name=groupby, value_name="gene")
    for col in ["pvals", "pvals_adj", "logfoldchanges"]:
        temp = pd.melt(pd.DataFrame(res[col]), var_name=groupby, value_name=col)
        temp = temp.drop(columns={groupby})
        output = output.join(temp, how="inner")
    temp = pd.melt(res["pts"], var_name=groupby, value_name="pts", ignore_index=False)
    temp.index.name = "gene"
    temp = temp.reset_index()
    output = output.merge(temp, on=["gene", groupby], how="inner")
    # the fraction of expressing cells in the reference cells: scanpy stores it per group for the other cells
    # (pts_rest), and in the column of the reference group of pts for a reference group
    reference = res["params"]["reference"]
    if reference == "rest":
        pts_ref = res["pts_rest"]
    else:
        pts_ref = pd.DataFrame({group: res["pts"][reference] for group in res["pts"].columns if group != reference})
    temp = pd.melt(pts_ref, var_name=groupby, value_name="pct_ref", ignore_index=False)
    temp.index.name = "gene"
    temp = temp.reset_index()
    output = output.merge(temp, on=["gene", groupby], how="inner")
    output = output.rename(columns={"logfoldchanges": "lfc", "pvals": "pval", "pvals_adj": "pval_adj", "pts": "pct"})
    # ann.uns["rank_genes_groups"] is left as it is (scanpy's own functions, e.g. its plots, need its format)
    return output


def get_ties[T1, T2](it: Iterable[T1], key: Callable[[T1], T2]) -> dict[T2, list[int]]:
    """
    Group the indices of the tied elements of an Iterable.

    Parameters
    ----------
    it : Iterable
        the values to get the ties from
    key : Callable
        function to call on elements of it which returns the values of interest

    Returns
    -------
    dict
        a dictionary which maps tied values to groups of indices whose corresponging values equal the key

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    if not isinstance(it, Iterable):
        raise TypeError(f"it should have type Iterable, was {type(it)}")
    if not isinstance(key, Callable):  # type: ignore
        raise TypeError(f"key should have type Callable, was {type(key)}")
    output: dict[T2, list[int]] = {}
    for i, v in enumerate(it):
        v = key(v)
        if v in output:
            output[v].append(i)
        else:
            output[v] = [i]  # type: ignore
    return output


def is_ligand_active(importances: pd.DataFrame) -> list[bool]:
    """
    Returns a list of booleans indicating whether a ligand is active or not.

    Parameters
    ----------
    importances : pandas.DataFrame
        a data frame which has at least "test_ligand" and "true_ligand" columns

    Returns
    -------
    list
        a list of booleans indicating whether a ligand is active or not
    """
    return [
        test_ligand == true_ligand
        if isinstance(true_ligand, str) or not isinstance(true_ligand, Iterable)
        else test_ligand in true_ligand or test_ligand == _combination_name(true_ligand)
        for test_ligand, true_ligand in zip(importances["test_ligand"], importances["true_ligand"], strict=True)
    ]


def linked_iter[T](start: T, nxt: Callable[[T], T], stp: Callable[[T, T], bool]):
    """
    Yields all values reached by calling nxt on the current value until stp returns True

    Parameters
    ----------
    start : Any
        the starting value
    nxt : Callable
        function that returns the next value
    stp : Callable
        function that takes the current and next element as input and decides when to stop

    Yields
    ------
    Any
        all encountered values
    """
    cur = start
    _nxt = nxt(cur)
    while not stp(cur, _nxt):
        yield cur
        cur = _nxt
        _nxt = nxt(cur)

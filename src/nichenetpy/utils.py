from nichenetpy.typing import nichenet_matrix

from scipy.sparse import hstack, vstack, csc_matrix, csr_matrix
from collections.abc import Iterable, Callable
from anndata import AnnData
from re import search

import numpy as np
import pandas as pd


def read_list_from_csv(filename:str) -> list[str]:
    '''
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
    '''
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    with open(filename) as file:
        lines = file.readlines()
    return [line.rstrip().strip("\"\'") for line in lines[1:]]

def read_matrix_from_csv(filename:str) -> tuple[np.ndarray, list[str], list[str]]:
    '''
    Reads a matrix from a csv file. 

    Parameters
    ----------
    filename : str
        the name of the csv file to read from
    
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
    '''
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    with open(filename) as file:
        lines = file.readlines()
    lines = [[word.strip("\"\'") for word in line.rstrip().split(",")] for line in lines]
    col_names = lines[0][1:]
    row_names = []
    rows = []
    for line in lines[1:]:
        row_names.append(line[0])
        rows.append([float(e) for e in line[1:]])
    return (np.array(rows, dtype=np.float64), row_names, col_names)

def read_csv_rows(filename:str) -> tuple[list[str], list[list[str]]]:
    '''
    Reads the rows from a csv file. 

    Parameters
    ----------
    filename : str
        the name of the csv file to read from
    
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
    '''
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    with open(filename) as file:
        lines = file.readlines()
    lines = [[word.strip("\"\'") for word in line.rstrip().split(",")] for line in lines]
    return (lines[0], lines[1:])

def read_csv_cols(filename:str) -> dict[str, list[str]]:
    '''
    Reads the columns from a csv file. 

    Parameters
    ----------
    filename : str
        the name of the csv file to read from
    
    Returns
    -------
    dict
        mapping of column names to columns (lists of strings)

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    with open(filename) as file:
        lines = file.readlines()
    lines = [[word.strip("\"\'") for word in line.rstrip().split(",")] for line in lines]
    return dict(zip(lines[0], zip(*lines[1:])))

def subset_matrix(
    mat:nichenet_matrix,
    rows:list[int|bool]|tuple[int|bool]|np.ndarray|None=None,
    cols:list[int|bool]|tuple[int|bool]|np.ndarray|None=None
) -> nichenet_matrix:
    '''
    Subsets a matrix. 

    Parameters
    ----------
    mat : numpy.ndarray or scipy.csc_matrix or scipy.csr_matrix
        the matrix to subset
    rows : None or list or tuple of int or bool
        list of row indices to keep or list of booleans indicating which rows to keep
    cols : None or list or tuple of int or bool
        list of column indices to keep or list of booleans indicating which columns to keep
    
    Returns
    -------
    numpy.ndarray, scipy.csc_matrix or scipy.csr_matrix
        the subsetted matrix

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if rows is not None and type(rows) is not tuple and type(rows) is not list and type(rows) is not np.ndarray:
        raise TypeError(f"rows should be of type list, tuple or numpy.ndarray, was {type(rows)}")
    if cols is not None and type(cols) is not tuple and type(cols) is not list and type(rows) is not np.ndarray:
        raise TypeError(f"cols should be of type list or tuple or numpy.ndarray, was {type(cols)}")
    if rows is None and cols is None:
        return mat
    if rows is not None and (type(rows[0]) is bool or type(rows[0]) is np.bool):
        rows = [i for i, e in enumerate(rows) if e]
    if cols is not None and (type(cols[0]) is bool or type(cols[0]) is np.bool):
        cols = [i for i, e in enumerate(cols) if e]
    if type(mat) is np.ndarray:
        if rows is None:
            if cols is None:
                output = mat
            else:
                output = mat[:, cols]
        else:
            output = mat[
                [[row] for row in rows],
                [col for col in (list(range(mat.shape[1])) if cols is None else cols)]
            ]
    elif type(mat) is csc_matrix:
        if cols is None: # rows is not None
            output = csr_matrix(mat)
        else:
            output = hstack([mat[:, col] for col in cols], format="csc" if rows is None else "csr")
        if rows is not None:
            output = vstack([output[row, :] for row in rows], format="csc")
    elif type(mat) is csr_matrix:
        if rows is None: # cols is not None
            output = csc_matrix(mat)
        else:
            output = vstack([mat[row, :] for row in rows], format="csr" if cols is None else "csc")
        if cols is not None:
            output = hstack([output[:, col] for col in cols], format="csr")
    else:
        raise TypeError(f"mat needs to be of type numpy.ndarray, scipy.csc_matrix or scipy.csr_matrix, not {type(mat)}")
    return output

def remove_zero_rows_cols(mat:np.ndarray) -> np.ndarray:
    '''
    Remove all 0-rows and 0-columns from a matrix. 

    Parameters
    ----------
    mat : numpy.ndarray
        the matrix to remove all 0-rows and 0-columns from
    
    Returns
    -------
    numpy.ndarray
        the subsetted matrix
    '''
    return subset_matrix(mat, mat.any(axis=1), mat.any(axis=0))

def combine_by_key(*args:tuple[list[str], list]) -> dict[str, list]:
    '''
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
    '''
    output = dict()
    for arg in args:
        for key, val in zip(arg[0], arg[1]):
            if key in output:
                output[key].append(val)
            else:
                output[key] = [val]
    return output

def combine_dicts(
    dict1:dict,
    dict2:dict,
    func:Callable|None=None
) -> dict:
    '''
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
    '''
    if func is None:
        func = lambda x, y : (x, y)
    if type(dict1) is not dict:
        raise TypeError(f"dict1 should have type dict, was {type(dict1)}")
    if type(dict2) is not dict:
        raise TypeError(f"dict2 should have type dict, was {type(dict2)}")
    return dict((key, func(dict1[key], dict2[key])) for key in set(dict1.keys()).intersection(set(dict2.keys())))

def ligand_activities_df(
    ligand_activities:dict[str, dict[str, float]]|Iterable[tuple[str, dict[str, float]]]
) -> pd.DataFrame:
    '''
    convert ligand activities to a pandas DataFrame

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
    '''
    if type(ligand_activities) is dict:
        ligands, activities = zip(*ligand_activities.items())
    elif isinstance(ligand_activities, Iterable):
        ligands, activities = zip(*ligand_activities)
    else:
        raise TypeError(f"ligand_activities should be of type dict or Iterable, was {type(ligand_activities)}")
    columns = tuple(activities[0].keys())
    data = [tuple(act.values()) for act in activities]
    df = pd.DataFrame(data=data, index=ligands, columns=columns)
    df.index.name = "ligand"
    return df

def df_grouped_apply(
    df:pd.DataFrame,
    groupby:str,
    func:Callable,
    dest:str
) -> pd.DataFrame:
    '''
    apply a function to groups of a dataframe

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
    '''
    if type(df) is not pd.DataFrame:
        raise TypeError(f"df should have type pandas.DataFrame, was {type(df)}")
    if type(groupby) is not str:
        raise TypeError(f"groupby should have type str, was {type(groupby)}")
    if not isinstance(func, Callable):
        raise TypeError(f"func should have type Callable, was {type(func)}")
    if type(dest) is not str:
        raise TypeError(f"dest should have type str, was {type(dest)}")
    pd.options.mode.chained_assignment = None # false positive warnings removal
    vals = sorted(set(df[groupby]))
    groups = []
    for val in vals:
        group = df[df[groupby] == val]
        group[dest] = func(group)
        groups.append(group)
    return pd.concat(groups)

def ncycle(
    it:Iterable,
    n:int
) -> Iterable:
    '''
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
    '''
    if not isinstance(it, Iterable):
        raise TypeError(f"it should have type Iterable, was {it}")
    if type(n) is not int:
        TypeError(f"n should have type int, was {n}")
    for _ in range(n):
        yield from it

def decomplexify(
    df:pd.DataFrame,
    from_col:str="ligand",
    to_col:str="receptor"
) -> pd.DataFrame:
    '''
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
    '''
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
    for fr, to in zip(df[from_col], df[to_col]):
        for nfr in fr.split("_"):
            for nto in to.split("_"):
                frs.append(nfr)
                tos.append(nto)
    return pd.DataFrame({
        from_col: frs,
        to_col: tos
    })

def rank_genes_groups_to_dataframe(
    ann:AnnData,
    groupby:str
) -> pd.DataFrame:
    '''
    Converts the output of scanpy.tl.rank_genes_groups to a pandas data frame. 

    Parameters
    ----------
    ann : AnnData
        the AnnData object (ann.uns["rank_genes_groups"] needs to be defined by using scanpy.tl.rank_genes_groups)
    groupby : str
        the "groupby" argument that was passed to scanpy.tl.rank_genes_groups

    Returns
    -------
    pandas.DataFrame
        the data frame which contains the output of scanpy.tl.rank_genes_groups
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if the arguments are invalid
    '''
    if type(ann) is not AnnData:
        raise TypeError(f"ann should have type anndata.AnnData, was {type(ann)}")
    if groupby not in ann.obs.columns:
        raise ValueError(f"There is no column '{groupby}' in the AnnData object")
    res = ann.uns["rank_genes_groups"]
    output = pd.melt(pd.DataFrame(res["names"]), var_name=groupby, value_name="gene")
    for col in ["pvals", "pvals_adj", "logfoldchanges"]:
        temp = pd.melt(pd.DataFrame(res[col]), var_name=groupby, value_name=col)
        temp.drop(columns={groupby}, inplace=True)
        output = output.join(temp, how="inner")
    temp = pd.melt(res["pts"], var_name=groupby, value_name="pts", ignore_index=False)
    temp.index.name = "gene"
    temp.reset_index(inplace=True)
    output = output.merge(temp, on=["gene", groupby], how="inner")
    output.rename(
        columns={
            "logfoldchanges": "lfc",
            "pvals": "pval",
            "pvals_adj": "pval_adj",
            "pts": "pct"
        },
        inplace=True
    )
    ann.uns["rank_genes_groups"] = output
    return output

def get_ties(
    it:Iterable,
    key:Callable
):
    '''
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
    '''
    if not isinstance(it, Iterable):
        raise TypeError(f"it should have type Iterable, was {type(it)}")
    if not isinstance(key, Callable):
        raise TypeError(f"key should have type Callable, was {type(key)}")
    output = dict()
    for i, v in enumerate(it):
        v = key(v)
        if v in output:
            output[v].append(i)
        else:
            output[v] = [i]
    return output

def extract_ligands_from_settings(
    settings:dict,
    combination:bool=True
):
    '''
    Extract all ligands from the settings. 

    Parameters
    ----------
    settings : dict
        the settings
    combination : bool
        whether to include combinations of ligands in the output

    Returns
    -------
    list
        a set which contains all ligands present in the settings
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(settings) is not dict:
        raise TypeError(f"settings should have type dict, was {type(settings)}")
    if type(combination) is not bool:
        raise TypeError(f"combination should have type bool, was {type(combination)}")
    output = set()
    for setting in settings.values():
        if type(setting["from"]) is str:
            output.add(setting["from"])
        else:
            if combination:
                output.add("-".join(setting["from"]))
            for ligand in setting["from"]:
                output.add(ligand)
    output_lst = []
    for e in output:
        res = search("-", e)
        if res is None:
            output_lst.append(e)
        else:
            res = res.span()
            output_lst.append([e[:res[0]], e[res[1]:]])
    return output_lst

def is_ligand_active(importances:pd.DataFrame):
    '''
    Returns a list of booleans indicating whether a ligand is active or not. 

    Parameters
    ----------
    importances : pandas.DataFrame
        a data frame which contains the metrics by which ligands can be ranked

    Returns
    -------
    list
        a list of booleans indicating whether a ligand is active or not
    '''
    return [
        test_ligand == true_ligand if type(test_ligand) is str else test_ligand in true_ligand
        for test_ligand, true_ligand in zip(importances["test_ligand"], importances["true_ligand"])
    ]
from scipy.sparse import hstack, vstack, csc_matrix, csr_matrix

from collections.abc import Iterable, Callable

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
    mat:np.ndarray|csc_matrix|csr_matrix,
    rows:list[int|bool]|tuple[int|bool]|np.ndarray=None,
    cols:list[int|bool]|tuple[int|bool]|np.ndarray=None
) -> np.ndarray|csc_matrix|csr_matrix:
    '''
    Subsets a matrix. 

    Parameters
    ----------
    mat : numpy.ndarray or scipy.csc_matrix or scipy.csr_matrix
        the matrix to subset
    rows : list or tuple of int or bool
        list of row indices to keep or list of booleans indicating which rows to keep
    cols : list or tuple of int or bool
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
                return mat
            else:
                return mat[:, cols]
        elif cols is None:
            return np.concatenate([[mat[row, :]] for row in rows])
        else:
            output = mat[
                [[row] for row in rows],
                [col for col in cols]
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

def combine_dicts(dict1:dict, dict2:dict) -> dict:
    '''
    Combine two dictionaries by their mutual keys. 

    Parameters
    ----------
    dict1 : dict
        one of the dictionaries to combine
    dict2 : dict
        one of the dictionaries to combine
    
    Returns
    -------
    dict
        a dictionary where each key is mapped to a tuple containing the mappings of the input dictionaries for that key
    
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
    if type(dict1) is not dict:
        raise TypeError(f"dict1 should have type dict, was {type(dict1)}")
    if type(dict2) is not dict:
        raise TypeError(f"dict2 should have type dict, was {type(dict2)}")
    return dict((key, (dict1[key], dict2[key])) for key in set(dict1.keys()).intersection(set(dict2.keys())))

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
        ligands, activities = ligand_activities.items()
    elif isinstance(ligand_activities, Iterable):
        ligands, activities = zip(*ligand_activities)
    else:
        raise TypeError(f"ligand_activities should be of type dict or Iterable, was {type(ligand_activities)}")
    columns = tuple(activities[0].keys())
    data = [tuple(act.values()) for act in activities]
    df = pd.DataFrame(data=data, index=ligands, columns=columns)
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
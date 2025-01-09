import numpy as np


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
    '''
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
    '''
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
    '''
    with open(filename) as file:
        lines = file.readlines()
    lines = [[word.strip("\"\'") for word in line.rstrip().split(",")] for line in lines]
    return (lines[0], lines[1:])

def read_csv_cols(filename:str) -> dict[list[str]]:
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
    '''
    with open(filename) as file:
        lines = file.readlines()
    lines = [[word.strip("\"\'") for word in line.rstrip().split(",")] for line in lines]
    return dict(zip(lines[0], zip(*lines[1:])))

def subset_matrix(mat:np.ndarray, rows:list[int]|list[bool]=None, cols:list[int]|list[bool]=None) -> np.ndarray:
    '''
    Subsets a matrix. 

    Parameters
    ----------
    mat : numpy.ndarray
        the matrix to subset
    rows : list of int or list of bool or None
        list of row indices to keep or list of booleans indicating which rows to keep
    cols : list of int or list of bool or None
        list of column indices to keep or list of booleans indicating which columns to keep
    
    Returns
    -------
    numpy.ndarray
        the subsetted matrix
    '''
    if rows is not None and (type(rows[0]) is bool or type(rows[0]) is np.bool):
        rows = [i for i, e in enumerate(rows) if e]
    if cols is not None and (type(cols[0]) is bool or type(cols[0]) is np.bool):
        cols = [i for i, e in enumerate(cols) if e]
    if rows is None:
        if cols is None:
            return mat
        else:
            return mat[:, cols]
    elif cols is None:
        return np.concatenate([[mat[row, :]] for row in rows])
    else:
        return mat[
            [[row] for row in rows],
            [col for col in cols]
        ]

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
    return dict((key, (dict1[key], dict2[key])) for key in set(dict1.keys()).intersection(set(dict2.keys())))
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
        the read matrix
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

def read_csv_cols(filename:str) -> dict[tuple[str]]:
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

def subset_matrix(mat:np.ndarray, rows:list[int]|list[bool], cols:list[int]|list[bool]) -> np.ndarray:
    '''
    Subsets a matrix. 

    Parameters
    ----------
    mat : numpy.ndarray
        the matrix to subset
    rows : list
        list of row indices to keep or list of booleans indicating which rows to keep
    cols : list
        list of column indices to keep or list of booleans indicating which columns to keep
    
    Returns
    -------
    numpy.ndarray
        the subsetted matrix
    '''
    if type(rows[0]) is bool or type(rows[0]) is np.bool:
        rows = [i for i, e in enumerate(rows) if e]
    if type(cols[0]) is bool or type(cols[0]) is np.bool:
        cols = [i for i, e in enumerate(cols) if e]
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
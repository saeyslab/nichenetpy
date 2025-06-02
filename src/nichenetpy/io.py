from nichenetpy.prediction import LigandActivityPredictor

from math import log2, ceil
from scipy.sparse import csc_matrix, csr_matrix

import numpy as np
import struct


_INT_SIZE = 8
_DOUBLE_SIZE = 8

def _write_chunks(filename, *chunks, data=None):
    if data is None:
        data = bytearray()
    for chunk in chunks:
        data.extend(len(chunk).to_bytes(length=_INT_SIZE))
        data.extend(chunk)
    with open(filename, "wb") as file:
        file.write(data)

def write_ligand_target_matrix(
    filename:str,
    predictor:LigandActivityPredictor|None=None,
    mat:np.ndarray|None=None,
    row_names:list[str]|None=None,
    col_names:list[str]|None=None
):
    '''
    Writes a ligand-target matrix to a file. 

    Parameters
    ----------
    filename : str
        the name of the file to write to
    predictor : LigandActvivityPredictor or None
        the predictor which contains the matrix
    mat : numpy.ndarray or None
        the ligand-target matrix
    row_names : list of str or None
        the names of the rows
    col_names : list of str or None
        the names of the columns
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if predictor is None:
        if type(mat) is not np.ndarray:
            raise TypeError(f"expected a numpy.ndarray for mat, got {type(mat)}")
        if type(row_names) is not list:
            raise TypeError(f"expected a list of strings for row_names, got {type(row_names)}")
        if type(col_names) is not list:
            raise TypeError(f"expected a list of strings for col_names, got {type(col_names)}")
    else:
        if type(predictor) is not LigandActivityPredictor:
            raise TypeError(f"expected a LigandActivityPredictor for predictor, got {type(predictor)}")
        mat = predictor.ligand_target_matrix
        row_names = predictor.row_names
        col_names = predictor.col_names
    _write_chunks(
        filename,
        "\n".join(row_names).encode("ascii"),
        "\n".join(col_names).encode("ascii"),
        mat.tobytes()
    )

def read_ligand_target_matrix(filename:str) -> tuple[np.ndarray, list[str], list[str]]:
    '''
    reads a ligand-target matrix from a file. 

    Parameters
    ----------
    filename : str
        the name of the file to read from
    
    Returns
    -------
    numpy.ndarray
        the ligand-target matrix
    list of str
        the names of the rows
    list of str
        the names of the columns

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    with open(filename, "rb") as file:
        row_names = file.read(int.from_bytes(file.read(_INT_SIZE)))
        col_names = file.read(int.from_bytes(file.read(_INT_SIZE)))
        mat = np.frombuffer(file.read(int.from_bytes(file.read(_INT_SIZE))))
    row_names = row_names.decode("ascii").split()
    col_names = col_names.decode("ascii").split()
    mat = mat.reshape((len(row_names), len(col_names)))
    return (mat, row_names, col_names)

def write_sparse_matrix(
    filename:str,
    mat:csc_matrix|csr_matrix,
    row_names:str,
    col_names:str
):
    '''
    Writes a sparse matrix to a file. 

    Parameters
    ----------
    filename : str
        the name of the file to write to
    mat : csc_matrix or csr_matrix
        the sparse matrix
    row_names : list of str
        the names of the rows
    col_names : list of str
        the names of the columns

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    if type(mat) is not csc_matrix and type(mat) is not csr_matrix:
        raise TypeError(f"mat should have type csc_matrix or csr_matrix, was {type(mat)}")
    if type(row_names) is not list:
        raise TypeError(f"row_names should be a list of strings, had type {type(row_names)}")
    if type(col_names) is not list:
        raise TypeError(f"col_names should be a list of strings, had type {type(col_names)}")
    _write_chunks(
        filename,
        "\n".join(row_names).encode("ascii"),
        "\n".join(col_names).encode("ascii"),
        mat.data.tobytes(),
        mat.indices.tobytes(),
        mat.indptr.tobytes()
    )

def _read_sparse_matrix(
    filename
):
    with open(filename, "rb") as file:
        row_names = file.read(int.from_bytes(file.read(_INT_SIZE)))
        col_names = file.read(int.from_bytes(file.read(_INT_SIZE)))
        data = np.frombuffer(file.read(int.from_bytes(file.read(_INT_SIZE))))
        indices = np.frombuffer(file.read(int.from_bytes(file.read(_INT_SIZE))), dtype=np.int32)
        indptr = np.frombuffer(file.read(int.from_bytes(file.read(_INT_SIZE))), dtype=np.int32)
    row_names = row_names.decode("ascii").split()
    col_names = col_names.decode("ascii").split()
    return (data, indices, indptr, row_names, col_names)

def read_csc_matrix(
    filename:str
) -> tuple[csc_matrix, list[str], list[str]]:
    '''
    reads a csc_matrix from a file. 

    Parameters
    ----------
    filename : str
        the name of the file to read from
    
    Returns
    -------
    csc_matrix
        the sparse matrix
    list of str
        the names of the rows
    list of str
        the names of the columns

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    data, indices, indptr, row_names, col_names = _read_sparse_matrix(filename)
    mat = csc_matrix((data, indices, indptr))
    return (mat, row_names, col_names)

def read_csr_matrix(
    filename:str
) -> tuple[csr_matrix, list[str], list[str]]:
    '''
    reads a csr_matrix from a file. 

    Parameters
    ----------
    filename : str
        the name of the file to read from
    
    Returns
    -------
    csr_matrix
        the sparse matrix
    list of str
        the names of the rows
    list of str
        the names of the columns

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    data, indices, indptr, row_names, col_names = _read_sparse_matrix(filename)
    mat = csr_matrix((data, indices, indptr))
    return (mat, row_names, col_names)

def write_network(filename:str, mapping:list[tuple[str, str]]):
    '''
    Writes a network to a file. 

    Parameters
    ----------
    filename : str
        the name of the file to write to
    mapping : list[tuple[str, str]]
        the network connections

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    if type(mapping) is not list:
        raise TypeError(f"mapping should have type list[tuple[str, str]], was {type(mapping)}")
    grouped_mapping = dict()
    name2id = dict()
    for fr, to in mapping:
        if fr in grouped_mapping:
            grouped_mapping[fr].append(to)
        else:
            grouped_mapping[fr] = [to]
        if fr not in name2id:
            name2id[fr] = len(name2id)
        if to not in name2id:
            name2id[to] = len(name2id)
    id_size = ceil(log2(len(name2id) - 1))
    data = bytearray()
    for fr, tos in grouped_mapping.items():
        data.extend(name2id[fr].to_bytes(length=id_size))
        data.extend(len(tos).to_bytes(length=_INT_SIZE))
        for to in tos:
            data.extend(name2id[to].to_bytes(length=id_size))
    _write_chunks(
        filename,
        "\n".join(name2id.keys()).encode("ascii"),
        data,
        data=bytearray(id_size.to_bytes(_INT_SIZE))
    )

def write_weighted_network(filename:str, mapping:list[tuple[str, str, float]]):
    '''
    Writes a weighted network to a file. 

    Parameters
    ----------
    filename : str
        the name of the file to write to
    mapping : list[tuple[str, str, float]]
        the weighted connections of the network

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    if type(mapping) is not list:
        raise TypeError(f"mapping should have type list[tuple[str, str, float]], was {type(mapping)}")
    grouped_mapping = dict()
    name2id = dict()
    for fr, to, w in mapping:
        if fr in grouped_mapping:
            grouped_mapping[fr].append((to, w))
        else:
            grouped_mapping[fr] = [(to, w)]
        if fr not in name2id:
            name2id[fr] = len(name2id)
        if to not in name2id:
            name2id[to] = len(name2id)
    id_size = ceil(log2(len(name2id) - 1))
    data = bytearray()
    for fr, group in grouped_mapping.items():
        data.extend(name2id[fr].to_bytes(length=id_size))
        data.extend(len(group).to_bytes(length=_INT_SIZE))
        for to, w in group:
            data.extend(name2id[to].to_bytes(length=id_size))
            data.extend(struct.pack("d", w))
    _write_chunks(
        filename,
        "\n".join(name2id.keys()).encode("ascii"),
        data,
        data=bytearray(id_size.to_bytes(_INT_SIZE))
    )

def read_network(filename:str) -> list[tuple[str, str]]:
    '''
    reads a network from a file. 

    Parameters
    ----------
    filename : str
        the name of the file to read from
    
    Returns
    -------
    list of tuples
        the network connections

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    with open(filename, "rb") as file:
        id_size = int.from_bytes(file.read(_INT_SIZE))
        names = file.read(int.from_bytes(file.read(_INT_SIZE)))
        mapping = file.read(int.from_bytes(file.read(_INT_SIZE)))
    names = names.decode("ascii").split()
    id2name = dict(enumerate(names))
    output = []
    i = 0
    while i < len(mapping):
        fr = id2name[int.from_bytes(mapping[i:i+id_size])]
        i += id_size
        k = int.from_bytes(mapping[i:i+_INT_SIZE])
        i += _INT_SIZE
        for _ in range(k):
            output.append((fr, id2name[int.from_bytes(mapping[i:i+id_size])]))
            i += id_size
    return output

def read_weighted_network(filename:str) -> list[tuple[str, str, float]]:
    '''
    reads a weighted network from a file. 

    Parameters
    ----------
    filename : str
        the name of the file to read from
    
    Returns
    -------
    list of tuples
        the weighted connections of the network

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    with open(filename, "rb") as file:
        id_size = int.from_bytes(file.read(_INT_SIZE))
        names = file.read(int.from_bytes(file.read(_INT_SIZE)))
        mapping = file.read(int.from_bytes(file.read(_INT_SIZE)))
    names = names.decode("ascii").split()
    id2name = dict(enumerate(names))
    output = []
    i = 0
    while i < len(mapping):
        fr = id2name[int.from_bytes(mapping[i:i+id_size])]
        i += id_size
        k = int.from_bytes(mapping[i:i+_INT_SIZE])
        i += _INT_SIZE
        for _ in range(k):
            output.append((
                fr,
                id2name[int.from_bytes(mapping[i:i+id_size])],
                struct.unpack("d", mapping[i+id_size:i+id_size+_DOUBLE_SIZE])[0]
            ))
            i += id_size + _DOUBLE_SIZE
    return output
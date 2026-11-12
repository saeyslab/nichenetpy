import struct
from collections.abc import Iterable, Sequence
from typing import overload

import numpy as np
import pandas as pd
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.typing import nichenet_matrix, validate_matrix_type

_INT_SIZE = 8
_DOUBLE_SIZE = 8


def _write_chunks(filename: str, *chunks: bytes | bytearray, data: bytearray | None = None):
    if data is None:
        data = bytearray()
    for chunk in chunks:
        data.extend(len(chunk).to_bytes(length=_INT_SIZE))
        data.extend(chunk)
    with open(filename, "wb") as file:
        file.write(data)


def _encode_names(names: Sequence[str]) -> bytes:
    # the names are separated by newlines, so a name can't contain one
    if any("\n" in name for name in names):
        raise ValueError("names can't contain a newline character")
    return "\n".join(names).encode("utf-8")


def _name_list(names: object, arg_name: str) -> list[str]:
    # any ordered sequence of names (list, tuple, numpy array, pandas Index), not a str or an unordered set
    if isinstance(names, str) or not isinstance(names, (Sequence, np.ndarray, pd.Index)):
        raise TypeError(f"{arg_name} should be a Sequence of strings, had type {type(names)}")
    return list(names)


def _check_names_shape(row_names: list[str], col_names: list[str], shape: tuple[int, ...]):
    # the shape isn't stored, the readers take it from the names
    if (len(row_names), len(col_names)) != shape:
        raise ValueError(
            f"mat has shape {shape}, but there are {len(row_names)} row names and {len(col_names)} column names"
        )


def _decode_names(names: bytes) -> list[str]:
    # split on newlines only (names can contain spaces), UTF-8 is a superset of ASCII so older files can still be read
    return names.decode("utf-8").split("\n") if names else []


@overload
def write_ligand_target_matrix(
    filename: str,
    predictor: None,
    mat: nichenet_matrix,
    row_names: Sequence[str],
    col_names: Sequence[str],
    column_major: bool = True,
): ...


@overload
def write_ligand_target_matrix(
    filename: str,
    predictor: LigandActivityPredictor[str],
    mat: None,
    row_names: None,
    col_names: None,
    column_major: bool = True,
): ...


def write_ligand_target_matrix(
    filename: str,
    predictor: LigandActivityPredictor[str] | None = None,
    mat: nichenet_matrix | None = None,
    row_names: Sequence[str] | None = None,
    col_names: Sequence[str] | None = None,
    column_major: bool = True,
):
    """
    Writes a ligand-target matrix to a file.

    Parameters
    ----------
    filename : str
        the name of the file to write to
    predictor : LigandActvivityPredictor or None
        the predictor which contains the matrix
    mat : nichenet_matrix or None
        the ligand-target matrix
    row_names : Sequence of str or None
        the names of the rows
    col_names : Sequence of str or None
        the names of the columns
    column_major : bool
        whether to use column_major or row_major data format

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if the number of row or column names doesn't match the shape of the matrix
        if a name contains a newline character
    """
    if type(column_major) is not bool:
        raise TypeError(f"column_major should have type bool, was {type(column_major)}")
    if predictor is None:
        validate_matrix_type(mat, f"expected a nichenet_matrix for mat, got {type(mat)}")
    else:
        if not isinstance(predictor, LigandActivityPredictor):
            raise TypeError(f"expected a LigandActivityPredictor for predictor, got {type(predictor)}")
        mat = predictor.ligand_target_matrix
        row_names = predictor.row_names
        col_names = predictor.col_names
    row_names = _name_list(row_names, "row_names")
    col_names = _name_list(col_names, "col_names")
    _check_names_shape(row_names, col_names, mat.shape)  # type: ignore # mat can't be None at this point
    if not isinstance(mat, np.ndarray):
        mat = mat.toarray(order="F" if column_major else "C")  # type: ignore # mat can't be None at this point
    _write_chunks(
        filename,
        _encode_names(row_names),
        _encode_names(col_names),
        # the reader assumes float64 data (np.frombuffer default), so other float types are converted
        mat.astype(np.float64, copy=False).tobytes(order=("F" if column_major else "C")),
    )


def read_ligand_target_matrix(filename: str, column_major: bool = True) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Reads a ligand-target matrix from a file.

    Parameters
    ----------
    filename : str
        the name of the file to read from
    column_major : bool
        whether to use column_major or row_major data format

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
    """
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    if type(column_major) is not bool:
        raise TypeError(f"column_major should have type bool, was {type(column_major)}")
    with open(filename, "rb") as file:
        row_names = file.read(int.from_bytes(file.read(_INT_SIZE)))
        col_names = file.read(int.from_bytes(file.read(_INT_SIZE)))
        mat = np.frombuffer(file.read(int.from_bytes(file.read(_INT_SIZE))))
    row_names = _decode_names(row_names)  # type: ignore
    col_names = _decode_names(col_names)  # type: ignore
    mat = mat.reshape((len(row_names), len(col_names)), order=("F" if column_major else "C"))
    return (mat, row_names, col_names)  # type: ignore


def write_sparse_matrix(
    filename: str,
    mat: csc_matrix[np.floating] | csr_matrix[np.floating] | csc_array[np.floating] | csr_array[np.floating],
    row_names: Sequence[str],
    col_names: Sequence[str],
):
    """
    Writes a sparse matrix to a file.

    Parameters
    ----------
    filename : str
        the name of the file to write to
    mat : csc_matrix or csr_matrix or csc_array or csr_array
        the sparse matrix (read it back with `read_csc_matrix` or `read_csr_matrix`, which return a csc_matrix or
        csr_matrix)
    row_names : Sequence of str
        the names of the rows
    col_names : Sequence of str
        the names of the columns

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if the number of row or column names doesn't match the shape of mat
        if mat is too large for the file format (more than 2^31 - 1 rows, columns or nonzero values)
        if a name contains a newline character
    """
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    if not isinstance(mat, (csc_matrix, csr_matrix, csc_array, csr_array)):
        raise TypeError(f"mat should have type csc_matrix, csr_matrix, csc_array or csr_array, was {type(mat)}")
    row_names = _name_list(row_names, "row_names")
    col_names = _name_list(col_names, "col_names")
    _check_names_shape(row_names, col_names, mat.shape)
    # the reader assumes int32 indices and indptr, scipy uses int64 for large matrices (and after some operations)
    int32_max = np.iinfo(np.int32).max
    if max(mat.shape) > int32_max or mat.nnz > int32_max:
        raise ValueError(f"mat is too large to write, it has shape {mat.shape} and {mat.nnz} nonzero values")
    _write_chunks(
        filename,
        _encode_names(row_names),
        _encode_names(col_names),
        # the reader assumes float64 data (np.frombuffer default), so other float types are converted
        mat.data.astype(np.float64, copy=False).tobytes(),
        mat.indices.astype(np.int32, copy=False).tobytes(),
        mat.indptr.astype(np.int32, copy=False).tobytes(),
    )


def _read_sparse_matrix(filename):
    with open(filename, "rb") as file:
        row_names = file.read(int.from_bytes(file.read(_INT_SIZE)))
        col_names = file.read(int.from_bytes(file.read(_INT_SIZE)))
        data = np.frombuffer(file.read(int.from_bytes(file.read(_INT_SIZE))))
        indices = np.frombuffer(file.read(int.from_bytes(file.read(_INT_SIZE))), dtype=np.int32)
        indptr = np.frombuffer(file.read(int.from_bytes(file.read(_INT_SIZE))), dtype=np.int32)
    row_names = _decode_names(row_names)
    col_names = _decode_names(col_names)
    return (data, indices, indptr, row_names, col_names)


def read_csc_matrix(filename: str) -> tuple[csc_matrix[np.float64], list[str], list[str]]:
    """
    Reads a csc_matrix from a file.

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
    """
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    data, indices, indptr, row_names, col_names = _read_sparse_matrix(filename)
    mat = csc_matrix((data, indices, indptr), shape=(len(row_names), len(col_names)))
    return (mat, row_names, col_names)


def read_csr_matrix(filename: str) -> tuple[csr_matrix[np.float64], list[str], list[str]]:
    """
    Reads a csr_matrix from a file.

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
    """
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    data, indices, indptr, row_names, col_names = _read_sparse_matrix(filename)
    mat = csr_matrix((data, indices, indptr), shape=(len(row_names), len(col_names)))
    return (mat, row_names, col_names)


def _id_size(n_names: int) -> int:
    # number of bytes needed to store the ids 0 to n_names - 1 (at least 1)
    return max(1, ((n_names - 1).bit_length() + 7) // 8)


def write_network(filename: str, mapping: Iterable[tuple[str, str]]):
    """
    Writes a network to a file.

    Parameters
    ----------
    filename : str
        the name of the file to write to
    mapping : Iterable of tuple[str, str]
        the network connections (iterated once, so e.g. a generator or zip is fine)

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if a gene name contains a newline character

    Notes
    -----
    Each gene name is stored once, the links are stored as pairs of gene ids of 1 byte (up to 256 genes), 2 bytes (up to
    65,536 genes), etc. `read_network` returns the links grouped by source gene, not in the original order.

    Compared with pickle (highest protocol) on random networks, files are about 5 times smaller (2.3 MB vs 12.0 MB for
    15,000 genes and 1 million links, gzip compressed pickle: 5.8 MB). Pickle is somewhat faster (3 million links: write
    0.9 s, read 0.4 s vs write 0.9 s, read 0.6 s), as this function and `read_network` loop over the links in Python.
    Unlike pickle files, these files don't execute code when they are read, so files from other sources are safe to read.
    """
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    if isinstance(mapping, str) or not isinstance(mapping, Iterable):
        raise TypeError(f"mapping should be an Iterable of tuple[str, str], was {type(mapping)}")
    grouped_mapping = {}
    name2id = {}
    for fr, to in mapping:
        if fr in grouped_mapping:
            grouped_mapping[fr].append(to)
        else:
            grouped_mapping[fr] = [to]
        if fr not in name2id:
            name2id[fr] = len(name2id)
        if to not in name2id:
            name2id[to] = len(name2id)
    id_size = _id_size(len(name2id))
    data = bytearray()
    for fr, tos in grouped_mapping.items():
        data.extend(name2id[fr].to_bytes(length=id_size))
        data.extend(len(tos).to_bytes(length=_INT_SIZE))
        for to in tos:
            data.extend(name2id[to].to_bytes(length=id_size))
    _write_chunks(
        filename, _encode_names(list(name2id)), data, data=bytearray(id_size.to_bytes(_INT_SIZE))
    )


def write_weighted_network(filename: str, mapping: Iterable[tuple[str, str, float]]):
    """
    Writes a weighted network to a file.

    Parameters
    ----------
    filename : str
        the name of the file to write to
    mapping : Iterable of tuple[str, str, float]
        the weighted connections of the network (iterated once, so e.g. a generator or zip is fine)

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if a gene name contains a newline character

    Notes
    -----
    Each gene name is stored once, the links are stored as pairs of gene ids of 1 byte (up to 256 genes), 2 bytes (up to
    65,536 genes), etc. followed by the weight (8 byte float). `read_weighted_network` returns the links grouped by source
    gene, not in the original order.

    Compared with pickle (highest protocol) on random networks, files are about 2 times smaller (10.3 MB vs 21.0 MB for
    15,000 genes and 1 million links, gzip compressed pickle: 14.5 MB); the weights take up most of the space. Pickle is
    faster (3 million links: write 1.0 s, read 0.5 s vs write 2.6 s, read 1.1 s), as this function and
    `read_weighted_network` loop over the links in Python. Unlike pickle files, these files don't execute code when they
    are read, so files from other sources are safe to read.
    """
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    if isinstance(mapping, str) or not isinstance(mapping, Iterable):
        raise TypeError(f"mapping should be an Iterable of tuple[str, str, float], was {type(mapping)}")
    grouped_mapping = {}
    name2id = {}
    for fr, to, w in mapping:
        if fr in grouped_mapping:
            grouped_mapping[fr].append((to, w))
        else:
            grouped_mapping[fr] = [(to, w)]
        if fr not in name2id:
            name2id[fr] = len(name2id)
        if to not in name2id:
            name2id[to] = len(name2id)
    id_size = _id_size(len(name2id))
    data = bytearray()
    for fr, group in grouped_mapping.items():
        data.extend(name2id[fr].to_bytes(length=id_size))
        data.extend(len(group).to_bytes(length=_INT_SIZE))
        for to, w in group:
            data.extend(name2id[to].to_bytes(length=id_size))
            data.extend(struct.pack("d", w))
    _write_chunks(
        filename, _encode_names(list(name2id)), data, data=bytearray(id_size.to_bytes(_INT_SIZE))
    )


def read_network(filename: str) -> list[tuple[str, str]]:
    """
    Reads a network from a file.

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
    """
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    with open(filename, "rb") as file:
        id_size = int.from_bytes(file.read(_INT_SIZE))
        names = file.read(int.from_bytes(file.read(_INT_SIZE)))
        mapping = file.read(int.from_bytes(file.read(_INT_SIZE)))
    names = _decode_names(names)  # type: ignore
    id2name = dict(enumerate(names))
    output = []
    i = 0
    while i < len(mapping):
        fr = id2name[int.from_bytes(mapping[i : i + id_size])]
        i += id_size
        k = int.from_bytes(mapping[i : i + _INT_SIZE])
        i += _INT_SIZE
        for _ in range(k):
            output.append((fr, id2name[int.from_bytes(mapping[i : i + id_size])]))
            i += id_size
    return output  # type: ignore


def read_weighted_network(filename: str) -> list[tuple[str, str, float]]:
    """
    Reads a weighted network from a file.

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
    """
    if type(filename) is not str:
        raise TypeError(f"filename should have type str, was {type(filename)}")
    with open(filename, "rb") as file:
        id_size = int.from_bytes(file.read(_INT_SIZE))
        names = file.read(int.from_bytes(file.read(_INT_SIZE)))
        mapping = file.read(int.from_bytes(file.read(_INT_SIZE)))
    names = _decode_names(names)  # type: ignore
    id2name = dict(enumerate(names))
    output = []
    i = 0
    while i < len(mapping):
        fr = id2name[int.from_bytes(mapping[i : i + id_size])]
        i += id_size
        k = int.from_bytes(mapping[i : i + _INT_SIZE])
        i += _INT_SIZE
        for _ in range(k):
            output.append(
                (
                    fr,
                    id2name[int.from_bytes(mapping[i : i + id_size])],
                    struct.unpack("d", mapping[i + id_size : i + id_size + _DOUBLE_SIZE])[0],
                )
            )
            i += id_size + _DOUBLE_SIZE
    return output  # type: ignore

from nichenetpy.prediction import LigandActivityPredictor

from math import log2, ceil
from scipy.sparse import csc_matrix, csr_matrix

import numpy as np
import struct


INT_SIZE = 8
DOUBLE_SIZE = 8

def _write_chunks(filename, *chunks, data=None):
    if data is None:
        data = bytearray()
    for chunk in chunks:
        data.extend(len(chunk).to_bytes(length=INT_SIZE))
        data.extend(chunk)
    with open(filename, "wb") as file:
        file.write(data)

def write_ligand_target_matrix(filename:str, *args):
    if len(args) == 1:
        if type(args[0]) is not LigandActivityPredictor:
            raise TypeError(f"expected a LigandActivityPredictor, got {type(args[0])}")
        predictor = args[0]
        mat = predictor.ligand_target_matrix
        row_names = predictor.row_names
        col_names = predictor.col_names
    elif len(args) == 3:
        if type(args[0]) is not np.ndarray:
            raise TypeError(f"expected a numpy.ndarray as first argument after filename, got {type(args[0])}")
        if type(args[1]) is not list:
            raise TypeError(f"expected a list of strings as second argument after filename, got {type(args[1])}")
        if type(args[2]) is not list:
            raise TypeError(f"expected a list of strings as third argument after filename, got {type(args[2])}")
        mat = args[0]
        row_names = args[1]
        col_names = args[2]
    else:
        raise TypeError(f"expected a LigandActivityPredictor or a numpy.ndarray and two lists of strings after filename, got {len(args)} arguments")
    _write_chunks(
        filename,
        "\n".join(row_names).encode("ascii"),
        "\n".join(col_names).encode("ascii"),
        mat.tobytes()
    )

def read_ligand_target_matrix(filename:str) -> tuple[np.ndarray, list[str], list[str]]:
    with open(filename, "rb") as file:
        row_names = file.read(int.from_bytes(file.read(INT_SIZE)))
        col_names = file.read(int.from_bytes(file.read(INT_SIZE)))
        mat = np.frombuffer(file.read(int.from_bytes(file.read(INT_SIZE))))
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
        row_names = file.read(int.from_bytes(file.read(INT_SIZE)))
        col_names = file.read(int.from_bytes(file.read(INT_SIZE)))
        data = np.frombuffer(file.read(int.from_bytes(file.read(INT_SIZE))))
        indices = np.frombuffer(file.read(int.from_bytes(file.read(INT_SIZE))), dtype=np.int32)
        indptr = np.frombuffer(file.read(int.from_bytes(file.read(INT_SIZE))), dtype=np.int32)
    row_names = row_names.decode("ascii").split()
    col_names = col_names.decode("ascii").split()
    return (data, indices, indptr, row_names, col_names)

def read_csc_matrix(
    filename:str
) -> tuple[csc_matrix, list[str], list[str]]:
    data, indices, indptr, row_names, col_names = _read_sparse_matrix(filename)
    mat = csc_matrix((data, indices, indptr))
    return (mat, row_names, col_names)

def read_csr_matrix(
    filename:str
) -> tuple[csr_matrix, list[str], list[str]]:
    data, indices, indptr, row_names, col_names = _read_sparse_matrix(filename)
    mat = csr_matrix((data, indices, indptr))
    return (mat, row_names, col_names)

def write_network(filename:str, mapping:list[tuple[str, str]]):
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
        data.extend(len(tos).to_bytes(length=INT_SIZE))
        for to in tos:
            data.extend(name2id[to].to_bytes(length=id_size))
    _write_chunks(
        filename,
        "\n".join(name2id.keys()).encode("ascii"),
        data,
        data=bytearray(id_size.to_bytes(INT_SIZE))
    )

def write_weighted_network(filename:str, mapping:list[tuple[str, str, float]]):
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
        data.extend(len(group).to_bytes(length=INT_SIZE))
        for to, w in group:
            data.extend(name2id[to].to_bytes(length=id_size))
            data.extend(struct.pack("d", w))
    _write_chunks(
        filename,
        "\n".join(name2id.keys()).encode("ascii"),
        data,
        data=bytearray(id_size.to_bytes(INT_SIZE))
    )

def read_network(filename:str) -> list[tuple[str, str]]:
    with open(filename, "rb") as file:
        id_size = int.from_bytes(file.read(INT_SIZE))
        names = file.read(int.from_bytes(file.read(INT_SIZE)))
        mapping = file.read(int.from_bytes(file.read(INT_SIZE)))
    names = names.decode("ascii").split()
    id2name = dict(enumerate(names))
    output = []
    i = 0
    while i < len(mapping):
        fr = id2name[int.from_bytes(mapping[i:i+id_size])]
        i += id_size
        k = int.from_bytes(mapping[i:i+INT_SIZE])
        i += INT_SIZE
        for _ in range(k):
            output.append((fr, id2name[int.from_bytes(mapping[i:i+id_size])]))
            i += id_size
    return output

def read_weighted_network(filename:str) -> list[tuple[str, str, float]]:
    with open(filename, "rb") as file:
        id_size = int.from_bytes(file.read(INT_SIZE))
        names = file.read(int.from_bytes(file.read(INT_SIZE)))
        mapping = file.read(int.from_bytes(file.read(INT_SIZE)))
    names = names.decode("ascii").split()
    id2name = dict(enumerate(names))
    output = []
    i = 0
    while i < len(mapping):
        fr = id2name[int.from_bytes(mapping[i:i+id_size])]
        i += id_size
        k = int.from_bytes(mapping[i:i+INT_SIZE])
        i += INT_SIZE
        for _ in range(k):
            output.append((
                fr,
                id2name[int.from_bytes(mapping[i:i+id_size])],
                struct.unpack("d", mapping[i+id_size:i+id_size+DOUBLE_SIZE])[0]
            ))
            i += id_size + DOUBLE_SIZE
    return output
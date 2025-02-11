from nichenetpy.prediction import LigandActivityPredictor

from math import log2, ceil

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
    mat.reshape((len(row_names), len(col_names)))
    return (mat, row_names, col_names)

def write_network(filename:str, mapping:list[tuple[str, str]]):
    name2id = dict()
    for fr, to in mapping:
        if fr not in name2id:
            name2id[fr] = len(name2id)
        if to not in name2id:
            name2id[to] = len(name2id)
    id_size = ceil(log2(len(name2id) - 1))
    data = bytearray()
    for fr, to in mapping: # run length encoding may have potential here (as an option)
        data.extend(name2id[fr].to_bytes(length=id_size))
        data.extend(name2id[to].to_bytes(length=id_size))
    _write_chunks(
        filename,
        "\n".join(name2id.keys()).encode("ascii"),
        data,
        data=bytearray(id_size.to_bytes(INT_SIZE))
    )

def write_weighted_network(filename:str, mapping:list[tuple[str, str, float]]):
    name2id = dict()
    for fr, to, _ in mapping:
        if fr not in name2id:
            name2id[fr] = len(name2id)
        if to not in name2id:
            name2id[to] = len(name2id)
    id_size = ceil(log2(len(name2id) - 1))
    data = bytearray()
    for fr, to, w in mapping: # run length encoding may have potential here (as an option)
        data.extend(name2id[fr].to_bytes(length=id_size))
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
    return [
        (
            id2name[int.from_bytes(mapping[i:i+id_size])],
            id2name[int.from_bytes(mapping[i+id_size:i+2*id_size])]
        )
        for i in range(0, len(mapping), 2*id_size)
    ]

def read_weighted_network(filename:str) -> list[tuple[str, str, float]]:
    with open(filename, "rb") as file:
        id_size = int.from_bytes(file.read(INT_SIZE))
        names = file.read(int.from_bytes(file.read(INT_SIZE)))
        mapping = file.read(int.from_bytes(file.read(INT_SIZE)))
    names = names.decode("ascii").split()
    id2name = dict(enumerate(names))
    return [
        (
            id2name[int.from_bytes(mapping[i:i+id_size])],
            id2name[int.from_bytes(mapping[i+id_size:i+2*id_size])],
            struct.unpack("d", mapping[i+2*id_size:i+2*id_size+DOUBLE_SIZE])[0]
        )
        for i in range(0, len(mapping), 2*id_size+DOUBLE_SIZE)
    ]
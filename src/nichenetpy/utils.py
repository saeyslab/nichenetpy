import numpy as np


def read_list_from_csv(filename:str) -> list[str]:
    with open(filename) as file:
        lines = file.readlines()
    return [line.rstrip().strip("\"\'") for line in lines[1:]]

def read_matrix_from_csv(filename:str) -> tuple[np.ndarray, list[str], list[str]]:
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
    with open(filename) as file:
        lines = file.readlines()
    lines = [[word.strip("\"\'") for word in line.rstrip().split(",")] for line in lines]
    return dict(zip(lines[0], zip(*lines[1:])))

def subset_matrix(mat:np.ndarray, rows:list[int]|list[bool], cols:list[int]|list[bool]) -> np.ndarray:
    if type(rows[0]) is bool or type(rows[0]) is np.bool:
        rows = [i for i, e in enumerate(rows) if e]
    if type(cols[0]) is bool or type(cols[0]) is np.bool:
        cols = [i for i, e in enumerate(cols) if e]
    return mat[
        [[row] for row in rows],
        [col for col in cols]
    ]

def remove_zero_rows_cols(mat:np.ndarray) -> np.ndarray:
    return subset_matrix(mat, mat.any(axis=1), mat.any(axis=0))
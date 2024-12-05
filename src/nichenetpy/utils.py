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
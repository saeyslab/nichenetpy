from numbers import Number
from collections.abc import Iterable
from math import log

import anndata
import os
import requests
import pickle


root_path = os.path.normpath("./tests/data/tutorial_files")
ann_path = os.path.join(root_path, "AnnData")
hnscc_path = os.path.join(root_path, "hnscc")
network_path = os.path.join(root_path, "model_construction")
eval_path = os.path.join(root_path, "model_evaluation")
train_path = os.path.normpath("./tutorial_files/model_optimization")

def download(url, max_tries=5):
    for i in max_tries:
        try:
            return requests.get(url)
        except requests.exceptions.ChunkedEncodingError as err:
            if i == max_tries - 1:
                raise err

def equals(
    x,
    y,
    err_bound=1e-2,
    zero_bound=1e-100
):
    if isinstance(x, Number) and isinstance(y, Number):
        return abs(x) <= zero_bound if y == 0 else abs(x - y) / y <= err_bound
    elif isinstance(x, Iterable) and isinstance(y, Iterable) and type(x) is not str and type(y) is not str:
        return equals_iter(x, y, err_bound, zero_bound)
    else:
        return x == y

def equals_iter(
    xs,
    ys,
    err_bound=1e-2,
    zero_bound=1e-100
):
    for x, y in zip(xs, ys):
        if not equals(x, y, err_bound, zero_bound):
            return False
    return True

def equals_ndarray(
    xs,
    ys,
    err_bound=1e-2,
    zero_bound=1e-100
):
    return equals_iter(xs.reshape(-1), ys.reshape(-1), err_bound, zero_bound)

def equals_dict(x, y):
    for key, val in y.items():
        if not equals(x[key], val):
            return False
    return True

def df2dict(df, key_cols, val_col):
    return dict(zip(
        zip(*(df[col] for col in key_cols)),
        df[val_col]
    ))

def get_model_pickle(type="mouse"):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    filename = f"nichenet_{type}.pkl"
    file_path = os.path.join(root_path, filename)
    if not os.path.exists(file_path):
        res = download(f"https://zenodo.org/records/17061000/files/{filename}")
        with open(file_path, "wb") as file:
            file.write(res.content)
    with open(os.path.join(root_path, filename), "rb") as file:
        model = pickle.loads(file.read())
    return model

def get_anndata_file(filename):
    if not os.path.exists(ann_path):
        os.makedirs(ann_path)
    file_path = os.path.join(ann_path, filename)
    if not os.path.exists(file_path):
        res = download(f"https://zenodo.org/records/15574665/files/{filename}")
        with open(file_path, "wb") as file:
            file.write(res.content)
    return anndata.io.read_h5ad(file_path)

def get_hnscc_file():
    if not os.path.exists(hnscc_path):
        os.makedirs(hnscc_path)
    for filename in (
        "expressed_genes.csv",
        "hnscc_expression.bin",
        "pemt_signature.txt",
        "sample_info.csv"
    ):
        file_path = os.path.join(hnscc_path, filename)
        if not os.path.exists(file_path):
            res = download(f"https://zenodo.org/records/14859451/files/{filename}")
            with open(file_path, "wb") as file:
                file.write(res.content)

def get_network_files():
    if not os.path.exists(network_path):
        os.makedirs(network_path)
    for filename in (
        "gr_human.csv",
        "lr_network_human.csv",
        "lr_sig_human.csv",
        "gr_mouse.csv",
        "lr_network_mouse.csv",
        "lr_sig_mouse.csv",
        "source_weights.csv",
        "optimized_source_weights.csv",
        "annotation_data_sources.csv"
    ):
        file_path = os.path.join(network_path, filename)
        if not os.path.exists(file_path):
            res = download(f"https://zenodo.org/records/15168364/files/{filename}")
            with open(file_path, "wb") as file:
                file.write(res.content)

def get_evaluation_files():
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    for filename in (
        #"cytosig_settings.json",
        "expression_settings_validation.json",
    ):
        file_path = os.path.join(eval_path, filename)
        if not os.path.exists(file_path):
            res = download(f"https://zenodo.org/records/15228527/files/{filename}")
            with open(file_path, "wb") as file:
                file.write(res.content)

def get_optimization_files():
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    for filename in (
        "settings_training_f1234.json",
        "settings_training_f1235.json",
        "settings_training_f1245.json",
        "settings_training_f1345.json",
        "settings_training_f2345.json"
    ):
        file_path = os.path.join(train_path, filename)
        if not os.path.exists(file_path):
            res = download(f"https://zenodo.org/records/15799578/files/{filename}")
            with open(file_path, "wb") as file:
                file.write(res.content)
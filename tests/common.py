# type: ignore

import os
import pickle
import random
import string
import time
from collections.abc import Iterable
from numbers import Number

import anndata
import requests

root_path = os.path.normpath("./tests/data/tutorial_files")
ann_path = os.path.join(root_path, "AnnData")
hnscc_path = os.path.join(root_path, "hnscc")
network_path = os.path.join(root_path, "model_construction")
eval_path = os.path.join(root_path, "model_evaluation")
train_path = os.path.join(root_path, "model_optimization")


def fetch(record, filename, directory, max_tries=5, timeout=60):
    """
    Download a file of a Zenodo record to directory, unless it's already there, and return its path.

    The file is downloaded to a temporary file that is renamed once it's complete, so a failed or interrupted download
    never leaves a file behind. Timeouts, connection errors and HTTP errors are retried with an increasing pause (Zenodo
    is sometimes briefly unavailable), after max_tries attempts a RuntimeError is raised.
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return file_path
    url = f"https://zenodo.org/records/{record}/files/{filename}"
    part_path = file_path + ".part"
    errors = []
    for attempt in range(max_tries):
        if attempt > 0:
            time.sleep(5 * 2 ** (attempt - 1))
        try:
            with requests.get(url, stream=True, timeout=timeout) as res:
                res.raise_for_status()
                size = 0
                with open(part_path, "wb") as file:
                    for chunk in res.iter_content(chunk_size=1 << 20):
                        file.write(chunk)
                        size += len(chunk)
                expected = res.headers.get("Content-Length")
                if size == 0 or (expected is not None and size != int(expected)):
                    raise requests.exceptions.ContentDecodingError(
                        f"incomplete download: {size} bytes, expected {expected}"
                    )
            os.replace(part_path, file_path)
            return file_path
        except requests.exceptions.RequestException as err:
            errors.append(f"attempt {attempt + 1}: {err}")
            if os.path.exists(part_path):
                os.remove(part_path)
    raise RuntimeError(
        f"could not download {filename} from Zenodo record {record} after {max_tries} attempts:\n" + "\n".join(errors)
    )


def equals(x, y, err_bound=1e-2, zero_bound=1e-100):
    if isinstance(x, Number) and isinstance(y, Number):
        return abs(x) <= zero_bound if y == 0 else abs(x - y) / y <= err_bound
    elif isinstance(x, Iterable) and isinstance(y, Iterable) and type(x) is not str and type(y) is not str:
        if not equals_iter(x, y, err_bound, zero_bound):
            print((x, y))
        return equals_iter(x, y, err_bound, zero_bound)
    else:
        return x == y


def equals_iter(xs, ys, err_bound=1e-2, zero_bound=1e-100):
    for x, y in zip(xs, ys, strict=True):
        if not equals(x, y, err_bound, zero_bound):
            return False
    return True


def equals_ndarray(xs, ys, err_bound=1e-2, zero_bound=1e-100):
    return equals_iter(xs.reshape(-1), ys.reshape(-1), err_bound, zero_bound)


def equals_dict(x, y, err_bound=1e-2, zero_bound=1e-100):
    for key, val in y.items():
        if not equals(x[key], val, err_bound, zero_bound):
            return False
    return True


def df2dict(df, key_cols, val_col):
    return dict(zip(zip(*(df[col] for col in key_cols), strict=True), df[val_col], strict=True))


def get_model_pickle(type="mouse"):
    file_path = fetch(17061000, f"nichenet_{type}.pkl", root_path)
    with open(file_path, "rb") as file:
        model = pickle.loads(file.read())
    return model


def get_ltf_matrix(type="mouse"):
    file_path = fetch(17061000, f"ltf_matrix_{type}.pkl", root_path)
    with open(file_path, "rb") as file:
        model = pickle.loads(file.read())
    return model


def get_anndata_file(filename):
    return anndata.io.read_h5ad(fetch(15574665, filename, ann_path))


def get_hnscc_file():
    for filename in ("expressed_genes.csv", "hnscc_expression.bin", "pemt_signature.txt", "sample_info.csv"):
        fetch(14859451, filename, hnscc_path)


def get_network_files():
    for filename in (
        "gr_human.csv",
        "lr_network_human.csv",
        "lr_sig_human.csv",
        "gr_mouse.csv",
        "lr_network_mouse.csv",
        "lr_sig_mouse.csv",
        "source_weights.csv",
        "optimized_source_weights.csv",
        "annotation_data_sources.csv",
        "geneinfo.csv",
        "geneinfo_alias_human.csv",
        "geneinfo_alias_mouse.csv",
    ):
        fetch(20395877, filename, network_path)


def get_evaluation_files():
    for filename in (
        # "cytosig_settings.json",
        "expression_settings_validation.json",
    ):
        fetch(15228527, filename, eval_path)


def get_optimization_files():
    for filename in (
        "settings_training_f1234.json",
        "settings_training_f1235.json",
        "settings_training_f1245.json",
        "settings_training_f1345.json",
        "settings_training_f2345.json",
    ):
        fetch(15799578, filename, train_path)


_random_char_choices = string.ascii_lowercase + string.digits


def random_string(n=6):
    return "".join(random.choices(_random_char_choices, k=random.randint(1, n)))

from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork
from nichenetpy.utils import (
    combine_by_key,
    combine_dicts
)
from nichenetpy.extraction import (
    get_expressed_genes,
    subset_ann,
    get_weighted_ligand_receptor_links,
    get_lfc_celltype
)
from nichenetpy.gene_symbol import mouse_alias_info
from nichenetpy.visualization import (
    prepare_ligand_target_visualization,
    prepare_ligand_receptor_visualization,
    heatmap_2d,
    heatmap_1d
)
from nichenetpy.metrics import group_metrics
from nichenetpy.io import read_ligand_target_matrix

from itertools import cycle, chain
from numbers import Number
from collections.abc import Iterable

import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import pickle

err_bound = 1e-3
root_path = os.path.normpath("./tests/data/tutorial_files")
ann_path = os.path.join(root_path, "AnnData")

top_10_ligand_activities = [
    ("Ifna1", 0.34233796),
    ("Ifnl3", 0.31641473),
    ("Ifnb1", 0.31168307),
    ("Il27", 0.30262806),
    ("Ifng", 0.29004163),
    ("Ifnk", 0.19970100),
    ("Ifne", 0.19817601),
    ("Ebi3", 0.17898331),
    ("Ifnl2", 0.16908673),
    ("Ifna2", 0.16606369)
]

def equals(x, y):
    if isinstance(x, Number) and isinstance(y, Number):
        return abs(x - y) < err_bound
    elif isinstance(x, Iterable) and isinstance(y, Iterable) and type(x) is not str and type(y) is not str:
        return equals_iter(x, y)
    else:
        return x == y

def equals_iter(xs, ys):
    for x, y in zip(xs, ys):
        if not equals(x, y):
            return False
    return True

def equals_ndarray(xs, ys):
    return equals_iter(xs.reshape(-1), ys.reshape(-1))

def get_model_pickle(type="mouse"):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    filename = f"nichenet_{type}.pkl"
    file_path = os.path.join(root_path, filename)
    if not os.path.exists(file_path):
        res = requests.get(f"https://zenodo.org/records/14887637/files/{filename}")
        with open(file_path, "wb") as file:
            file.write(res.content)

def get_anndata_file(filename):
    if not os.path.exists(ann_path):
        os.makedirs(ann_path)
    file_path = os.path.join(ann_path, filename)
    if not os.path.exists(file_path):
        res = requests.get(f"https://zenodo.org/records/14859451/files/{filename}")
        with open(file_path, "wb") as file:
            file.write(res.content)

def test_steps():
    get_model_pickle("mouse")
    get_anndata_file("annData3531889.h5")
    ann = anndata.io.read_h5ad(os.path.join(ann_path, "annData3531889.h5"))
    ann.var_names = ann.var["gene"]
    mouse_alias_info.alias_to_symbol(ann)
    with open(os.path.join(root_path, "nichenet_mouse.pkl"), "rb") as file:
        model = pickle.loads(file.read())
    predictor = model["predictor"]
    lr_network = model["lr_network"]
    lr_sig = model["lr_sig"]
    receiver = "CD8 T"
    expressed_genes_receiver = set(get_expressed_genes(receiver, ann, 0.05))
    all_receptors = lr_network.get_receptors()
    expressed_receptors = all_receptors.intersection(expressed_genes_receiver)
    potential_ligands = set(
        key for key, group in lr_network.item_iter()
        if len(group.intersection(expressed_receptors)) > 0
    )
    sender_celltypes = ("CD4 T", "Treg", "Mono", "NK", "B", "DC")
    list_expressed_genes_sender = [get_expressed_genes(ct, ann, pct=0.05) for ct in sender_celltypes]
    expressed_genes_sender = set(e for l in list_expressed_genes_sender for e in l)
    potential_ligands_focused = potential_ligands.intersection(expressed_genes_sender)
    assert len(expressed_genes_sender) == 8492
    assert len(potential_ligands) == 475
    assert len(potential_ligands_focused) == 122
    ann_receiver = subset_ann(
        ann,
        val=receiver,
        val_col="celltype",
        layers=["data", "counts"]
    )
    group_metrics(
        ann_receiver,
        groupby="aggregate",
        layer="data",
        min_pct=0.05,
        min_abs_lfc=0.25
    )
    DE_table = ann_receiver.uns["group_metrics"]
    geneset = set(
        DE_table[
            (DE_table["aggregate"] == "LCMV") &
            (DE_table["pval_adj"] <= 0.05)
        ]["gene"]
    )
    geneset.intersection_update(predictor.get_genes())
    assert len(geneset) == 241
    assert len(expressed_genes_receiver) == 3903
    ligand_activities = predictor.predict_ligand_activities(
        geneset=geneset,
        background_expressed_genes=expressed_genes_receiver,
        potential_ligands=potential_ligands
    )
    ligand_activities_sorted = sorted(ligand_activities.items(), key=lambda x : x[1]["aupr_corrected"], reverse=True)
    assert equals_iter(
        ((ligand, act["aupr_corrected"]) for ligand, act in ligand_activities_sorted[:10]),
        top_10_ligand_activities
    )
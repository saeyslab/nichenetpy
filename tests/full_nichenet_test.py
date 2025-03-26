from nichenetpy.utils import combine_by_key
from nichenetpy.extraction import (
    get_expressed_genes,
    subset_ann,
    get_weighted_ligand_receptor_links,
    get_lfc_celltype
)
from nichenetpy.gene_symbol import mouse_alias_info
from nichenetpy.metrics import group_metrics
from nichenetpy.wrappers import run_nichenet

from itertools import cycle, chain
from numbers import Number
from collections.abc import Iterable

import anndata
import os
import requests
import pickle
import pandas as pd

err_bound = 1e-3
root_path = os.path.normpath("./tests/data/tutorial_files")
ann_path = os.path.join(root_path, "AnnData")

# nichenetr output
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
top_10_active_ligand_target_links = [
    ("Il27", "Irf1", 0.3393635),
    ("Ifng", "Irf1", 0.2864755),
    ("Ifng", "Stat1", 0.2846630),
    ("Ifna1", "Stat1", 0.2828309),
    ("Ifna1", "Ifit2", 0.2794132),
    ("Ifna1", "Ifi35", 0.2789524),
    ("Ifna1", "Irf7", 0.2785550),
    ("Tnf", "Irf1", 0.2769230),
    ("Tnf", "Irf7", 0.2753571),
    ("Ifna1", "Stat2", 0.2731093)
]
top_10_ligand_receptor_links = [
    ("Ifng", "Ifngr1", 1.5938121),
    ("Ifna1", "Ifnar1", 1.5579253),
    ("H2-M3", "Cd8a", 1.4430922),
    ("Ifna1", "Ifnar2", 1.3783434),
    ("Ifnb1", "Ifnar1", 1.3397792),
    ("Il27", "Il27ra", 1.2787938),
    ("Ifna11", "Ifnar1", 1.2454250),
    ("Ifna12", "Ifnar1", 1.2454250),
    ("Ifna13", "Ifnar1", 1.2454250),
    ("Ifna14", "Ifnar1", 1.2454250)
]
top_10_ligand_activities_focused = [
    ("Il27", 0.30262806),
    ("Ebi3", 0.17898331),
    ("Tnf", 0.12740633),
    ("Ptprc", 0.11700399),
    ("H2-Eb1", 0.11242171),
    ("H2-M3", 0.11226548),
    ("Vsig10", 0.11064793),
    ("Clcf1", 0.09289923),
    ("H2-M2", 0.09154333),
    ("H2-T10", 0.09154333)
]
top_10_active_ligand_target_links_focused = [
    ("Il27", "Irf1", 0.33936348),
    ("Tnf", "Irf1", 0.27692301),
    ("Tnf", "Irf7", 0.27535714),
    ("Il27", "Stat1", 0.25249051),
    ("Tnf", "Tap1", 0.25076038),
    ("Il27", "Ifit3", 0.24560454),
    ("Tnf", "Ddx58", 0.24433255),
    ("Tnf", "Gbp2", 0.24165079),
    ("Il27", "Stat2", 0.23349733),
    ("Il27", "Ifi35", 0.22961687)
]
top_10_ligand_receptor_links_focused = [
    ("Il2", "Il2rb", 1.4541219),
    ("H2-D1", "Cd8a", 1.4430922),
    ("H2-K1", "Cd8a", 1.4430922),
    ("H2-M3", "Cd8a", 1.4430922),
    ("H2-Q4", "Cd8a", 1.4430922),
    ("H2-Q6", "Cd8a", 1.4430922),
    ("H2-Q7", "Cd8a", 1.4430922),
    ("Il27", "Il27ra", 1.2787938),
    ("Ptprc", "Cd247", 1.1547412),
    ("H2-M2", "Cd8a", 1.1484508)
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
    ligand_activities_sorted = sorted(
        ligand_activities.items(),
        key=lambda x : (-x[1]["aupr_corrected"], x[0]),
    )
    assert len(ligand_activities_sorted) == 475
    assert equals_iter(
        ((ligand, act["aupr_corrected"]) for ligand, act in ligand_activities_sorted[:10]),
        top_10_ligand_activities
    )
    best_upstream_ligands = [e[0] for e in ligand_activities_sorted[:30]]
    active_ligand_target_links = [
        predictor.get_weighted_ligand_target_links(ligand, geneset, n=100)
        for ligand in best_upstream_ligands
    ]
    # combine weighted ligand-target links of different ligands
    active_ligand_target_links = list(
        chain(
            *(zip(cycle([e["ligand"]]), e["target"], e["weight"]) for e in active_ligand_target_links)
        )
    )
    assert len(active_ligand_target_links) == 579
    assert equals_iter(
        sorted(active_ligand_target_links, key=lambda x : x[2], reverse=True)[:10],
        top_10_active_ligand_target_links
    )
    ligand_receptor_links = get_weighted_ligand_receptor_links(
        best_upstream_ligands,
        expressed_receptors,
        lr_network,
        lr_sig
    )
    assert len(ligand_receptor_links) == 52
    assert equals_iter(
        sorted(ligand_receptor_links._mapping, key=lambda x : x[2], reverse=True)[:10],
        top_10_ligand_receptor_links
    )
    ligand_activities = dict(
        (key, val) for key, val in ligand_activities.items() if key in potential_ligands_focused
    )
    ligand_activities_sorted = sorted(
        ligand_activities.items(),
        key=lambda x : (-x[1]["aupr_corrected"], x[0])
    )
    assert len(ligand_activities_sorted) == 122
    assert equals_iter(
        ((ligand, act["aupr_corrected"]) for ligand, act in ligand_activities_sorted[:10]),
        top_10_ligand_activities_focused
    )
    best_upstream_ligands = [e[0] for e in ligand_activities_sorted[:30]]
    active_ligand_target_links = [
        predictor.get_weighted_ligand_target_links(ligand, geneset, n=100)
        for ligand in best_upstream_ligands
    ]
    active_ligand_target_links = list(
        chain(
            *(zip(cycle([e["ligand"]]), e["target"], e["weight"]) for e in active_ligand_target_links)
        )
    )
    assert len(active_ligand_target_links) == 313
    assert equals_iter(
        sorted(active_ligand_target_links, key=lambda x : x[2], reverse=True)[:10],
        top_10_active_ligand_target_links_focused
    )
    ligand_receptor_links = get_weighted_ligand_receptor_links(
        best_upstream_ligands,
        expressed_receptors,
        lr_network,
        lr_sig
    )
    assert len(ligand_receptor_links) == 54
    assert equals_iter(
        sorted(ligand_receptor_links._mapping, key=lambda x : x[2], reverse=True)[:10],
        top_10_ligand_receptor_links_focused
    )
    sub_ann = subset_ann(
        ann,
        val=sender_celltypes,
        val_col="celltype",
        layers=["data"]
    )
    sub_ann.var = ann.var
    sub_ann.X = sub_ann.layers["data"]
    lfcs = combine_by_key(*(
        get_lfc_celltype(
            ann,
            celltype,
            "aggregate",
            condition_oi="LCMV",
            condition_ref="SS",
            layer="data",
            features=best_upstream_ligands
        )
        for celltype in sender_celltypes
    ))
    df = pd.DataFrame(data=lfcs)
    df.rename(index=dict(enumerate(sender_celltypes)), inplace=True)
    assert equals(df["Il27"]["CD4 T"], 1.3103791)
    assert equals(df["Ebi3"]["Treg"], 4.22547764)
    assert equals(df["Tnf"]["Mono"], 1.24955145)
    assert equals(df["H2-Eb1"]["NK"], -1.5845094)
    assert equals(df["Ptprc"]["B"], 0.44807439)
    assert equals(df["Il2"]["DC"], 1.3173053)

def test_wrapper():
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
    sender_celltypes = ("CD4 T", "Treg", "Mono", "NK", "B", "DC")
    res = run_nichenet(
        ann,
        predictor,
        lr_network,
        "CD8 T",
        "aggregate",
        "LCMV",
        "SS",
        sender_celltypes=sender_celltypes,
        lr_sig=lr_sig,
        get_ltl=True,
        get_lfc=True,
        expression_pct=0.05,
        targets_top_n=100
    )
    ligand_activities_sorted = res["ligand_activities_sorted"]
    active_ligand_target_links = res["active_ligand_target_links"]
    ligand_receptor_links = res["ligand_receptor_links"]
    ligand_activities_sorted_focused = res["ligand_activities_sorted_focused"]
    active_ligand_target_links_focused = res["active_ligand_target_links_focused"]
    ligand_receptor_links_focused = res["ligand_receptor_links_focused"]
    lfcs = res["lfcs"]
    geneset = res["geneset_oi"]
    expressed_genes_receiver = res["expressed_genes_receiver"]
    assert len(geneset) == 241
    assert len(expressed_genes_receiver) == 3903
    assert len(ligand_activities_sorted) == 475
    assert equals_iter(
        ((ligand, act["aupr_corrected"]) for ligand, act in ligand_activities_sorted[:10]),
        top_10_ligand_activities
    )
    assert len(active_ligand_target_links) == 579
    assert equals_iter(
        sorted(active_ligand_target_links, key=lambda x : x[2], reverse=True)[:10],
        top_10_active_ligand_target_links
    )
    assert len(ligand_receptor_links) == 52
    assert equals_iter(
        sorted(ligand_receptor_links._mapping, key=lambda x : x[2], reverse=True)[:10],
        top_10_ligand_receptor_links
    )
    assert len(ligand_activities_sorted_focused) == 122
    print([(ligand, act["aupr_corrected"]) for ligand, act in ligand_activities_sorted_focused[:10]])
    assert equals_iter(
        ((ligand, act["aupr_corrected"]) for ligand, act in ligand_activities_sorted_focused[:10]),
        top_10_ligand_activities_focused
    )
    assert len(active_ligand_target_links_focused) == 313
    assert equals_iter(
        sorted(active_ligand_target_links_focused, key=lambda x : x[2], reverse=True)[:10],
        top_10_active_ligand_target_links_focused
    )
    assert len(ligand_receptor_links_focused) == 54
    assert equals_iter(
        sorted(ligand_receptor_links_focused._mapping, key=lambda x : x[2], reverse=True)[:10],
        top_10_ligand_receptor_links_focused
    )
    df = pd.DataFrame(data=combine_by_key(*lfcs))
    df.rename(index=dict(enumerate(sender_celltypes)), inplace=True)
    assert equals(df["Il27"]["CD4 T"], 1.3103791)
    assert equals(df["Ebi3"]["Treg"], 4.22547764)
    assert equals(df["Tnf"]["Mono"], 1.24955145)
    assert equals(df["H2-Eb1"]["NK"], -1.5845094)
    assert equals(df["Ptprc"]["B"], 0.44807439)
    assert equals(df["Il2"]["DC"], 1.3173053)
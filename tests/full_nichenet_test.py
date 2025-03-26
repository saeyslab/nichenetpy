from nichenetpy.utils import combine_by_key
from nichenetpy.extraction import (
    get_expressed_genes,
    subset_ann,
    get_weighted_ligand_receptor_links,
    get_lfc_celltype
)
from nichenetpy.gene_symbol import (
    mouse_alias_info,
    human_alias_info
)
from nichenetpy.metrics import group_metrics
from nichenetpy.wrappers import (
    run_nichenet,
    combine_weighted_ligand_target_links,
    get_weighted_ligand_receptor_links,
)
from nichenetpy.io import read_csc_matrix
from nichenetpy.utils import (
    read_csv_rows,
    subset_matrix
)
from nichenetpy.normalization import scale_quantile

from itertools import cycle, chain
from numbers import Number
from collections.abc import Iterable
from math import log

import anndata
import os
import requests
import pickle
import pandas as pd
import numpy as np


err_bound = 1e-3

root_path = os.path.normpath("./tests/data/tutorial_files")
ann_path = os.path.join(root_path, "AnnData")
hnscc_path = os.path.join(root_path, "hnscc")

BASIC_top_10_ligand_activities = [
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
BASIC_top_10_active_ligand_target_links = [
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
BASIC_top_10_ligand_receptor_links = [
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
BASIC_top_10_ligand_activities_focused = [
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
BASIC_top_10_active_ligand_target_links_focused = [
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
BASIC_top_10_ligand_receptor_links_focused = [
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
LAG_top_10_ligand_activities = [
    ("TGFB2", 0.10465456),
    ("BMP8A", 0.06994420),
    ("INHBA", 0.06846069),
    ("CXCL12", 0.06762803),
    ("LTBP1", 0.06094220),
    ("CCN2", 0.05813802),
    ("TNXB", 0.05644615),
    ("ENG", 0.05506822),
    ("BMP5", 0.05380455),
    ("VCAN", 0.05342251)
]
LAG_top_10_active_ligand_target_links = [
    ("TGFB2", "SERPINE1", 0.26148032),
    ("HGF", "MMP1", 0.24897915),
    ("CXCL12", "SERPINE1", 0.24490880),
    ("TGFB2", "COL1A1", 0.24316609),
    ("CXCL12", "MT2A", 0.22410012),
    ("BMP4", "PLOD2", 0.19921317),
    ("CXCL12", "TGFBI", 0.19480944),
    ("HGF", "PLAU", 0.19166812),
    ("HGF", "SERPINE1", 0.18975110),
    ("HGF", "F3", 0.18298783)
]
LAG_top_10_ligand_receptor_links = [
    ("HGF", "MET", 1.9353637),
    ("TIMP2", "MMP2", 1.4827621),
    ("MMP14", "MMP2", 1.4273549),
    ("MMP14", "MMP13", 1.4187061),
    ("VCAM1", "ITGB1", 1.2688764),
    ("CD47", "SIRPA", 1.2628425),
    ("TGFB2", "TGFBR1", 1.1137881),
    ("FN1", "ITGA5", 1.0053305),
    ("INHBA", "ACVR2A", 0.9669868),
    ("CFH", "CFB", 0.9611272)
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
    with open(os.path.join(root_path, filename), "rb") as file:
        model = pickle.loads(file.read())
    return model

def get_anndata_file(filename):
    if not os.path.exists(ann_path):
        os.makedirs(ann_path)
    file_path = os.path.join(ann_path, filename)
    if not os.path.exists(file_path):
        res = requests.get(f"https://zenodo.org/records/14859451/files/{filename}")
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
            res = requests.get(f"https://zenodo.org/records/14859451/files/{filename}")
            with open(file_path, "wb") as file:
                file.write(res.content)

def get_exp(mat, cols):
    agg_exp = [
        log(sum((10*(2**x - 1) for x in mat[:, i]))/mat.shape[0] + 1, 2)
        for i in range(mat.shape[1])
    ]
    return {gene for gene, x in zip(cols, agg_exp) if x >= 4}

def test_steps():
    model = get_model_pickle("mouse")
    ann = get_anndata_file("annData3531889.h5")
    ann.var_names = ann.var["gene"]
    mouse_alias_info.alias_to_symbol(ann)
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
        BASIC_top_10_ligand_activities
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
        BASIC_top_10_active_ligand_target_links
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
        BASIC_top_10_ligand_receptor_links
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
        BASIC_top_10_ligand_activities_focused
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
        BASIC_top_10_active_ligand_target_links_focused
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
        BASIC_top_10_ligand_receptor_links_focused
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
    model = get_model_pickle("mouse")
    ann = get_anndata_file("annData3531889.h5")
    ann.var_names = ann.var["gene"]
    mouse_alias_info.alias_to_symbol(ann)
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
        BASIC_top_10_ligand_activities
    )
    assert len(active_ligand_target_links) == 579
    assert equals_iter(
        sorted(active_ligand_target_links, key=lambda x : x[2], reverse=True)[:10],
        BASIC_top_10_active_ligand_target_links
    )
    assert len(ligand_receptor_links) == 52
    assert equals_iter(
        sorted(ligand_receptor_links._mapping, key=lambda x : x[2], reverse=True)[:10],
        BASIC_top_10_ligand_receptor_links
    )
    assert len(ligand_activities_sorted_focused) == 122
    print([(ligand, act["aupr_corrected"]) for ligand, act in ligand_activities_sorted_focused[:10]])
    assert equals_iter(
        ((ligand, act["aupr_corrected"]) for ligand, act in ligand_activities_sorted_focused[:10]),
        BASIC_top_10_ligand_activities_focused
    )
    assert len(active_ligand_target_links_focused) == 313
    assert equals_iter(
        sorted(active_ligand_target_links_focused, key=lambda x : x[2], reverse=True)[:10],
        BASIC_top_10_active_ligand_target_links_focused
    )
    assert len(ligand_receptor_links_focused) == 54
    assert equals_iter(
        sorted(ligand_receptor_links_focused._mapping, key=lambda x : x[2], reverse=True)[:10],
        BASIC_top_10_ligand_receptor_links_focused
    )
    df = pd.DataFrame(data=combine_by_key(*lfcs))
    df.rename(index=dict(enumerate(sender_celltypes)), inplace=True)
    assert equals(df["Il27"]["CD4 T"], 1.3103791)
    assert equals(df["Ebi3"]["Treg"], 4.22547764)
    assert equals(df["Tnf"]["Mono"], 1.24955145)
    assert equals(df["H2-Eb1"]["NK"], -1.5845094)
    assert equals(df["Ptprc"]["B"], 0.44807439)
    assert equals(df["Il2"]["DC"], 1.3173053)

def test_ligand_activity_geneset():
    model = get_model_pickle("human")
    get_hnscc_file()
    predictor = model["predictor"]
    lr_network = model["lr_network"]
    lr_sig = model["lr_sig"]
    exp_mat, rows, cols = read_csc_matrix(os.path.join(hnscc_path, "hnscc_expression.bin"))
    cols = human_alias_info.alias_to_symbol(cols)
    sample_info_col_names, sample_info = read_csv_rows(os.path.join(hnscc_path, "sample_info.csv"))
    sample_info = [
        [
            int(processed_by_Maxima_enzyme),
            int(Lymph_node),
            int(classified_as_cancer_cell),
            int(classified_as_non_cancer_cells),
            non_cancer_cell_type,
            cell,
            tumor
        ]
        for
            processed_by_Maxima_enzyme,
            Lymph_node,
            classified_as_cancer_cell,
            classified_as_non_cancer_cells,
            non_cancer_cell_type,
            cell,
            tumor
        in sample_info
    ]
    tumors_remove = {"HN10","HN","HN12", "HN13", "HN24", "HN7", "HN8","HN23"}
    CAF_cells = [e[5] for e in sample_info if e[1] == 0 and e[4] == "CAF" and e[6] not in tumors_remove]
    malignant_cells = [e[5] for e in sample_info if e[1] == 0 and e[2] == 1 and e[6] not in tumors_remove]
    row2id = dict(zip(rows, range(len(rows))))
    exp_mat = np.array(exp_mat.todense())
    expressed_genes_sender = get_exp(subset_matrix(exp_mat, [row2id[cell] for cell in CAF_cells]), cols)
    assert len(expressed_genes_sender) == 6706
    expressed_genes_receiver = get_exp(subset_matrix(exp_mat, [row2id[cell] for cell in malignant_cells]), cols)
    assert len(expressed_genes_receiver) == 6351
    ligands = lr_network.get_ligands()
    expressed_ligands = ligands.intersection(expressed_genes_sender)
    receptors = lr_network.get_receptors()
    expressed_receptors = receptors.intersection(expressed_genes_receiver)
    potential_ligands = {ligand for ligand, receptor in lr_network if ligand in expressed_ligands and receptor in expressed_receptors}
    assert len(potential_ligands) == 212
    with open(os.path.join(hnscc_path, "pemt_signature.txt")) as file:
        geneset = file.readlines()
    lt_ligands = set(predictor.row_names)
    geneset = {gene.rstrip() for gene in geneset}.intersection(lt_ligands)
    assert len(geneset) == 96
    background_expressed_genes = expressed_genes_receiver.intersection(lt_ligands)
    #assert len(background_expressed_genes) == 6288
    ligand_activities = predictor.predict_ligand_activities(
        geneset,
        background_expressed_genes,
        potential_ligands
    )
    ligand_activities_sorted = sorted(
        ligand_activities.items(),
        key=lambda x : (-x[1]["aupr_corrected"], x[0])
    )
    assert len(ligand_activities_sorted) == 212
    assert equals_iter(
        ((ligand, act["aupr_corrected"]) for ligand, act in ligand_activities_sorted[:10]),
        LAG_top_10_ligand_activities
    )
    best_upstream_ligands = [e[0] for e in ligand_activities_sorted[:30]]
    active_ligand_target_links = combine_weighted_ligand_target_links((
        predictor.get_weighted_ligand_target_links(ligand, geneset, n=200)
        for ligand in best_upstream_ligands
    ))
    assert len(active_ligand_target_links) == 405
    assert equals_iter(
        sorted(active_ligand_target_links, key=lambda x : x[2], reverse=True)[:10],
        LAG_top_10_active_ligand_target_links
    )
    ligand_receptor_links = get_weighted_ligand_receptor_links(
        best_upstream_ligands,
        expressed_receptors,
        lr_network,
        lr_sig
    )
    assert len(ligand_receptor_links) == 84
    assert equals_iter(
        sorted(ligand_receptor_links._mapping, key=lambda x : x[2], reverse=True)[:10],
        LAG_top_10_ligand_receptor_links
    )
    col2id = dict(zip(cols, range(len(cols))))
    exp_mat_caf = subset_matrix(
        exp_mat,
        [row2id[cell] for cell in CAF_cells],
        [col2id[ligand] for ligand in best_upstream_ligands]
    )
    exp_df_caf = pd.DataFrame(
        exp_mat_caf,
        index=CAF_cells,
        columns=best_upstream_ligands
    )
    exp_df_caf.index.name = 'cell'
    exp_df_caf.reset_index(inplace=True)
    sample_info = pd.DataFrame(
        sample_info,
        columns=sample_info_col_names
    )
    exp_df_caf = exp_df_caf.merge(sample_info[["cell", "tumor"]], on="cell", how="inner")
    agg_exp_caf = exp_df_caf[exp_df_caf.columns.difference(["cell"])].groupby("tumor").mean()
    agg_exp_caf.reset_index(inplace=True)
    tumors = list(agg_exp_caf["tumor"])
    agg_exp_caf = agg_exp_caf[agg_exp_caf.columns.difference(["tumor"])]
    agg_exp_caf = agg_exp_caf.transpose()
    agg_exp_caf.columns = tumors
    # This order was determined based on the paper from Puram et al. Tumors are ordered according to p-EMT score.
    agg_exp_caf = agg_exp_caf[["HN6","HN20","HN26","HN28","HN22","HN25","HN5","HN18","HN17","HN16"]]
    assert equals(agg_exp_caf["HN16"]["TGFB2"], 1.6119743)
    assert equals(agg_exp_caf["HN17"]["BMP8A"], 0.39233327)
    assert equals(agg_exp_caf["HN18"]["INHBA"], 1.880096)
    assert equals(agg_exp_caf["HN20"]["CXCL12"], 4.5775333)
    assert equals(agg_exp_caf["HN22"]["LTBP1"], 4.626786)
    assert equals(agg_exp_caf["HN25"]["CCN2"], 7.100225)
    assert equals(agg_exp_caf["HN26"]["TNXB"], 0.3709426)
    assert equals(agg_exp_caf["HN28"]["ENG"], 2.0476525)
    assert equals(agg_exp_caf["HN5"]["BMP5"], 0.1361916)
    assert equals(agg_exp_caf["HN6"]["VCAN"], 3.282618)
    exp_mat_target = subset_matrix(
        exp_mat,
        [row2id[cell] for cell in malignant_cells],
        [col2id[gene] for gene in geneset]
    )
    exp_df_target = pd.DataFrame(
        exp_mat_target,
        index=malignant_cells,
        columns=list(geneset)
    )
    exp_df_target.index.name = 'cell'
    exp_df_target.reset_index(inplace=True)
    exp_df_target = exp_df_target.merge(sample_info[["cell", "tumor"]], on="cell", how="inner")
    agg_exp_target = exp_df_target[exp_df_target.columns.difference(["cell"])].groupby("tumor").mean()
    agg_exp_target.reset_index(inplace=True)
    tumors = list(agg_exp_target["tumor"])
    agg_exp_target = agg_exp_target[agg_exp_target.columns.difference(["tumor"])]
    agg_exp_target = agg_exp_target.transpose()
    agg_exp_target.columns = tumors
    targets = list({target for _, target, _ in active_ligand_target_links})
    # This order was determined based on the paper from Puram et al. Tumors are ordered according to p-EMT score.
    agg_exp_target = agg_exp_target[["HN6","HN20","HN26","HN28","HN22","HN25","HN5","HN18","HN17","HN16"]]
    agg_exp_target = agg_exp_target.loc[targets]
    agg_exp_target = pd.DataFrame(
        scale_quantile(agg_exp_target.to_numpy()),
        index=agg_exp_target.index,
        columns=agg_exp_target.columns
    )
    agg_exp_target = agg_exp_target.transpose()
    assert equals(agg_exp_target["SERPINE1"]["HN16"], 0.6207398)
    assert equals(agg_exp_target["TGFBI"]["HN17"], 0.85252048)
    #TODO
from nichenetpy.utils import (
    combine_by_key,
    ligand_activities_df
)
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
    generate_info_tables,
    normalize_single_cell_ligand_activities
)
from nichenetpy.io import read_csc_matrix
from nichenetpy.utils import (
    read_csv_rows,
    read_csv_cols,
    subset_matrix
)
from nichenetpy.normalization import scale_quantile
from nichenetpy.prioritization import (
    generate_prioritization_table
)
from nichenetpy.visualization import get_ligand_signaling_path
from nichenetpy.prediction import assess_rf_class_probabilities
from nichenetpy.model_construction import (
    construct_weighted_networks,
    apply_hub_correction,
    construct_ligand_target_matrix
)

from itertools import cycle, chain, repeat
from numbers import Number
from collections.abc import Iterable
from math import log
from scipy.stats import pearsonr

import anndata
import os
import requests
import pickle
import pandas as pd
import numpy as np


root_path = os.path.normpath("./tests/data/tutorial_files")
ann_path = os.path.join(root_path, "AnnData")
hnscc_path = os.path.join(root_path, "hnscc")
network_path = os.path.join(root_path, "model_construction", "human")

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
LTSP_top_10_tf_signaling = [
    ("SMAD4", "SMAD3", 1.7500000),
    ("SMAD3", "SMAD4", 1.6443880),
    ("TGFBR2", "SMAD3", 1.5239365),
    ("TGFB2", "TGFBR2", 1.4895029),
    ("SP1", "SMAD3", 1.3610625),
    ("TGFB2", "SMAD3", 1.3594946),
    ("SMAD3", "SP1", 1.3544942),
    ("SMAD3", "TGFBR2", 1.3185690),
    ("SMAD4", "SP1", 1.2399626),
    ("TGFBR2", "SMAD4", 1.1920359)
]
LTSP_tf_regulatory = [
    ("SP1", "COL1A1", 1.7500000),
    ("SMAD3", "SERPINE1", 1.6759015),
    ("SP1", "SERPINE1", 1.6137685),
    ("SMAD4", "SERPINE1", 1.5588446),
    ("TGFB2", "COL1A1", 1.4040920),
    ("TGFB2", "SERPINE1", 1.4021392),
    ("NFKB1", "COL1A1", 1.0721552),
    ("NFKB1", "SERPINE1", 1.0010732),
    ("SMAD3", "COL1A1", 0.9130368),
    ("TGFBR2", "SERPINE1", 0.8196570),
    ("SMAD4", "COL1A1", 0.7500000)
]
TPEG_top_10_gene_predictions_0 = [
    ("Trim12a", 0, 0.961),
    ("Trim30c", 1, 0.961),
    ("Gbp8", 1, 0.955),
    ("Gbp6", 1, 0.955),
    ("Rpl34-ps1", 0, 0.953),
    ("Gimap9", 0, 0.950),
    ("Ms4a4c", 1, 0.950),
    ("Ly6c2", 1, 0.950),
    ("H2-Q4", 0, 0.941),
    ("H2-D1", 1, 0.941)
]
LASC_scaled_expression_4_5 = [
    0.88919062,
    0.02170838,
    0,
    0,
    0,
    0.64007038,
    0.04018212,
    0,
    0,
    0,
    0.10484696,
    0,
    0,
    0,
    0,
    0.82183122,
    0,
    0.58461674,
    0.728983487,
    0.18617532
]
LASC_top_10_ligand_activities = [
    ("HNSCC5_p3_HNSCC5_P3_H01", "ANGPTL4", 0.6454203, 0.06823189, 0.1249527032),
    ("HNSCC5_p3_HNSCC5_P3_H01", "FGF7", 0.6603868, 0.06610786, 0.1176608430),
    ("HNSCC5_p3_HNSCC5_P3_G07", "TGFB2", 0.5077907, 0.06588319, 0.0004083035),
    ("HNSCC5_p3_HNSCC5_P3_G07", "ANGPTL2", 0.5080898, 0.06565360, 0.0102401022),
    ("HNSCC5_p3_HNSCC5_P3_G07", "APP", 0.4959796, 0.06504474, 0.0198369284),
    ("HNSCC5_p3_HNSCC5_P3_G07", "IL24", 0.4959934, 0.06501017, 0.0013083112),
    ("HNSCC5_p3_HNSCC5_P3_G07", "CLCF1", 0.5100628, 0.06495958, 0.0023989515),
    ("HNSCC5_p3_HNSCC5_P3_G07", "ANXA2", 0.4846088, 0.06448384, 0.0050817067),
    ("HNSCC5_p3_HNSCC5_P3_G07", "JAG1", 0.4932325, 0.06444520, 0.0089657196),
    ("HNSCC5_p3_HNSCC5_P3_G07", "CLU", 0.4883133, 0.06377363, -0.0011606326)
]
LASC_cell_scores = [
    ("HNSCC5_p3_HNSCC5_P3_F07", 1),
    ("HNSCC5_p3_HNSCC5_P3_E03", 0.9910366),
    ("HNSCC5_p3_HNSCC5_P3_G07", 0.9346274),
    ("HNSCC5_p3_HNSCC5_P3_H01", 0.8563181),
    ("HNSCC5_p9_HNSCC5_P9_D08", 0.8466327),
    ("HNSCC5_p3_HNSCC5_P3_A04", 0.7845420),
    ("HNSCC5_p3_HNSCC5_P3_D10", 0.7150538),
    ("HNSCC5_p9_HNSCC5_P9_B10", 0.6803033),
    ("HNSCC5_p9_HNSCC5_P9_D03", 0.6274404),
    ("HNSCC5_p3_HNSCC5_P3_C11", 0.5479999)
]
LASC_top_10_output_correlation = [
    ("OGN", 0.829840982),
    ("ANGPTL2", 0.740343113),
    ("CXCL10", 0.582682959),
    ("NID1", 0.576785251),
    ("MMP14", 0.571631596),
    ("CXCL12", 0.560159409),
    ("COL11A1", 0.554847989),
    ("BGN", 0.549549614),
    ("CLCF1", 0.526221930),
    ("TFPI", 0.509691274)
]
MC_top_10_lr_sig_0 = {
    ("GSK3B", "FRAT1"): 8.362857,
    ("LCK", "LCP2"): 7.723804,
    ("LCK", "ITK"): 7.578648,
    ("LCK", "ZAP70"): 7.523063,
    ("IL4", "IL4R"): 7.485833,
    ("PTK6", "STAP2"): 7.412247,
    ("LCK", "VAV1"): 7.408178,
    ("MYD88", "IRAK4"): 7.379728,
    ("CSNK1E", "PER2"): 7.184271,
    ("TBK1", "IRF3"): 7.176711
}
MC_top_10_gr_0 = {
    ("EGR1", "NAB2"): 9.121336,
    ("TP53", "CDKN1A"): 8.876599,
    ("ESR1", "GREB1"): 8.407263,
    ("MYC", "ODC1"): 8.011052,
    ("MYC", "CDK4"): 7.863986,
    ("GATA1", "HEMGN"): 7.845768,
    ("ESR1", "TFF1"): 7.794308,
    ("HNF4A", "HNF1A"): 7.705854,
    ("GATA1", "EPOR"): 7.620628,
    ("MYC", "PAICS"): 7.556020
}
MC_top_10_lr_sig_1 = {
    ("GSK3B", "FRAT1"): 2.364072,
    ("MAPK14", "MAPKAPK2"): 2.225577,
    ("MAPK1", "ELK1"): 2.224897,
    ("LCK", "LCP2"): 2.193538,
    ("CSNK1E", "PER2"): 2.193134,
    ("LCK", "ITK"): 2.186991,
    ("STK11", "STRADA"): 2.185282,
    ("MAPK1", "DUSP1"): 2.182768,
    ("LCK", "VAV1"): 2.165965,
    ("PTK6", "STAP2"): 2.152541
}
MC_top_10_gr_1 = {
    ("TP53", "CDKN1A"): 2.474588,
    ("ESR1", "GREB1"): 2.410098,
    ("MYC", "CDK4"): 2.383487,
    ("MYC", "TERT"): 2.338537,
    ("STAT3", "SOCS3"): 2.293794,
    ("EGR1", "NAB2"): 2.292260,
    ("STAT3", "BCL6"): 2.280768,
    ("MYC", "PAICS"): 2.252887,
    ("MYC", "FASN"): 2.238074,
    ("MYC", "SRM"): 2.231288
}
MC_top_10_lr_sig_2 = {
    ("LY86", "CD180"): 5.541908,
    ("IRAK1", "IRAK3"): 5.483837,
    ("GSK3B", "FRAT1"): 5.340887,
    ("CDC7", "DBF4"): 5.323177,
    ("RSPO1", "RNF43"): 5.289391,
    ("CLCF1", "CRLF1"): 5.265573,
    ("LCK", "ITK"): 5.174903,
    ("EGFR", "GPNMB"): 5.173386,
    ("CSF1", "CSF1R"): 5.076013,
    ("KIT", "SH2B3"): 5.059568
}
MC_top_10_gr_2 = {
    ("GATA1", "HEMGN"): 6.804552,
    ("ESR1", "GREB1"): 6.621335,
    ("ESR1", "RARA"): 6.202072,
    ("REST", "GRIN1"): 6.171134,
    ("EGR1", "NAB2"): 6.135783,
    ("MYC", "CDK4"): 6.114949,
    ("REST", "GLRA1"): 6.039138,
    ("ESR1", "TFF1"): 6.006676,
    ("GATA1", "GFI1B"): 5.973500,
    ("SPI1", "NCF2"): 5.964865
}
MC_top_10_ligand_target_matrix_PPR_0 = [
    (0, 0),
    (1.095173e-02, 1.216394e-02),
    (5.084809e-03, 5.172813e-03),
    (1.249955e-02, 1.348655e-02),
    (3.030508e-01, 2.451550e-01),
    (3.341745e-03, 3.556587e-03),
    (1.549648e-01, 8.327667e-02),
    (4.062058e-03, 6.028271e-03),
    (4.627761e-03, 5.194608e-03),
    (1.554942e-01, 9.057184e-02)
]
MC_top_10_ligand_target_matrix_SPL_0 = [
    (0, 0),
    (183.030059, 189.253667),
    (93.526666, 101.429093),
    (267.723330, 267.135023),
    (528.769601, 521.107878),
    (83.730980, 83.682842),
    (189.540212, 200.824258),
    (46.706778, 50.200017),
    (102.604214, 112.111570),
    (312.282657, 335.296730)
]
MC_top_10_ligand_target_matrix_direct_0 = [
    (0, 0),
    (26.6568522, 19.1356107),
    (10.9007838, 6.8273971),
    (32.6182852, 22.4380537),
    (93.1365089, 80.5938133),
    (9.7052011, 6.7828120),
    (34.0457086, 21.1353234),
    (9.9244653, 9.8564317),
    (9.8125185, 7.1930697),
    (45.6791846, 36.4299388)
]
MC_top_10_ligand_target_matrix_PPR_1 = [
    0,
    2.488822e-03,
    7.229267e-04,
    3.258491e-03,
    1.106382e-01,
    3.157756e-04,
    5.457279e-02,
    5.690325e-04,
    5.049286e-04,
    4.036876e-02
]
MC_top_10_ligand_target_matrix_SPL_1 = [
    0,
    277.733194,
    96.906154,
    359.753151,
    1078.687723,
    49.600525,
    427.305863,
    40.870570,
    90.409308,
    513.872206
]
MC_top_10_ligand_target_matrix_direct_1 = [
    0,
    2.15539955,
    0.57405933,
    2.56142133,
    10.25546831,
    0.36363283,
    3.11143536,
    0.44221322,
    0.23877735,
    4.31137936
]
MC_top_10_ligand_target_matrix_PPR_2 = [
    0.0082132096,
    0.0049895962,
    0.0082667868,
    0.0185473962,
    0.0028839713,
    0.0076032847,
    0.0043716716,
    0.0044186443,
    0.1559356676,
    0.0017981498
]
MC_top_10_ligand_target_matrix_SPL_2 = [
    80.510371,
    62.470823,
    105.588839,
    245.624279,
    60.187512,
    115.459780,
    44.534696,
    62.472138,
    191.286479,
    41.827194
]
MC_top_10_ligand_target_matrix_direct_2 = [
    6.9358440,
    5.0016499,
    10.1736364,
    26.6262723,
    3.5195458,
    9.7560967,
    5.8818588,
    5.6810110,
    14.9366186,
    1.6602571
]

def equals(
    x,
    y,
    err_bound=1e-2,
    zero_bound=1e-100
):
    if isinstance(x, Number) and isinstance(y, Number):
        return abs(x) < zero_bound if y == 0 else abs(x - y) / y < err_bound
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

def get_network_files():
    if not os.path.exists(network_path):
        os.makedirs(network_path)
    for filename in (
        "gr_human.csv",
        "lr_network_human.csv",
        "lr_sig_human.csv",
        "optimized_source_weights.csv",
        "annotation_data_sources.csv"
    ):
        file_path = os.path.join(network_path, filename)
        if not os.path.exists(file_path):
            res = requests.get(f"https://zenodo.org/records/14929618/files/{filename}")
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
    exp_mat = exp_mat.toarray()
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
    # assert len(background_expressed_genes) == 6288
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
    assert equals(agg_exp_target["HN16"]["SERPINE1"], 2.3770934)
    assert equals(agg_exp_target["HN17"]["TGFBI"], 5.3953940)
    assert equals(agg_exp_target["HN18"]["MMP10"], 1.0519966)
    assert equals(agg_exp_target["HN20"]["LAMC2"], 1.913600975)
    assert equals(agg_exp_target["HN22"]["P4HA2"], 2.52934714)
    agg_exp_target = pd.DataFrame(
        scale_quantile(agg_exp_target.to_numpy(), by_row=True),
        index=agg_exp_target.index,
        columns=agg_exp_target.columns
    )
    agg_exp_target = agg_exp_target.transpose()
    assert equals(agg_exp_target["SERPINE1"]["HN16"], 0.6207398)
    assert equals(agg_exp_target["TGFBI"]["HN17"], 0.85252048)
    assert equals(agg_exp_target["MMP10"]["HN18"], 0.47556523)
    assert equals(agg_exp_target["LAMC2"]["HN20"], 0.15968678)
    assert equals(agg_exp_target["P4HA2"]["HN22"], 0.49290029)

def test_steps_prioritization():
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
        expression_pct=0.05,
        targets_top_n=100
    )
    ligand_activities_sorted = res["ligand_activities_sorted_focused"]
    best_upstream_ligands = res["best_upstream_ligands_focused"]
    expressed_ligands = res["expressed_ligands"]
    expressed_receptors = res["expressed_receptors"]
    lr_network_filtered = lr_network.subset_sep(expressed_ligands, expressed_receptors)
    info_tables = generate_info_tables(
        ann,
        "celltype",
        sender_celltypes,
        ["CD8 T"],
        lr_network_filtered,
        "aggregate",
        "LCMV",
        "SS",
        case_control=True
    )
    df = info_tables["sender_receiver_de"]
    row = df[
        (df["sender"] == "DC") &
        (df["receiver"] == "CD8 T") &
        (df["ligand"] == "H2-M2") &
        (df["receptor"] == "Cd8a")
    ].iloc[0]
    assert equals(row["lfc_ligand"], 11.0024120)
    assert equals(row["lfc_receptor"], 2.383806589)
    assert equals(row["ligand_receptor_lfc_avg"], 6.693109)
    assert equals(row["pval_ligand"], 1.017174e-272)
    assert equals(row["pval_adj_ligand"], 1.377355e-268)
    assert equals(row["pval_receptor"], 5.250531e-206)
    assert equals(row["pval_adj_receptor"], 7.109745e-202)
    assert equals(row["pct_expressed_sender"], 0.429)
    assert equals(row["pct_expressed_receiver"], 0.659)
    df = info_tables["sender_receiver_info"]
    row = df[
        (df["sender"] == "DC") &
        (df["receiver"] == "Mono") &
        (df["ligand"] == "B2m") &
        (df["receptor"] == "Tap1")
    ].iloc[0]
    assert equals(row["avg_ligand"], 216.171733)
    assert equals(row["avg_receptor"], 8.5863090)
    assert equals(row["ligand_receptor_prod"], 1856.1173)
    df = info_tables["lr_condition_de"]
    row = df[
        (df["ligand"] == "Cxcl11") &
        (df["receptor"] == "Dpp4")
    ].iloc[0]
    assert equals(row["lfc_ligand"], 7.1973441001)
    assert equals(row["lfc_receptor"], 0.7345097723)
    assert equals(row["ligand_receptor_lfc_avg"], 3.96592694)
    assert equals(row["pval_ligand"], 1.621364e-04)
    assert equals(row["pval_adj_ligand"], 1)
    assert equals(row["pval_receptor"], 1.170731e-06)
    assert equals(row["pval_adj_receptor"], 1.585287e-02)
    prior_table = generate_prioritization_table(
        info_tables["sender_receiver_info"],
        info_tables["sender_receiver_de"],
        ligand_activities_sorted,
        info_tables["lr_condition_de"]
    )
    row = prior_table[
        (prior_table["sender"] == "NK") &
        (prior_table["receiver"] == "CD8 T") &
        (prior_table["ligand"] == "Ptprc") &
        (prior_table["receptor"] == "Dpp4")
    ].iloc[0]
    assert len(prior_table) == 1212
    assert equals(row["lfc_ligand"], 0.64193917)
    assert equals(row["lfc_receptor"], 0.299171963)
    assert equals(row["ligand_receptor_lfc_avg"], 0.47055556)
    assert equals(row["pval_ligand"], 2.182674e-07)
    assert equals(row["pval_adj_ligand"], 2.955559e-03)
    assert equals(row["pval_receptor"], 6.628900e-04)
    assert equals(row["pval_adj_receptor"], 1)
    assert equals(row["pct_expressed_sender"], 0.894)
    assert equals(row["pct_expressed_receiver"], 0.148)
    assert equals(row["avg_ligand"], 16.61807231)
    assert equals(row["avg_receptor"], 1.3524264)
    assert equals(row["ligand_receptor_prod"], 2.247472e+01)
    assert equals(row["lfc_pval_ligand"], 4.275964e+00)
    assert equals(row["pval_adapted_ligand"], 6.661011140)
    assert equals(row["scaled_lfc_ligand"], 0.7277778)
    assert equals(row["scaled_pval_ligand"], 0.821974965)
    assert equals(row["scaled_lfc_pval_ligand"], 0.83031989)
    assert equals(row["scaled_pval_adapted_ligand"], 0.87065369)
    assert equals(row["activity"], 0.117003988)
    assert equals(row["rank"], 4)
    assert equals(row["activity_zscore"], 1.81234289)
    assert equals(row["scaled_activity"], 0.66009987)
    assert equals(row["lfc_pval_receptor"], 0.950935599)
    assert equals(row["pval_adapted_receptor"], 3.1785586)
    assert equals(row["scaled_lfc_receptor"], 0.78461538)
    assert equals(row["scaled_pval_receptor"], 0.8153846)
    assert equals(row["scaled_lfc_pval_receptor"], 0.83076923)
    assert equals(row["scaled_pval_adapted_receptor"], 0.84615385)
    assert equals(row["scaled_avg_exprs_ligand"], 1.001000000)
    assert equals(row["scaled_avg_exprs_receptor"], 1.0010000)
    assert equals(row["lfc_ligand_group"], 0.39227123)
    assert equals(row["pval_ligand_group"], 3.189997e-10)
    assert equals(row["lfc_pval_ligand_group"], 3.725089816)
    assert equals(row["pval_adapted_ligand_group"], 9.49620970)
    assert equals(row["scaled_lfc_ligand_group"], 0.4508197)
    assert equals(row["scaled_lfc_pval_ligand_group"], 0.6147541)
    assert equals(row["lfc_receptor_group"], 0.73450977)
    assert equals(row["pval_receptor_group"], 1.170731e-06)
    assert equals(row["lfc_pval_receptor_group"], 4.356776089)
    assert equals(row["pval_adapted_receptor_group"], 5.93154271)
    assert equals(row["scaled_lfc_receptor_group"], 0.8636364)
    assert equals(row["scaled_pval_receptor_group"], 0.80303030)
    assert equals(row["scaled_lfc_pval_receptor_group"], 0.8636364)
    assert equals(row["scaled_pval_adapted_receptor_group"], 0.83333333)

def test_ligand_target_signaling_path():
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    for filename in (
        ["ltf_matrix.pkl", "nichenet_human.pkl"]
    ):
        file_path = os.path.join(root_path, filename)
        if not os.path.exists(file_path):
            res = requests.get(f"https://zenodo.org/records/14944315/files/{filename}")
            with open(file_path, "wb") as file:
                file.write(res.content)
    with open(os.path.join(root_path, "ltf_matrix.pkl"), "rb") as file:
        ltf_matrix = pickle.loads(file.read())
    row_names = ltf_matrix["row_names"]
    col_names = ltf_matrix["col_names"]
    ltf_matrix = ltf_matrix["mat"]
    with open(os.path.join(root_path, "nichenet_human.pkl"), "rb") as file:
        model = pickle.loads(file.read())
    lr_sig = pd.DataFrame(model["lr_sig"]._mapping, columns=["from", "to", "weight"])
    gr = pd.DataFrame(model["gr"]._mapping, columns=["from", "to", "weight"])
    ligands_oi = ["TGFB2"]
    targets_oi = ["SERPINE1", "COL1A1"]
    tf_signaling, tf_regulatory = get_ligand_signaling_path(
        ltf_matrix,
        ligands_oi,
        targets_oi,
        row_names,
        col_names,
        lr_sig,
        gr,
        top_n_regulators=4,
        minmax_scaling=True
    )
    assert equals_iter(
        tf_signaling.sort_values(by="weight", ascending=False).head(10).to_numpy(),
        LTSP_top_10_tf_signaling
    )
    assert equals_iter(
        tf_regulatory.sort_values(by="weight", ascending=False).to_numpy(),
        LTSP_tf_regulatory
    )

# not deterministic
'''def test_target_prediction_evaluation_geneset():
    model = get_model_pickle("mouse")
    ann = get_anndata_file("annData3531889.h5")
    ann.var_names = ann.var["gene"]
    mouse_alias_info.alias_to_symbol(ann)
    predictor = model["predictor"]
    lr_network = model["lr_network"]
    lr_sig = model["lr_sig"]
    receiver = "CD8 T"
    sender_celltypes = ["CD4 T","Treg", "Mono", "NK", "B", "DC"]
    output = run_nichenet(
        ann,
        predictor,
        lr_network,
        lr_sig=lr_sig,
        receiver=receiver,
        sender_celltypes=sender_celltypes,
        condition_col="aggregate",
        condition_oi="LCMV",
        condition_ref="SS",
        expression_pct=0.05,
        targets_top_n=100
    )
    geneset_oi = output["geneset_oi"]
    expressed_genes_receiver = output["expressed_genes_receiver"]
    ligands_oi = output["best_upstream_ligands"]
    n = 2
    k = 3
    gene_predictions_top30_list = [
        assess_rf_class_probabilities(
            folds=k,
            geneset=geneset_oi,
            background_expressed_genes=expressed_genes_receiver,
            ligands_oi=ligands_oi,
            predictor=predictor
        ) for _ in range(n)
    ]
    assert equals_iter(
        gene_predictions_top30_list[0].sort_values(by="prediction", ascending=False).head(10).to_numpy(),
        TPEG_top_10_gene_predictions_0
    )'''

def test_ligand_activity_single_cell():
    model = get_model_pickle("human")
    predictor = model["predictor"]
    lr_network = model["lr_network"]
    lr_sig = model["lr_sig"]
    get_hnscc_file()
    exp_mat, exp_mat_rows, exp_mat_cols = read_csc_matrix(os.path.join(hnscc_path, "hnscc_expression.bin"))
    #cols = human_alias_info.alias_to_symbol(cols)
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
    row2id = dict(zip(exp_mat_rows, range(len(exp_mat_rows))))
    exp_mat = np.array(exp_mat.todense())
    expressed_genes_CAFs = get_exp(subset_matrix(exp_mat, [row2id[cell] for cell in CAF_cells]), exp_mat_cols)
    expressed_genes_malignant = get_exp(subset_matrix(exp_mat, [row2id[cell] for cell in malignant_cells]), exp_mat_cols)
    assert len(expressed_genes_CAFs) == 6706
    assert len(expressed_genes_malignant) == 6351
    ligands = lr_network.get_ligands()
    expressed_ligands = ligands.intersection(expressed_genes_CAFs)
    assert len(expressed_ligands) == 329
    receptors = lr_network.get_receptors()
    expressed_receptors = receptors.intersection(expressed_genes_malignant)
    assert len(expressed_receptors) == 198
    potential_ligands = {
        ligand
        for ligand, receptor in lr_network
        if ligand in expressed_ligands and receptor in expressed_receptors
    }
    assert len(potential_ligands) == 203
    background_expressed_genes = expressed_genes_malignant.intersection(predictor.row_names)
    assert len(background_expressed_genes) == 5891
    row2id = dict(zip(exp_mat_rows, range(len(exp_mat_rows))))
    col2id = dict(zip(exp_mat_cols, range(len(exp_mat_cols))))
    expression_scaled_col_ids, expression_scaled_cols = zip(*sorted((col2id[e], e) for e in background_expressed_genes))
    expression_scaled = scale_quantile(
        subset_matrix(
            exp_mat,
            rows=[row2id[e] for e in malignant_cells],
            cols=expression_scaled_col_ids
        )
    )
    assert equals_iter(
        expression_scaled[:4, :5].reshape((-1,)),
        LASC_scaled_expression_4_5
    )
    malignant_hn5_cells = [e[5] for e in sample_info if e[6] == "HN5" and e[1] == 0 and e[2] == 1][:10]
    ligand_activities = predictor.predict_single_cell_ligand_activities(
        malignant_hn5_cells,
        expression_scaled,
        malignant_cells,
        expression_scaled_cols,
        potential_ligands
    )
    ligand_activities = ligand_activities_df(ligand_activities)
    ligand_activities.sort_index(inplace=True)
    ligand_activities.reset_index(inplace=True)
    ligand_activities.rename(columns={"level_0": "cell", "level_1": "ligand"}, inplace=True)
    assert len(ligand_activities) == 2030
    assert equals_iter(
        ligand_activities[["cell", "ligand", "auroc", "aupr", "pearson"]].sort_values(by="aupr", ascending=False).head(10).to_numpy(),
        LASC_top_10_ligand_activities
    )
    row2id = dict(zip(malignant_cells, range(len(malignant_cells))))
    cell_scores = pd.DataFrame({
        "cell": malignant_hn5_cells,
        "score": subset_matrix(
            expression_scaled,
            rows=[row2id[e] for e in malignant_hn5_cells],
            cols=[expression_scaled_cols.index("TGFBI")]
        ).reshape((-1,))
    })
    assert len(cell_scores) == 10
    assert equals_iter(
        cell_scores.sort_values(by="score", ascending=False).to_numpy(),
        LASC_cell_scores
    )
    normalized_ligand_activities = normalize_single_cell_ligand_activities(ligand_activities)
    normalized_ligand_activities.reset_index(inplace=True)
    normalized_ligand_activities.columns.name = None
    assert normalized_ligand_activities.shape[1] == 204
    assert equals(
        normalized_ligand_activities[normalized_ligand_activities["cell"] == "HNSCC5_p3_HNSCC5_P3_A04"]["A2M"].iloc[0],
        0.2382057
    )
    assert equals(
        normalized_ligand_activities[normalized_ligand_activities["cell"] == "HNSCC5_p3_HNSCC5_P3_C11"]["ADAM10"].iloc[0],
        -0.842991150
    )
    assert equals(
        normalized_ligand_activities[normalized_ligand_activities["cell"] == "HNSCC5_p3_HNSCC5_P3_D10"]["ADAM12"].iloc[0],
        -0.87919944
    )
    assert equals(
        normalized_ligand_activities[normalized_ligand_activities["cell"] == "HNSCC5_p3_HNSCC5_P3_E03"]["ADAM15"].iloc[0],
        -0.531216348
    )
    assert equals(
        normalized_ligand_activities[normalized_ligand_activities["cell"] == "HNSCC5_p3_HNSCC5_P3_F07"]["ADAM17"].iloc[0],
        0.5201329
    )
    assert equals(
        normalized_ligand_activities[normalized_ligand_activities["cell"] == "HNSCC5_p3_HNSCC5_P3_G07"]["ADAM9"].iloc[0],
        -0.20499353
    )
    assert equals(
        normalized_ligand_activities[normalized_ligand_activities["cell"] == "HNSCC5_p3_HNSCC5_P3_H01"]["ADM"].iloc[0],
        1.2439335
    )
    assert equals(
        normalized_ligand_activities[normalized_ligand_activities["cell"] == "HNSCC5_p9_HNSCC5_P9_B10"]["ANG"].iloc[0],
        0.4335136
    )
    assert equals(
        normalized_ligand_activities[normalized_ligand_activities["cell"] == "HNSCC5_p9_HNSCC5_P9_D03"]["ANGPTL1"].iloc[0],
        -0.57912971
    )
    assert equals(
        normalized_ligand_activities[normalized_ligand_activities["cell"] == "HNSCC5_p9_HNSCC5_P9_D08"]["ANGPTL2"].iloc[0],
        1.0040180
    )
    combined = normalized_ligand_activities.merge(cell_scores)
    res = np.array(combined["score"])
    preds = combined.drop(columns=["score", "cell"])
    output_correlation_analysis = pd.DataFrame({
        "ligand": preds.columns,
        "pearson": (pearsonr(preds.iloc[:, i], res).statistic for i in range(preds.shape[1]))
    })
    assert equals_iter(
        output_correlation_analysis.sort_values(by="pearson", ascending=False).to_numpy(),
        LASC_top_10_output_correlation
    )

def test_model_construction():
    get_network_files()
    gr_network = pd.DataFrame(read_csv_cols(os.path.join(network_path, "gr_human.csv")))
    lr_network = pd.DataFrame(read_csv_cols(os.path.join(network_path, "lr_network_human.csv")))
    sig_network = pd.DataFrame(read_csv_cols(os.path.join(network_path, "lr_sig_human.csv")))
    source_weights = dict(zip(set(chain(gr_network["source"], lr_network["source"], sig_network["source"])), repeat(1)))
    weighted_networks = construct_weighted_networks(
        lr_network,
        sig_network,
        gr_network,
        source_weights
    )
    weighted_networks["lr_sig"] = apply_hub_correction(weighted_networks["lr_sig"], hub=0.115)
    weighted_networks["gr"] = apply_hub_correction(weighted_networks["gr"], hub=0.0803)
    assert len(weighted_networks["lr_sig"]) == 3923501
    assert equals_dict(
        df2dict(weighted_networks["lr_sig"], ("from", "to"), "weight"),
        MC_top_10_lr_sig_0
    )
    assert len(weighted_networks["gr"]) == 4640268
    assert equals_dict(
        df2dict(weighted_networks["gr"], ("from", "to"), "weight"),
        MC_top_10_gr_0
    )
    ligands = [["TNF"], ["TNF", "IL6"]]
    row_names, col_names, mat = construct_ligand_target_matrix(
        weighted_networks,
        lr_network,
        ligands,
        damping_factor=0.789,
        ltf_cutoff=0.926
    )
    df = pd.DataFrame(mat, index=row_names, columns=col_names)
    assert len(df) == 33354
    assert equals_iter(
        df.head(10).to_numpy(),
        MC_top_10_ligand_target_matrix_PPR_0
    )
    row_names, col_names, mat = construct_ligand_target_matrix(
        weighted_networks,
        lr_network,
        ligands,
        damping_factor=0.789,
        ltf_cutoff=0.926,
        algorithm="SPL"
    )
    df = pd.DataFrame(mat, index=row_names, columns=col_names)
    assert len(df) == 33354
    assert equals_iter(
        df.head(10).to_numpy(),
        MC_top_10_ligand_target_matrix_SPL_0,
        err_bound=0.02 # correct but bigger difference between nichenetr and nichenetpy
    )
    row_names, col_names, mat = construct_ligand_target_matrix(
        weighted_networks,
        lr_network,
        ligands,
        damping_factor=0.789,
        ltf_cutoff=0.926,
        algorithm="direct"
    )
    df = pd.DataFrame(mat, index=row_names, columns=col_names)
    assert len(df) == 33354
    assert equals_iter(
        df.head(10).to_numpy(),
        MC_top_10_ligand_target_matrix_direct_0,
        err_bound=0.12 # TODO: acceptable?
    )
    optimized_source_weights = tuple(zip(*read_csv_rows(os.path.join(network_path, "optimized_source_weights.csv"))[1]))
    optimized_source_weights = dict(zip(optimized_source_weights[0], [float(e) for e in optimized_source_weights[1]]))
    weighted_networks = construct_weighted_networks(
        lr_network,
        sig_network,
        gr_network,
        optimized_source_weights
    )
    weighted_networks["lr_sig"] = apply_hub_correction(weighted_networks["lr_sig"], hub=0.115)
    weighted_networks["gr"] = apply_hub_correction(weighted_networks["gr"], hub=0.0803)
    assert len(weighted_networks["lr_sig"]) == 3923501
    assert equals_dict(
        df2dict(weighted_networks["lr_sig"], ("from", "to"), "weight"),
        MC_top_10_lr_sig_1
    )
    assert len(weighted_networks["gr"]) == 4640268
    assert equals_dict(
        df2dict(weighted_networks["gr"], ("from", "to"), "weight"),
        MC_top_10_gr_1
    )
    ligands = [["TNF"]]
    row_names, col_names, mat = construct_ligand_target_matrix(
        weighted_networks,
        lr_network,
        ligands,
        damping_factor=0.789,
        ltf_cutoff=0.926
    )
    df = pd.DataFrame(mat, index=row_names, columns=col_names)
    assert len(df) == 33354
    assert equals_iter(
        df.head(10).to_numpy().reshape((-1,)),
        MC_top_10_ligand_target_matrix_PPR_1
    )
    row_names, col_names, mat = construct_ligand_target_matrix(
        weighted_networks,
        lr_network,
        ligands,
        damping_factor=0.789,
        ltf_cutoff=0.926,
        algorithm="SPL"
    )
    df = pd.DataFrame(mat, index=row_names, columns=col_names)
    assert len(df) == 33354
    assert equals_iter(
        df.head(10).to_numpy().reshape((-1,)),
        MC_top_10_ligand_target_matrix_SPL_1
    )
    row_names, col_names, mat = construct_ligand_target_matrix(
        weighted_networks,
        lr_network,
        ligands,
        damping_factor=0.789,
        ltf_cutoff=0.926,
        algorithm="direct"
    )
    df = pd.DataFrame(mat, index=row_names, columns=col_names)
    assert len(df) == 33354
    assert equals_iter(
        df.head(10).to_numpy().reshape((-1,)),
        MC_top_10_ligand_target_matrix_direct_1
    )
    annotations = pd.DataFrame(read_csv_cols(os.path.join(network_path, "annotation_data_sources.csv")))
    data_sources_to_keep = set(annotations[[e in ["literature", "comprehensive_db", "ChIP"] for e in annotations["type_db"]]]["source"])
    new_source_weights = dict((key, val) for key, val in source_weights.items() if key in data_sources_to_keep)
    new_lr_network = lr_network[[e in data_sources_to_keep for e in lr_network["source"]]]
    new_sig_network = sig_network[[e in data_sources_to_keep for e in sig_network["source"]]]
    new_gr_network = gr_network[[e in data_sources_to_keep for e in gr_network["source"]]]
    weighted_networks = construct_weighted_networks(
        new_lr_network,
        new_sig_network,
        new_gr_network,
        new_source_weights
    )
    weighted_networks["lr_sig"] = apply_hub_correction(weighted_networks["lr_sig"], hub=0.115)
    weighted_networks["gr"] = apply_hub_correction(weighted_networks["gr"], hub=0.0803)
    assert len(weighted_networks["lr_sig"]) == 2765375
    assert equals_dict(
        df2dict(weighted_networks["lr_sig"], ("from", "to"), "weight"),
        MC_top_10_lr_sig_2
    )
    assert len(weighted_networks["gr"]) == 3645365
    assert equals_dict(
        df2dict(weighted_networks["gr"], ("from", "to"), "weight"),
        MC_top_10_gr_2
    )
    ligands = [["TNF"]]
    row_names, col_names, mat = construct_ligand_target_matrix(
        weighted_networks,
        new_lr_network,
        ligands,
        damping_factor=0.789,
        ltf_cutoff=0.926
    )
    df = pd.DataFrame(mat, index=row_names, columns=col_names)
    assert len(df) == 28821
    assert equals_iter(
        df.head(10).to_numpy().reshape((-1)),
        MC_top_10_ligand_target_matrix_PPR_2
    )
    row_names, col_names, mat = construct_ligand_target_matrix(
        weighted_networks,
        new_lr_network,
        ligands,
        damping_factor=0.789,
        ltf_cutoff=0.926,
        algorithm="SPL"
    )
    df = pd.DataFrame(mat, index=row_names, columns=col_names)
    assert len(df) == 28821
    assert equals_iter(
        df.head(10).to_numpy().reshape((-1)),
        MC_top_10_ligand_target_matrix_SPL_2
    )
    row_names, col_names, mat = construct_ligand_target_matrix(
        weighted_networks,
        new_lr_network,
        ligands,
        damping_factor=0.789,
        ltf_cutoff=0.926,
        algorithm="direct"
    )
    df = pd.DataFrame(mat, index=row_names, columns=col_names)
    assert len(df) == 28821
    assert equals_iter(
        df.head(10).to_numpy().reshape((-1)),
        MC_top_10_ligand_target_matrix_direct_2
    )
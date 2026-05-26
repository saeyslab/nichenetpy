'''
Here we test the LigandActivityPredictor class. 
'''

from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.utils import (
    read_matrix_from_csv,
    read_list_from_csv,
    read_csv_cols,
    ligand_activities_df
)
from nichenetpy.network import (
    LigandReceptorNetwork
)

import os
import requests
import pandas as pd

from common import (
    get_model_pickle,
    equals_iter
)


FIBRO_top_10_ligand_activities = [
    ("TGFB1", 0.7053512, 0.4250502, 0.24693812, 0.35939569),
    ("TGFB2", 0.6135986, 0.3121863, 0.13407426, 0.21476737),
    ("BMP2", 0.6327001, 0.3031900, 0.12507798, 0.19386352),
    ("GDF11", 0.6214612, 0.3028873, 0.12477526, 0.17826664),
    ("MSTN", 0.6172203, 0.2990393, 0.12092720, 0.18910862),
    ("IGF2", 0.6196072, 0.2962454, 0.11813333, 0.17182906),
    ("FGF1", 0.6210493, 0.2925995, 0.11448738, 0.16029547),
    ("NOG", 0.6375389, 0.2918207, 0.11370866, 0.18599441),
    ("BDNF", 0.6195013, 0.2897681, 0.11165606, 0.17457172),
    ("FGF2", 0.6115726, 0.2887976, 0.11068553, 0.17116988)
]

H1HESC_top_10_ligand_activities = [
    ("NTF4", 0.6584075, 0.013344221, 8.834207e-03, 9.559437e-02),
    ("LRP1B", 0.4590744, 0.008082794, 3.572781e-03, 2.581370e-02),
    ("DCHS1", 0.4773549, 0.007131004, 2.620991e-03, 1.509082e-02),
    ("GPI", 0.4722401, 0.007083311, 2.573297e-03, 3.552039e-02),
    ("IGFBPL1", 0.5249448, 0.006485894, 1.975880e-03, 1.858596e-02),
    ("LAMB1", 0.4995777, 0.006341005, 1.830991e-03, 1.374030e-02),
    ("LRIG2", 0.5435611, 0.006134705, 1.624691e-03, 1.411201e-02),
    ("SLITRK1", 0.5742214, 0.005667876, 1.157862e-03, 1.547085e-02),
    ("BDNF", 0.4823916, 0.005578658, 1.068644e-03, 4.616971e-03),
    ("BMP7", 0.5099153, 0.005508112, 9.980982e-04, 2.545053e-02)
]

HSC_top_10_ligand_activities = [
    ("TGFB1", 0.7266981, 0.3733326, 0.24413483, 0.36508836),
    ("TGFB2", 0.6318449, 0.2867826, 0.15758489, 0.25730138),
    ("IL1B", 0.6424481, 0.2655145, 0.13631676, 0.22967041),
    ("FGF1", 0.6424418, 0.2583335, 0.12913576, 0.18337982),
    ("GDF11", 0.6385873, 0.2511260, 0.12192824, 0.17571097),
    ("BMP2", 0.6496446, 0.2480818, 0.11888407, 0.19673414),
    ("GDF5", 0.6359335, 0.2473969, 0.11819915, 0.16990932),
    ("FGF7", 0.6335544, 0.2467590, 0.11756128, 0.17172997),
    ("IGF2", 0.6345342, 0.2460899, 0.11689218, 0.16745465),
    ("NOG", 0.6544142, 0.2420827, 0.11288496, 0.18578765)
]

PC_PANC1_top_10_ligand_activities = [
    ("TGFB1", 0.7796829, 0.14333852, 0.11987014, 0.25985304),
    ("TGFB2", 0.6520165, 0.07585426, 0.05238588, 0.15502246),
    ("MMP14", 0.6540166, 0.06487743, 0.04140905, 0.11726678),
    ("WNT7A", 0.6535455, 0.06268234, 0.03921396, 0.09440047),
    ("COL1A1", 0.6394335, 0.06226818, 0.03879980, 0.09820971),
    ("COL4A1", 0.6486240, 0.06138303, 0.03791465, 0.08881984),
    ("COL5A1", 0.6439879, 0.06032205, 0.03685367, 0.08727523),
    ("FGF2", 0.6459635, 0.06021913, 0.03675075, 0.11575300),
    ("ZDHHC5", 0.6856707, 0.05898545, 0.03551708, 0.10398222),
    ("SLC6A8", 0.6532393, 0.05890604, 0.03543766, 0.10442897)
]

PROST_FIB_top_10_ligand_activities = [
    ("TGFB1", 0.7825570, 0.2831018, 0.21645682, 0.35408794),
    ("TGFB2", 0.6632513, 0.2017243, 0.13507932, 0.25325038),
    ("GDF11", 0.6672949, 0.1680240, 0.10137902, 0.18956518),
    ("BMP2", 0.6793312, 0.1526366, 0.08599162, 0.17486507),
    ("GDF5", 0.6676549, 0.1526052, 0.08596022, 0.15280073),
    ("FGF7", 0.6631136, 0.1524573, 0.08581239, 0.15842560),
    ("FGF1", 0.6714325, 0.1514308, 0.08478590, 0.14998648),
    ("PDGFD", 0.6409933, 0.1467668, 0.08012184, 0.17795645),
    ("BMP4", 0.6648839, 0.1455312, 0.07888625, 0.14922262),
    ("IGF2", 0.6639373, 0.1430035, 0.07635853, 0.14689385)
]

root = os.path.dirname(__file__)
err_bound = 1e-12
data_path = os.path.join(root, "data/TGFB1")
if not os.path.exists(data_path):
    os.makedirs(data_path)
for filename in (
    "TGFB1_24_H_FIBRO_GSE277437.csv",
    "TGFB1_24_H_H1HESC_GSE247021.csv",
    "TGFB1_48_H_HSC_GSE151251.csv",
    "TGFB1_72_H_PC_PANC1_GSE88757.csv",
    "TGFB1_72_H_PROST_FIB_GSE205378.csv"
):
    file_path = os.path.join(data_path, filename)
    if not os.path.exists(file_path):
        res = requests.get(f"https://zenodo.org/records/16097675/files/{filename}")
        with open(file_path, "wb") as file:
            file.write(res.content)

def process_expected_output(cols:dict[str, list[str]]) -> dict[str, dict[str, float]]:
    return dict(
        zip(
            cols["test_ligand"],
            zip(
                [float(e) for e in cols["auroc"]],
                [float(e) for e in cols["aupr"]],
                [float(e) for e in cols["aupr_corrected"]],
                [float(e) for e in cols["pearson"]],
                [float(e) for e in cols["rank"]]
            )
        )
    )

def template_predict_ligand_activities(test_id):
    predictor = LigandActivityPredictor(*read_matrix_from_csv(os.path.join(root, f"data/predictor/{test_id}/ligand_target_matrix.csv")))
    geneset = read_list_from_csv(os.path.join(root, f"data/predictor/{test_id}/geneset_oi.csv"))
    background_expressed_genes = read_list_from_csv(os.path.join(root, f"data/predictor/{test_id}/background_expressed_genes.csv"))
    potential_ligands = read_list_from_csv(os.path.join(root, f"data/predictor/{test_id}/potential_ligands.csv"))
    expected_output = process_expected_output(read_csv_cols(os.path.join(root, f"data/predictor/{test_id}/output.csv")))
    ligand_activities = predictor.predict_ligand_activities(
        geneset,
        background_expressed_genes,
        potential_ligands
    )
    assert len(ligand_activities) == len(expected_output), (
        f"expected {len(expected_output)} ligands, got {len(ligand_activities)}")
    for ligand, values in expected_output.items():
        assert ligand in ligand_activities, f"the expected ligand {ligand} is missing in the output"
        assert abs(ligand_activities[ligand]["aupr"] - values[1]) < err_bound, (
            f"expected aupr for {ligand} to be {values[1]}, got {ligand_activities[ligand]["aupr"]}")
        assert abs(ligand_activities[ligand]["aupr_corrected"] - values[2]) < err_bound, (
            f"expected aupr_corrected for {ligand} to be {values[2]}, got {ligand_activities[ligand]["aupr_corrected"]}")

def test_predict_ligand_activities_0():
    template_predict_ligand_activities(0)

def test_prediction_TGFB1():
    model = get_model_pickle("human")
    predictor : LigandActivityPredictor = model["predictor"]
    lr_network : LigandReceptorNetwork = model["lr_network"]
    ligand_activities = dict()
    for filename in os.listdir(data_path):
        data = pd.DataFrame(read_csv_cols(os.path.join(data_path, filename)))
        act = predictor.predict_ligand_activities(
            geneset=data[data["is_de"] == "TRUE"][""],
            background_expressed_genes=data[""],
            potential_ligands=lr_network.get_ligands().intersection(data[""])
        )
        ligand_activities[filename] = ligand_activities_df(act)
    assert equals_iter(
        ligand_activities["TGFB1_24_H_FIBRO_GSE277437.csv"][
            ["auroc", "aupr", "aupr_corrected", "pearson"]
        ].sort_values(by="aupr", ascending=False).reset_index().head(10).to_numpy(),
        FIBRO_top_10_ligand_activities
    )

def test_prediction_H1HESC():
    model = get_model_pickle("human")
    predictor : LigandActivityPredictor = model["predictor"]
    lr_network : LigandReceptorNetwork = model["lr_network"]
    ligand_activities = dict()
    for filename in os.listdir(data_path):
        data = pd.DataFrame(read_csv_cols(os.path.join(data_path, filename)))
        act = predictor.predict_ligand_activities(
            geneset=data[data["is_de"] == "TRUE"][""],
            background_expressed_genes=data[""],
            potential_ligands=lr_network.get_ligands().intersection(data[""])
        )
        ligand_activities[filename] = ligand_activities_df(act)
    assert equals_iter(
        ligand_activities["TGFB1_24_H_H1HESC_GSE247021.csv"][
            ["auroc", "aupr", "aupr_corrected", "pearson"]
        ].sort_values(by="aupr", ascending=False).reset_index().head(10).to_numpy(),
        H1HESC_top_10_ligand_activities
    )

def test_prediction_HSC():
    model = get_model_pickle("human")
    predictor : LigandActivityPredictor = model["predictor"]
    lr_network : LigandReceptorNetwork = model["lr_network"]
    ligand_activities = dict()
    for filename in os.listdir(data_path):
        data = pd.DataFrame(read_csv_cols(os.path.join(data_path, filename)))
        act = predictor.predict_ligand_activities(
            geneset=data[data["is_de"] == "TRUE"][""],
            background_expressed_genes=data[""],
            potential_ligands=lr_network.get_ligands().intersection(data[""])
        )
        ligand_activities[filename] = ligand_activities_df(act)
    assert equals_iter(
        ligand_activities["TGFB1_48_H_HSC_GSE151251.csv"][
            ["auroc", "aupr", "aupr_corrected", "pearson"]
        ].sort_values(by="aupr", ascending=False).reset_index().head(10).to_numpy(),
        HSC_top_10_ligand_activities
    )

def test_prediction_PC_PANC1():
    model = get_model_pickle("human")
    predictor : LigandActivityPredictor = model["predictor"]
    lr_network : LigandReceptorNetwork = model["lr_network"]
    ligand_activities = dict()
    for filename in os.listdir(data_path):
        data = pd.DataFrame(read_csv_cols(os.path.join(data_path, filename)))
        act = predictor.predict_ligand_activities(
            geneset=data[data["is_de"] == "TRUE"][""],
            background_expressed_genes=data[""],
            potential_ligands=lr_network.get_ligands().intersection(data[""])
        )
        ligand_activities[filename] = ligand_activities_df(act)
    assert equals_iter(
        ligand_activities["TGFB1_72_H_PC_PANC1_GSE88757.csv"][
            ["auroc", "aupr", "aupr_corrected", "pearson"]
        ].sort_values(by="aupr", ascending=False).reset_index().head(10).to_numpy(),
        PC_PANC1_top_10_ligand_activities
    )

def test_prediction_PROST_FIB():
    model = get_model_pickle("human")
    predictor : LigandActivityPredictor = model["predictor"]
    lr_network : LigandReceptorNetwork = model["lr_network"]
    ligand_activities = dict()
    for filename in os.listdir(data_path):
        data = pd.DataFrame(read_csv_cols(os.path.join(data_path, filename)))
        act = predictor.predict_ligand_activities(
            geneset=data[data["is_de"] == "TRUE"][""],
            background_expressed_genes=data[""],
            potential_ligands=lr_network.get_ligands().intersection(data[""])
        )
        ligand_activities[filename] = ligand_activities_df(act)
    assert equals_iter(
        ligand_activities["TGFB1_72_H_PROST_FIB_GSE205378.csv"][
            ["auroc", "aupr", "aupr_corrected", "pearson"]
        ].sort_values(by="aupr", ascending=False).reset_index().head(10).to_numpy(),
        PROST_FIB_top_10_ligand_activities
    )
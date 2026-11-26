# type: ignore

"""
Here we test some errors.
"""

import os

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from common import get_anndata_file, get_hnscc_file, get_model_pickle, hnscc_path

from nichenetpy.ann_utils import subset_ann
from nichenetpy.extraction import average_expression, get_expressed_genes
from nichenetpy.io import read_csc_matrix
from nichenetpy.metrics import group_metrics
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.prioritization import get_avg_exp


def test_group_metrics_error():
    ann = get_anndata_file("ann_hnscc.h5")
    print(ann.obs.columns)
    with pytest.raises(RuntimeError):
        ann_receiver = subset_ann(ann, val="T.cell", val_col="celltype", layers=["data"])
        group_metrics(ann_receiver, groupby="classified..as.cancer.cell", layer="data", min_pct=0.05, min_abs_lfc=0.25)


def test_subset_ann_error():
    ann = get_anndata_file("annData3531889.h5")
    gene_that_doesnt_exist = "Eugene"
    celltype_that_doesnt_exist = "celluloid heroes"
    missing_col = "MIA"
    with pytest.raises(ValueError, match=rf"{gene_that_doesnt_exist}.*not present"):
        subset_ann(ann, genes=[gene_that_doesnt_exist])
    with pytest.raises(ValueError, match=rf"{celltype_that_doesnt_exist}.*not present"):
        subset_ann(ann, val=celltype_that_doesnt_exist, val_col="celltype")
    with pytest.raises(ValueError, match=rf"no column.*{missing_col}"):
        subset_ann(ann, val="anything", val_col=missing_col)


def test_get_expressed_genes_error():
    ann = get_anndata_file("annData3531889.h5")
    celltype_that_doesnt_exist = "celluloid heroes"
    missing_col = "MIA"
    with pytest.raises(ValueError, match=rf"no cells of types \['{celltype_that_doesnt_exist}'\]"):
        get_expressed_genes(celltype_that_doesnt_exist, ann, celltype_col="celltype")
    with pytest.raises(ValueError, match=rf"no column '{missing_col}'"):
        get_expressed_genes("CD8 T", ann, celltype_col=missing_col)


def test_average_expression_error():
    ann = get_anndata_file("annData3531889.h5")
    missing_col = "MIA"
    with pytest.raises(ValueError, match=rf"no column '{missing_col}'"):
        average_expression(ann, groupby=missing_col)


def test_get_avg_exp_condition_error():
    # condition_oi and condition_col are given together (as in NicheNetR)
    obs = pd.DataFrame({"celltype": ["A", "B"], "condition": ["x", "y"]}, index=["c0", "c1"])
    ann = AnnData(X=np.ones((2, 1)), obs=obs)
    ann.layers["counts"] = ann.X
    for kwargs in ({"condition_oi": "x"}, {"condition_col": "condition"}):
        with pytest.raises(ValueError, match="together"):
            get_avg_exp(ann, "celltype", **kwargs)


def test_evaluate_target_prediction_error():
    ligand_that_doesnt_exist = "Triboulet"
    model = get_model_pickle("mouse")
    predictor = model["predictor"]
    with pytest.raises(ValueError, match=rf"{ligand_that_doesnt_exist} not in ligand_target_matrix"):
        predictor.evaluate_target_prediction(ligand=ligand_that_doesnt_exist, response={"a": 1, "b": 1, "c": 0})


def test_predict_single_cell_ligand_activities_error():
    cell_that_doesnt_exist = "Celcius"
    ligand_that_doesnt_exist = "Yolof"
    model = get_model_pickle("mouse")
    predictor = model["predictor"]
    get_hnscc_file()
    exp_mat, exp_mat_rows, exp_mat_cols = read_csc_matrix(os.path.join(hnscc_path, "hnscc_expression.bin"))
    exp_mat = exp_mat.toarray()
    with pytest.raises(ValueError, match=rf"'{cell_that_doesnt_exist}' is not in expression_scaled_rows"):
        predictor.predict_single_cell_ligand_activities(
            [cell_that_doesnt_exist], exp_mat, exp_mat_rows, exp_mat_cols, predictor.get_ligands()
        )
    with pytest.raises(ValueError, match=rf"{ligand_that_doesnt_exist} not in ligand_target_matrix"):
        predictor.predict_single_cell_ligand_activities(
            exp_mat_rows[:3], exp_mat, exp_mat_rows, exp_mat_cols, [ligand_that_doesnt_exist]
        )


def _small_predictor():
    return LigandActivityPredictor(
        np.asfortranarray([[0.1, 0.5], [0.4, 0.2], [0.3, 0.9]]), ["g0", "g1", "g2"], ["L1", "L2"]
    )


def test_predict_single_cell_ligand_activities_names_error():
    # two cells and three genes; the names must fit the shape and be unique
    expression = np.array([[0.5, -1.0, 2.0], [1.5, 0.2, -0.3]])
    predictor = _small_predictor()
    for rows, cols, message in (
        (["c0", "c1"], ["g0", "g1"], "shape"),
        (["c0", "c1", "c2"], ["g0", "g1", "g2"], "shape"),
        (["c0", "c0"], ["g0", "g1", "g2"], "duplicates"),
        (["c0", "c1"], ["g0", "g1", "g1"], "duplicates"),
    ):
        with pytest.raises(ValueError, match=message):
            predictor.predict_single_cell_ligand_activities(rows, expression, rows, cols, ["L1", "L2"])


def test_get_weighted_ligand_target_links_error():
    with pytest.raises(ValueError, match="at least 0"):
        _small_predictor().get_weighted_ligand_target_links("L1", ["g0", "g1"], n=-1)


def test_average_expression_keys_error():
    obs = pd.DataFrame({"celltype": ["A", "A", "B"]}, index=["c0", "c1", "c2"])
    mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ann = AnnData(X=mat, obs=obs, var=pd.DataFrame(index=["g0", "g1"]))
    ann.layers["counts"] = mat
    # keys that aren't values of the column are left out, none of them is an error
    result = average_expression(ann, "celltype", keys=["A", "Z"], layer="counts")
    assert result.columns.tolist() == ["A"]
    np.testing.assert_allclose(result["A"], [2.0, 3.0])
    with pytest.raises(ValueError, match="none of the keys"):
        average_expression(ann, "celltype", keys=["Z"], layer="counts")
    with pytest.raises(ValueError, match="no column 'MIA'"):
        average_expression(ann, "MIA", keys=["A"], layer="counts")

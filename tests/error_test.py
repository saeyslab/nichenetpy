from nichenetpy.metrics import group_metrics
from nichenetpy.ann_utils import subset_ann
from nichenetpy.io import read_csc_matrix
from nichenetpy.extraction import (
    get_expressed_genes,
    average_expression
)

from common import (
    get_anndata_file,
    get_model_pickle,
    get_hnscc_file,
    hnscc_path
)

import pytest
import os


def test_group_metrics_error():
    ann = get_anndata_file("ann_hnscc.h5")
    print(ann.obs.columns)
    with pytest.raises(RuntimeError):
        ann_receiver = subset_ann(
            ann,
            val="T.cell",
            val_col="celltype",
            layers=["data"]
        )
        group_metrics(
            ann_receiver,
            groupby="classified..as.cancer.cell",
            layer="data",
            min_pct=0.05,
            min_abs_lfc=0.25
        )

def test_subset_ann_error():
    ann = get_anndata_file("annData3531889.h5")
    gene_that_doesnt_exist = "Eugene"
    celltype_that_doesnt_exist = "celluloid heroes"
    missing_col = "MIA"
    with pytest.raises(
        ValueError,
        match=rf"{gene_that_doesnt_exist}.*not present"
    ):
        subset_ann(
            ann,
            genes=[gene_that_doesnt_exist]
        )
    with pytest.raises(
        ValueError,
        match=rf"{celltype_that_doesnt_exist}.*not present"
    ):
        subset_ann(
            ann,
            val=celltype_that_doesnt_exist,
            val_col="celltype"
        )
    with pytest.raises(
        ValueError,
        match=rf"no column.*{missing_col}"
    ):
        subset_ann(
            ann,
            val="anything",
            val_col=missing_col
        )

def test_get_expressed_genes_error():
    ann = get_anndata_file("annData3531889.h5")
    celltype_that_doesnt_exist = "celluloid heroes"
    missing_col = "MIA"
    with pytest.raises(
        ValueError,
        match=rf"no cells of types \['{celltype_that_doesnt_exist}'\]"
    ):
        get_expressed_genes(
            celltype_that_doesnt_exist,
            ann,
            celltype_col="celltype"
        )
    with pytest.raises(
        ValueError,
        match=rf"no column '{missing_col}'"
    ):
        get_expressed_genes(
            "CD8 T",
            ann,
            celltype_col=missing_col
        )

def test_average_expression_error():
    ann = get_anndata_file("annData3531889.h5")
    missing_col = "MIA"
    with pytest.raises(
        ValueError,
        match=rf"no column '{missing_col}'"
    ):
        average_expression(
            ann,
            groupby=missing_col
        )

def test_evaluate_target_prediction_error():
    ligand_that_doesnt_exist = "Triboulet"
    model = get_model_pickle("mouse")
    predictor = model["predictor"]
    with pytest.raises(
        ValueError,
        match=rf"{ligand_that_doesnt_exist} not in ligand_target_matrix"
    ):
        predictor.evaluate_target_prediction(
            ligand=ligand_that_doesnt_exist,
            response={
                "a": 1,
                "b": 1,
                "c": 0
            }
        )

def test_predict_single_cell_ligand_activities_error():
    cell_that_doesnt_exist = "Celcius"
    ligand_that_doesnt_exist = "Yolof"
    model = get_model_pickle("mouse")
    predictor = model["predictor"]
    get_hnscc_file()
    exp_mat, exp_mat_rows, exp_mat_cols = read_csc_matrix(os.path.join(hnscc_path, "hnscc_expression.bin"))
    exp_mat = exp_mat.toarray()
    with pytest.raises(
        ValueError,
        match=rf"{cell_that_doesnt_exist} not in ligand_target_matrix"
    ):
        predictor.predict_single_cell_ligand_activities(
            [cell_that_doesnt_exist],
            exp_mat,
            exp_mat_rows,
            exp_mat_cols,
            predictor.get_ligands()
        )
    with pytest.raises(
        ValueError,
        match=rf"{ligand_that_doesnt_exist} not in ligand_target_matrix"
    ):
        predictor.predict_single_cell_ligand_activities(
            exp_mat_rows[:3],
            exp_mat,
            exp_mat_rows,
            exp_mat_cols,
            [ligand_that_doesnt_exist]
        )
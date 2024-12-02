import pytest
import os
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.utils import read_matrix_from_csv, read_list_from_csv, read_csv_cols

root = os.path.dirname(__file__)
err_bound = 1e-12

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
    predictor = LigandActivityPredictor(*read_matrix_from_csv(os.path.join(root, f"./data/{test_id}/ligand_target_matrix.csv")))
    geneset = read_list_from_csv(os.path.join(root, f"./data/{test_id}/geneset_oi.csv"))
    background_expressed_genes = read_list_from_csv(os.path.join(root, f"./data/{test_id}/background_expressed_genes.csv"))
    potential_ligands = read_list_from_csv(os.path.join(root, f"./data/{test_id}/potential_ligands.csv"))
    expected_output = process_expected_output(read_csv_cols(os.path.join(root, f"./data/{test_id}/output.csv")))
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
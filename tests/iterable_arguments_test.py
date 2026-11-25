# type: ignore

"""
Arguments annotated as an Iterable:
- an Iterable that is iterated more than once also works when it's an iterator (e.g. a generator), with the same result as
  a list;
- an Iterable of str (gene symbols, metric names, labels, ...) is not a single str: a str raises a TypeError (unless the
  annotation also allows a str).
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from nichenetpy.ann_utils import subset_ann
from nichenetpy.evaluation import evaluate_single_importances_ligand_prediction, get_single_ligand_importances
from nichenetpy.extraction import average_expression, get_weighted_ligand_receptor_links
from nichenetpy.gene_symbol import GeneAliasInfo, GeneInfo
from nichenetpy.metrics import calculate_evaluation_metrics, group_metrics
from nichenetpy.model_construction import construct_ligand_target_matrix, construct_ligand_tf_matrix
from nichenetpy.parameter_optimization import compute_evaluation_scores, construct_and_evaluate
from nichenetpy.prediction import LigandActivityPredictor, assess_rf_class_probabilities
from nichenetpy.prioritization import calculate_de, get_avg_exp
from nichenetpy.utils import ncycle
from nichenetpy.visualization import create_radar_plot
from nichenetpy.wrappers import create_ligand_receptor_links_prioritization_circos_plot


def _ann():
    obs = pd.DataFrame(
        {"celltype": ["A", "A", "B", "B"], "condition": ["c", "c", "c", "c"]}, index=[f"cell{i}" for i in range(4)]
    )
    mat = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [3.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    ann = AnnData(X=mat, obs=obs, var=pd.DataFrame(index=["g0", "g1", "g2"]))
    ann.layers["data"] = mat
    ann.layers["counts"] = mat
    return ann


def _predictor():
    return LigandActivityPredictor(
        np.asfortranarray([[0.1, 0.5], [0.4, 0.2], [0.3, 0.9]]), ["g0", "g1", "g2"], ["L1", "L2"]
    )


def _eval_res():
    # the output of evaluate_model for two settings and the ligands L1 and L2
    target = pd.DataFrame(
        {
            "aupr_corrected": [0.2, 0.4, 0.1, 0.3],
            "auroc": [0.6, 0.7, 0.55, 0.65],
            "setting": ["s1", "s2", "s1", "s2"],
            "ligand": ["L1", "L1", "L2", "L2"],
        }
    )
    ligand = pd.DataFrame(
        {
            "metric": ["aupr_corrected", "auroc", "aupr_corrected", "auroc"],
            "group": ["s1", "s1", "s2", "s2"],
            "ligand": ["L1", "L1", "L2", "L2"],
            "aupr_corrected": [0.3, 0.2, 0.5, 0.4],
            "auroc": [0.7, 0.6, 0.8, 0.75],
        }
    )
    return {"performances_target_prediction": target, "performances_ligand_prediction": ligand}


# iterated more than once: an iterator gives the same result as a list


def test_compute_evaluation_scores_generator():
    expected = compute_evaluation_scores(_eval_res(), ["L1", "L2"])
    assert compute_evaluation_scores(_eval_res(), (ligand for ligand in ["L1", "L2"])) == expected


def test_ncycle():
    assert list(ncycle([1, 2], 3)) == [1, 2, 1, 2, 1, 2]
    assert list(ncycle(iter([1, 2]), 3)) == [1, 2, 1, 2, 1, 2]
    assert list(ncycle(iter([1, 2]), 1)) == [1, 2]
    assert list(ncycle([1, 2], 0)) == []
    # the arguments are checked when ncycle is called, not when it's iterated
    with pytest.raises(TypeError):
        ncycle(5, 2)
    with pytest.raises(TypeError):
        ncycle([1, 2], 2.0)


def test_subset_ann_layers_generator():
    expected = subset_ann(_ann(), val="A", val_col="celltype", layers=["data"])
    result = subset_ann(_ann(), val="A", val_col="celltype", layers=(layer for layer in ["data"]))
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result.layers["data"], expected.layers["data"])


def test_subset_ann_genes_generator():
    expected = subset_ann(_ann(), genes=["g2", "g0"], layers=["data"])
    result = subset_ann(_ann(), genes=(gene for gene in ["g2", "g0"]), layers=["data"])
    assert result.var_names.tolist() == expected.var_names.tolist() == ["g2", "g0"]
    np.testing.assert_array_equal(result.layers["data"], expected.layers["data"])


def test_get_single_ligand_importances_generators():
    settings = [{"name": "s1", "from": "L1", "response": {"g0": 1, "g1": 0, "g2": 1}}]
    expected = get_single_ligand_importances(_predictor(), settings, ["L1", "L2"])
    result = get_single_ligand_importances(_predictor(), iter(settings), (ligand for ligand in ["L1", "L2"]))
    pd.testing.assert_frame_equal(result, expected)


def test_predict_ligand_activities_metrics_generator():
    predictor = _predictor()
    kwargs = {
        "geneset": {"g0", "g2"},
        "background_expressed_genes": {"g0", "g1", "g2"},
        "potential_ligands": ["L1", "L2"],
    }
    expected = predictor.predict_ligand_activities(target_evaluation_metrics=["aupr", "auroc"], **kwargs)
    result = predictor.predict_ligand_activities(target_evaluation_metrics=(m for m in ["aupr", "auroc"]), **kwargs)
    assert result == expected


# an Iterable of str is not a single str


def _str_calls():
    ann = _ann()
    predictor = _predictor()
    importances = pd.DataFrame({"setting": ["s1"], "true_ligand": ["L1"], "aupr": [0.5]})
    return [
        ("subset_ann genes", lambda: subset_ann(ann, genes="g0")),
        ("subset_ann layers", lambda: subset_ann(ann, val="A", val_col="celltype", layers="data")),
        ("group_metrics features", lambda: group_metrics(ann, "celltype", layer="data", features="g0")),
        ("calculate_de features", lambda: calculate_de(ann, "celltype", "c", "condition", features="g0")),
        ("get_avg_exp features", lambda: get_avg_exp(ann, "celltype", features="g0")),
        ("average_expression keys", lambda: average_expression(ann, "celltype", keys="A")),
        (
            "get_weighted_ligand_receptor_links ligands",
            lambda: get_weighted_ligand_receptor_links("L1", ["R1"], None, None),
        ),
        (
            "get_weighted_ligand_receptor_links receptors",
            lambda: get_weighted_ligand_receptor_links(["L1"], "R1", None, None),
        ),
        ("calculate_evaluation_metrics metrics", lambda: calculate_evaluation_metrics([1, 0], [0.5, 0.1], "aupr")),
        ("construct_ligand_tf_matrix ligands", lambda: construct_ligand_tf_matrix({}, "L1")),
        ("construct_ligand_target_matrix ligands", lambda: construct_ligand_target_matrix({}, pd.DataFrame(), "L1")),
        ("compute_evaluation_scores ligands", lambda: compute_evaluation_scores(_eval_res(), "L1")),
        ("get_single_ligand_importances ligands", lambda: get_single_ligand_importances(predictor, [], "L1")),
        ("get_single_ligand_importances metrics", lambda: get_single_ligand_importances(predictor, [], ["L1"], "aupr")),
        (
            "evaluate_single_importances_ligand_prediction metrics",
            lambda: evaluate_single_importances_ligand_prediction(importances, "s1", ligand_evaluation_metrics="aupr"),
        ),
        (
            "construct_and_evaluate metrics",
            lambda: construct_and_evaluate(
                {}, 0, 0, 0.5, 0.5, None, None, None, None, ligand_evaluation_metrics="aupr"
            ),
        ),
        (
            "predict_ligand_activities ligands",
            lambda: predictor.predict_ligand_activities({"g0"}, {"g0", "g1"}, potential_ligands="L1"),
        ),
        ("get_weighted_ligand_target_links geneset", lambda: predictor.get_weighted_ligand_target_links("L1", "g0")),
        ("evaluate_target_prediction metrics", lambda: predictor.evaluate_target_prediction("L1", {"g0": 1}, "aupr")),
        (
            "assess_rf_class_probabilities ligands",
            lambda: assess_rf_class_probabilities(2, {"g0"}, {"g1"}, "L1", predictor),
        ),
        ("alias_to_symbol", lambda: GeneAliasInfo({}).alias_to_symbol("ABC")),
        ("convert_human_to_mouse_symbols", lambda: GeneInfo([("TNF", "Tnf")]).convert_human_to_mouse_symbols("TNF")),
        ("convert_mouse_to_human_symbols", lambda: GeneInfo([("TNF", "Tnf")]).convert_mouse_to_human_symbols("Tnf")),
        ("create_radar_plot value_labels", lambda: create_radar_plot([[1.0, 2.0]], "ab")),
        (
            "circos plot senders",
            lambda: create_ligand_receptor_links_prioritization_circos_plot("A", ["B"], ["L1"], ["R1"], {}),
        ),
        (
            "circos plot receptors",
            lambda: create_ligand_receptor_links_prioritization_circos_plot(["A"], ["B"], ["L1"], "R1", {}),
        ),
    ]


@pytest.mark.parametrize(("name", "call"), _str_calls(), ids=[name for name, _ in _str_calls()])
def test_iterable_of_str_rejects_str(name, call):
    with pytest.raises(TypeError, match="not a single str"):
        call()


def test_not_iterable_message():
    # the message only mentions a str when a str was passed
    with pytest.raises(TypeError, match="features should be an Iterable of gene symbols, was <class 'int'>"):
        get_avg_exp(_ann(), "celltype", features=5)
    with pytest.raises(TypeError, match="features should be an Iterable of gene symbols, not a single str: 'g0'"):
        get_avg_exp(_ann(), "celltype", features="g0")


def test_get_avg_exp_features_iterable():
    # an Iterable of genes (also a generator) works, the genes in the given order
    expected = get_avg_exp(_ann(), "celltype", features=["g2", "g0"])
    result = get_avg_exp(_ann(), "celltype", features=(gene for gene in ["g2", "g0"]))
    pd.testing.assert_frame_equal(result, expected)

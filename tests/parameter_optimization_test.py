from nichenetpy.parameter_optimization import (
    evaluate_model,
    compute_evaluation_scores,
    construct_and_evaluate
)
from nichenetpy.utils import (
    read_csv_rows,
    read_csv_cols
)
from nichenetpy.evaluation import EvaluationData

from common import (
    equals_iter,
    get_model_pickle,
    get_optimization_files,
    get_network_files,
    train_path,
    network_path
)

import os
import json
import pandas as pd


def test_optimization_score_0():
    model = get_model_pickle("human")
    get_optimization_files()
    with open(os.path.join(train_path, "settings_training_f1234.json"), "rb") as file:
        settings_CV = json.loads(file.read())
    evaluation_data = EvaluationData(settings_CV["settings"])
    scores = compute_evaluation_scores(
        evaluate_model(model["predictor"], evaluation_data),
        evaluation_data.get_ligands(combination=True)
    )
    assert equals_iter(scores, (0.588, 0.949), err_bound=0.05)

def test_optimization_score_1():
    model = get_model_pickle("human")
    get_optimization_files()
    with open(os.path.join(train_path, "settings_training_f1235.json"), "rb") as file:
        settings_CV = json.loads(file.read())
    evaluation_data = EvaluationData(settings_CV["settings"])
    scores = compute_evaluation_scores(
        evaluate_model(model["predictor"], evaluation_data),
        evaluation_data.get_ligands(combination=True)
    )
    assert equals_iter(scores, (0.624, 0.954), err_bound=0.05)

def test_optimization_score_2():
    model = get_model_pickle("human")
    get_optimization_files()
    with open(os.path.join(train_path, "settings_training_f1245.json"), "rb") as file:
        settings_CV = json.loads(file.read())
    evaluation_data = EvaluationData(settings_CV["settings"])
    scores = compute_evaluation_scores(
        evaluate_model(model["predictor"], evaluation_data),
        evaluation_data.get_ligands(combination=True)
    )
    assert equals_iter(scores, (0.636, 0.959), err_bound=0.05)

def test_optimization_score_3():
    model = get_model_pickle("human")
    get_optimization_files()
    with open(os.path.join(train_path, "settings_training_f1345.json"), "rb") as file:
        settings_CV = json.loads(file.read())
    evaluation_data = EvaluationData(settings_CV["settings"])
    scores = compute_evaluation_scores(
        evaluate_model(model["predictor"], evaluation_data),
        evaluation_data.get_ligands(combination=True)
    )
    assert equals_iter(scores, (0.607, 0.970), err_bound=0.05)

def test_optimization_score_4():
    model = get_model_pickle("human")
    get_optimization_files()
    with open(os.path.join(train_path, "settings_training_f2345.json"), "rb") as file:
        settings_CV = json.loads(file.read())
    evaluation_data = EvaluationData(settings_CV["settings"])
    scores = compute_evaluation_scores(
        evaluate_model(model["predictor"], evaluation_data),
        evaluation_data.get_ligands(combination=True)
    )
    assert equals_iter(scores, (0.629, 0.966), err_bound=0.05)

def optuna_objective(
    lr_network,
    gr_network,
    sig_network,
    source_weights,
    lr_sig_hub,
    gr_hub,
    ltf_cutoff,
    damping_factor,
    evaluation_data
):
    res = construct_and_evaluate(
        source_weights,
        lr_sig_hub,
        gr_hub,
        ltf_cutoff,
        damping_factor,
        lr_network,
        gr_network,
        sig_network,
        evaluation_data
    )
    return (res[1], res[2])

def test_optuna_objective_optimized_source_weights():
    get_network_files()
    source_weights = tuple(zip(*read_csv_rows(os.path.join(network_path, "optimized_source_weights.csv"))[1]))
    source_weights = dict(zip(source_weights[0], [float(e) for e in source_weights[1]]))
    lr_network = pd.DataFrame(read_csv_cols(os.path.join(network_path, "lr_network_human.csv")))
    sig_network = pd.DataFrame(read_csv_cols(os.path.join(network_path, "lr_sig_human.csv")))
    gr_network = pd.DataFrame(read_csv_cols(os.path.join(network_path, "gr_human.csv")))
    get_optimization_files()
    with open(os.path.join(train_path, "settings_training_f1245.json"), "rb") as file:
        settings_CV = json.loads(file.read())
    evaluation_data = EvaluationData(settings_CV["settings"])
    scores = optuna_objective(
        lr_network,
        gr_network,
        sig_network,
        source_weights,
        lr_sig_hub=0.115,
        gr_hub=0.0803,
        ltf_cutoff=0.926,
        damping_factor=0.789,
        evaluation_data=evaluation_data
    )
    assert scores[0] > 0.4
    assert scores[1] > 0.9
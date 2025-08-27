from nichenetpy.parameter_optimization import (
    evaluate_model,
    compute_evaluation_scores
)
from nichenetpy.utils import extract_ligands_from_settings
from nichenetpy.parameter_optimization import (
    evaluate_model,
    compute_evaluation_scores
)

from common import (
    equals_iter,
    get_model_pickle,
    get_optimization_files,
    train_path
)

import os
import json


def test_optimization_score_0():
    model = get_model_pickle("human")
    get_optimization_files()
    with open(os.path.join(train_path, "settings_training_f1234.json"), "rb") as file:
        settings_CV = json.loads(file.read())
    settings = settings_CV["settings"]
    scores = compute_evaluation_scores(
        evaluate_model(model["predictor"], settings),
        extract_ligands_from_settings(settings, combination=True)
    )
    assert equals_iter(scores, (0.588, 0.949), err_bound=0.05)

def test_optimization_score_1():
    model = get_model_pickle("human")
    get_optimization_files()
    with open(os.path.join(train_path, "settings_training_f1235.json"), "rb") as file:
        settings_CV = json.loads(file.read())
    settings = settings_CV["settings"]
    scores = compute_evaluation_scores(
        evaluate_model(model["predictor"], settings),
        extract_ligands_from_settings(settings, combination=True)
    )
    assert equals_iter(scores, (0.624, 0.954), err_bound=0.05)

def test_optimization_score_2():
    model = get_model_pickle("human")
    get_optimization_files()
    with open(os.path.join(train_path, "settings_training_f1245.json"), "rb") as file:
        settings_CV = json.loads(file.read())
    settings = settings_CV["settings"]
    scores = compute_evaluation_scores(
        evaluate_model(model["predictor"], settings),
        extract_ligands_from_settings(settings, combination=True)
    )
    assert equals_iter(scores, (0.636, 0.959), err_bound=0.05)

def test_optimization_score_3():
    model = get_model_pickle("human")
    get_optimization_files()
    with open(os.path.join(train_path, "settings_training_f1345.json"), "rb") as file:
        settings_CV = json.loads(file.read())
    settings = settings_CV["settings"]
    scores = compute_evaluation_scores(
        evaluate_model(model["predictor"], settings),
        extract_ligands_from_settings(settings, combination=True)
    )
    assert equals_iter(scores, (0.607, 0.970), err_bound=0.05)

def test_optimization_score_4():
    model = get_model_pickle("human")
    get_optimization_files()
    with open(os.path.join(train_path, "settings_training_f2345.json"), "rb") as file:
        settings_CV = json.loads(file.read())
    settings = settings_CV["settings"]
    scores = compute_evaluation_scores(
        evaluate_model(model["predictor"], settings),
        extract_ligands_from_settings(settings, combination=True)
    )
    assert equals_iter(scores, (0.629, 0.966), err_bound=0.05)
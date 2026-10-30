# type: ignore

"""
Here we test some parts of the optimization process.
"""

from common import (
    equals_dict,
    get_model_pickle,
    get_optimization_files,
    train_path,
)
import json
import os
from itertools import chain

from common import equals_iter, get_model_pickle, get_network_files, get_optimization_files, network_path, train_path

from nichenetpy.evaluation import EvaluationData
from nichenetpy.parameter_optimization import compute_evaluation_scores, construct_and_evaluate, evaluate_model
from nichenetpy.typing import gene_t
from nichenetpy.utils import read_csv_rows, read_network_file

def template_optimization_score(filename, exp, err_bound=0.05):
    model = get_model_pickle("human")
    get_optimization_files()
    with open(os.path.join(train_path, filename), "rb") as file:
        settings_CV = json.loads(file.read())
    evaluation_data = EvaluationData(settings_CV["settings"])
    scores = compute_evaluation_scores(
        evaluate_model(model["predictor"], evaluation_data),
        evaluation_data.get_ligands(combination=True)
    )
    assert equals_dict(scores, exp, err_bound=err_bound)

def test_optimization_score_0():
    template_optimization_score(
        "settings_training_f1234.json",
        {
            "target_prediction": {
                "aupr_corrected": 0.588,
                "auroc": 0.978
            },
            "ligand_prediction": {
                "aupr_corrected": 0.949,
                "auroc": 0.991
            }
        }
    )


def test_optimization_score_1():
    template_optimization_score(
        "settings_training_f1235.json",
        {
            "target_prediction": {
                "aupr_corrected": 0.624,
                "auroc": 0.98
            },
            "ligand_prediction": {
                "aupr_corrected": 0.954,
                "auroc": 0.991
            }
        }
    )


def test_optimization_score_2():
    template_optimization_score(
        "settings_training_f1245.json",
        {
            "target_prediction": {
                "aupr_corrected": 0.636,
                "auroc": 0.978
            },
            "ligand_prediction": {
                "aupr_corrected": 0.959,
                "auroc": 0.995
            }
        }
    )


def test_optimization_score_3():
    # a bit more deviation from NNv2 than usual
    template_optimization_score(
        "settings_training_f1345.json",
        {
            "target_prediction": {
                "aupr_corrected": 0.607,
                "auroc": 0.98
            },
            "ligand_prediction": {
                "aupr_corrected": 0.97,
                "auroc": 0.949
            }
        },
        err_bound=0.06
    )

def test_optimization_score_4():
    template_optimization_score(
        "settings_training_f2345.json",
        {
            "target_prediction": {
                "aupr_corrected": 0.629,
                "auroc": 0.986
            },
            "ligand_prediction": {
                "aupr_corrected": 0.966,
                "auroc": 0.994
            }
        }
    )


def optuna_objective(
    lr_network, gr_network, sig_network, source_weights, lr_sig_hub, gr_hub, ltf_cutoff, damping_factor, evaluation_data
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
        evaluation_data,
        return_all_matrices=False,
        return_weighted_networks=False,
    )
    return res[1]


def test_optuna_objective_optimized_source_weights():
    get_network_files()
    source_weights = tuple(
        zip(*read_csv_rows(os.path.join(network_path, "optimized_source_weights.csv"))[1], strict=True)
    )
    source_weights = dict(zip(source_weights[0], [float(e) for e in source_weights[1]], strict=True))
    lr_network = read_network_file(os.path.join(network_path, "lr_network_human.csv"))
    sig_network = read_network_file(os.path.join(network_path, "lr_sig_human.csv"))
    gr_network = read_network_file(os.path.join(network_path, "gr_human.csv"))
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
        evaluation_data=evaluation_data,
    )
    assert scores["target_prediction"]["auroc"] > 0.9
    assert scores["target_prediction"]["aupr_corrected"] > 0.4
    assert scores["ligand_prediction"]["auroc"]  > 0.9
    assert scores["ligand_prediction"]["aupr_corrected"]  > 0.9


def test_optuna_objective_optimized_source_weights_with_integer_mapping():
    get_network_files()
    source_weights = tuple(
        zip(*read_csv_rows(os.path.join(network_path, "optimized_source_weights.csv"))[1], strict=True)
    )
    source_weights = dict(zip(source_weights[0], [float(e) for e in source_weights[1]], strict=True))
    lr_network = read_network_file(os.path.join(network_path, "lr_network_human.csv"))
    sig_network = read_network_file(os.path.join(network_path, "lr_sig_human.csv"))
    gr_network = read_network_file(os.path.join(network_path, "gr_human.csv"))
    get_optimization_files()
    with open(os.path.join(train_path, "settings_training_f1245.json"), "rb") as file:
        settings_CV = json.loads(file.read())
    evaluation_data = EvaluationData(settings_CV["settings"])
    # database column is no longer required so remove to save memory
    lr_network.drop("database", axis=1, inplace=True)
    gr_network.drop("database", axis=1, inplace=True)
    sig_network.drop("database", axis=1, inplace=True)
    # map strings to integers to save a lot of memory in the subprocesses
    syms = sorted(
        set(
            chain(
                lr_network["from"],
                lr_network["to"],
                lr_network["source"],
                gr_network["from"],
                gr_network["to"],
                gr_network["source"],
                sig_network["from"],
                sig_network["to"],
                sig_network["source"],
                evaluation_data.get_ligands(combination=False),
                set(chain(k for e in evaluation_data.values() for k in e[evaluation_data._de_genes_name].keys())),
            )
        )
    )
    sym2id = dict(zip(syms, range(len(syms)), strict=True))
    lr_network["from"] = [sym2id[e] for e in lr_network["from"]]
    lr_network["to"] = [sym2id[e] for e in lr_network["to"]]
    lr_network["source"] = [sym2id[e] for e in lr_network["source"]]
    gr_network["from"] = [sym2id[e] for e in gr_network["from"]]
    gr_network["to"] = [sym2id[e] for e in gr_network["to"]]
    gr_network["source"] = [sym2id[e] for e in gr_network["source"]]
    sig_network["from"] = [sym2id[e] for e in sig_network["from"]]
    sig_network["to"] = [sym2id[e] for e in sig_network["to"]]
    sig_network["source"] = [sym2id[e] for e in sig_network["source"]]
    for dct in evaluation_data.values():
        ligand = dct[evaluation_data._ligand_name]
        dct[evaluation_data._ligand_name] = (
            sym2id[ligand] if isinstance(ligand, gene_t.__value__) else tuple(sym2id[e] for e in ligand)
        )
        dct[evaluation_data._de_genes_name] = {sym2id[k]: v for k, v in dct[evaluation_data._de_genes_name].items()}
    scores = optuna_objective(
        lr_network,
        gr_network,
        sig_network,
        {sym2id[s]: w for s, w in source_weights.items()},
        lr_sig_hub=0.115,
        gr_hub=0.0803,
        ltf_cutoff=0.926,
        damping_factor=0.789,
        evaluation_data=evaluation_data,
    )
    assert scores["target_prediction"]["auroc"] > 0.9
    assert scores["target_prediction"]["aupr_corrected"] > 0.4
    assert scores["ligand_prediction"]["auroc"]  > 0.9
    assert scores["ligand_prediction"]["aupr_corrected"]  > 0.9

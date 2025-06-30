from nichenetpy.utils import (
    extract_ligands_from_settings
)
from nichenetpy.model_construction import (
    construct_weighted_networks,
    construct_ligand_target_matrix,
    apply_hub_correction
)
from nichenetpy.evaluation import (
    evaluate_single_importances_ligand_prediction
)
from nichenetpy.prediction import LigandActivityPredictor

from itertools import chain, repeat
from collections.abc import Iterable

import pandas as pd


def evaluate_model(
    predictor: LigandActivityPredictor,
    settings: dict
):
    performances_target_prediction = {
        k: predictor.evaluate_target_prediction(v["from"] if type(v["from"]) is str else "-".join(v["from"]), v["response"])
        for k, v in settings.items()
    }
    all_ligands = extract_ligands_from_settings(settings, combination=False)
    ligand_importances = {
        "setting": [],
        "test_ligand": [],
        "true_ligand": [],
        "auroc": [],
        "pearson": [],
        "aupr": [],
        "aupr_corrected": []
    }
    for setting_id, setting in list(settings.items()):
        for ligand in all_ligands:
            ligand_importances["setting"].append(setting_id)
            ligand_importances["test_ligand"].append(ligand)
            ligand_importances["true_ligand"].append(setting["from"])
            for k, v in predictor.evaluate_target_prediction(ligand, setting["response"]).items():
                ligand_importances[k].append(v)
    ligand_importances = pd.DataFrame(ligand_importances)
    # TODO: deal with potential NaNs
    performances_ligand_prediction_single = [
        evaluate_single_importances_ligand_prediction(ligand_importances, group=setting_id)
        for setting_id in set(ligand_importances["setting"])
    ]
    return {
        "performances_target_prediction": performances_target_prediction,
        "performances_ligand_prediction_single": performances_ligand_prediction_single
    }

def construct_and_evaluate(
    lr_network: pd.DataFrame,
    gr_network: pd.DataFrame,
    sig_network: pd.DataFrame,
    settings: dict
):
    ligands = extract_ligands_from_settings(settings)
    source_weights = dict(zip(set(chain(gr_network["source"], lr_network["source"], sig_network["source"])), repeat(1)))
    weighted_networks = construct_weighted_networks(
        lr_network,
        sig_network,
        gr_network,
        source_weights
    )
    weighted_networks["lr_sig"] = apply_hub_correction(weighted_networks["lr_sig"], hub=0.115)
    weighted_networks["gr"] = apply_hub_correction(weighted_networks["gr"], hub=0.0803)
    predictor = LigandActivityPredictor(
        *construct_ligand_target_matrix(
            weighted_networks,
            lr_network,
            ligands,
            damping_factor=0.789,
            ltf_cutoff=0.926
        )
    )
    predictor.replace_zero_col_by_noisy_scores()
    eval_res = evaluate_model(predictor, settings)
    return {
        "predictor": predictor,
        "performances_target_prediction": eval_res["performances_target_prediction"],
        "performances_ligand_prediction_single": eval_res["performances_ligand_prediction_single"]
    }
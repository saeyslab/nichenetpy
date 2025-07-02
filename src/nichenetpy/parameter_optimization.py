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

import pandas as pd


def _average_performances(ligand_oi, performances):
    performances_oi = performances[[
        any(
            ligand in true_ligand if type(true_ligand) is list else ligand == true_ligand
            for ligand in (
                (ligand_oi,) if type(ligand_oi) is str else ligand_oi
            )
        )
        for true_ligand in performances["ligand"]
    ]]
    return performances_oi["aupr_corrected"].median()

def evaluate_model(
    predictor: LigandActivityPredictor,
    settings: dict
):
    performances_target_prediction = {
        "setting": [],
        "ligand": [],
        "auroc": [],
        "pearson": [],
        "aupr": [],
        "aupr_corrected": []
    }
    for setting_id, setting in settings.items():
        performances_target_prediction["setting"].append(setting_id)
        performances_target_prediction["ligand"].append(setting["from"])
        for k, v in predictor.evaluate_target_prediction(
            setting["from"]
            if type(setting["from"]) is str
            else "-".join(setting["from"]),
            setting["response"]
        ).items():
            performances_target_prediction[k].append(v)
    performances_target_prediction = pd.DataFrame(performances_target_prediction)
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
    for setting_id, setting in settings.items():
        for ligand in all_ligands:
            ligand_importances["setting"].append(setting_id)
            ligand_importances["test_ligand"].append(ligand)
            ligand_importances["true_ligand"].append(setting["from"])
            for k, v in predictor.evaluate_target_prediction(ligand, setting["response"]).items():
                ligand_importances[k].append(v)
    ligand_importances = pd.DataFrame(ligand_importances)
    # TODO: deal with potential NaNs
    performances_ligand_prediction_single = pd.concat(
        evaluate_single_importances_ligand_prediction(ligand_importances, group=setting_id)
        for setting_id in set(ligand_importances["setting"])
    )
    return {
        "performances_target_prediction": performances_target_prediction,
        "performances_ligand_prediction": performances_ligand_prediction_single
    }

def construct_and_evaluate(
    source_weights:dict[str, float]|pd.DataFrame,
    lr_network: pd.DataFrame,
    gr_network: pd.DataFrame,
    sig_network: pd.DataFrame,
    settings: dict
):
    ligands = extract_ligands_from_settings(settings)
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
        "performances_ligand_prediction": eval_res["performances_ligand_prediction"]
    }
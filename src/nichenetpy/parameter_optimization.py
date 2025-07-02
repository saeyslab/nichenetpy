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
import numpy as np


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
    '''
    Evaluate the ligand-target matrix. 

    Parameters
    ----------
    predictor : LigandActivityPredictor
        The predictor that holds the ligand-target matrix to evaluate
    settings : dict
        An Iterable of dictionaries that have the following keys: 
        
            name: the name of the setting

            ligand: the name of the ligand which is known to be active in the setting of interest
        
            from:  the name of the ligand of which the predictive performance need to be assessed
        
            response:   the observed target response, indicates for a gene whether it was a target
                        or not in the setting of interest

    Returns
    -------
    dict
        A dictionary with keys 'performances_target_prediction' and 'performances_ligand_prediction'

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(predictor) is not LigandActivityPredictor:
        raise TypeError(f"predictor should have type LigandActivityPredictor, was {type(predictor)}")
    if type(settings) is not dict:
        raise TypeError(f"settings should have type dict, was {type(settings)}")
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
    '''
    Construct and evaluate the ligand-target matrix. 
    Returns the matrices and the prediction scores

    Parameters
    ----------
    source_weights : pandas.DataFrame or dictionary
        Dataframe or dictionary which contains the weights associated to each individual data source.
        Sources with higher weights will contribute more to the final model performance.
        Note that only interactions described by sources included here, will be retained during model construction.
    lr_network : pandas.DataFrame
        dataframe which contains ligand-receptor interactions
    gr_network : pandas.DataFrame
        dataframe which contains gene regulatory interactions
    sig_network : pandas.DataFrame
        dataframe which contains signaling interactions
    settings : dict
        An Iterable of dictionaries that have the following keys: 
        
            name: the name of the setting

            ligand: the name of the ligand which is known to be active in the setting of interest
        
            from:  the name of the ligand of which the predictive performance need to be assessed
        
            response:   the observed target response, indicates for a gene whether it was a target
                        or not in the setting of interest

    Returns
    -------
    dict
        A dictionary with keys 'weighted networks', 'grn matrix', 'ltf matrix' and 'ligand-target matrix'
    float
        target prediction score
    float
        ligand prediction score
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    ligands = extract_ligands_from_settings(settings)
    weighted_networks = construct_weighted_networks(
        lr_network,
        sig_network,
        gr_network,
        source_weights
    )
    weighted_networks["lr_sig"] = apply_hub_correction(weighted_networks["lr_sig"], hub=0.115)
    weighted_networks["gr"] = apply_hub_correction(weighted_networks["gr"], hub=0.0803)
    ligand2target, grn_matrix, ltf_matrix = construct_ligand_target_matrix(
        weighted_networks,
        lr_network,
        ligands,
        damping_factor=0.789,
        ltf_cutoff=0.926,
        return_all_matrices=True
    )
    predictor = LigandActivityPredictor(*ligand2target)
    predictor.replace_zero_col_by_noisy_scores()
    eval_res = evaluate_model(predictor, settings)
    ligands_evaluation = extract_ligands_from_settings(settings, combination=True)
    performances_target_prediction_averaged = [
        _average_performances(ligand, eval_res["performances_target_prediction"])
        for ligand in ligands_evaluation
    ]
    ligand_activity_performance_setting_summary = eval_res["performances_ligand_prediction"][[
        "metric",
        "aupr",
        "aupr_corrected",
        "auroc",
        "pearson"
    ]].groupby("metric").mean()
    ligand_activity_performance_setting_summary["geom_average"] = [
        np.exp((np.log(aupr) + np.log(auroc)) / 2)
        for aupr, auroc in zip(
            ligand_activity_performance_setting_summary["aupr_corrected"],
            ligand_activity_performance_setting_summary["auroc"]
        )
    ]
    ligand_activity_performance_setting_summary.reset_index(inplace=True)
    best_metric = max(
        zip(
            ligand_activity_performance_setting_summary["metric"],
            ligand_activity_performance_setting_summary["geom_average"]
        ),
        key=lambda x : x[1]
    )[0]
    performances_ligand_prediction_summary = eval_res["performances_ligand_prediction"][
        eval_res["performances_ligand_prediction"]["metric"] == best_metric
    ]
    performances_ligand_prediction_averaged = [
        _average_performances(ligand, performances_ligand_prediction_summary)
        for ligand in ligands_evaluation
    ]
    return (
        {
            "weighted networks": weighted_networks,
            "grn matrix": grn_matrix,
            "ltf matrix": ltf_matrix,
            "ligand-target matrix": ligand2target
        },
        np.mean(performances_target_prediction_averaged),
        (np.median(performances_ligand_prediction_averaged) + np.mean(performances_ligand_prediction_averaged)) / 2
    )
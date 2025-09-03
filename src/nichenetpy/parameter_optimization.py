from nichenetpy.utils import (
    extract_ligands_from_settings
)
from nichenetpy.model_construction import (
    construct_weighted_networks,
    construct_ligand_target_matrix,
    construct_tf_target_matrix,
    apply_hub_correction
)
from nichenetpy.evaluation import (
    evaluate_single_importances_ligand_prediction
)
from nichenetpy.prediction import LigandActivityPredictor

from collections.abc import Iterable

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

def _evaluate_single_importances_ligand_prediction(
    ligand_importances,
    group
):
    try:
        return evaluate_single_importances_ligand_prediction(ligand_importances, group)
    except ValueError:
        return None

def evaluate_model(
    predictor:LigandActivityPredictor,
    settings:dict
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
        try:
            performances_target_prediction["ligand"].append(setting["from"])
            for k, v in predictor.evaluate_target_prediction(
                setting["from"]
                if type(setting["from"]) is str
                else "-".join(setting["from"]),
                setting["response"]
            ).items():
                performances_target_prediction[k].append(v)
        except ValueError:
            # the metrics are undefined -> roleback
            max_len = len(performances_target_prediction["setting"]) - 1
            for e in performances_target_prediction.values():
                if len(e) > max_len:
                    e.pop()
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
            try:
                ligand_importances["setting"].append(setting_id)
                ligand_importances["test_ligand"].append(ligand)
                ligand_importances["true_ligand"].append(setting["from"])
                for k, v in predictor.evaluate_target_prediction(ligand, setting["response"]).items():
                    ligand_importances[k].append(v)
            except ValueError:
                # the metrics are undefined -> roleback
                max_len = len(ligand_importances["setting"]) - 1
                for e in ligand_importances.values():
                    if len(e) > max_len:
                        e.pop()
    ligand_importances = pd.DataFrame(ligand_importances)
    performances_ligand_prediction_single = [
        e for e in (
            _evaluate_single_importances_ligand_prediction(ligand_importances, group=setting_id)
            for setting_id in set(ligand_importances["setting"])
        ) if e is not None
    ]
    if len(performances_ligand_prediction_single) > 0:
        performances_ligand_prediction_single = pd.concat(performances_ligand_prediction_single)
    else:
        performances_ligand_prediction_single = None
    return {
        "performances_target_prediction": performances_target_prediction,
        "performances_ligand_prediction": performances_ligand_prediction_single
    }

def compute_evaluation_scores(
    eval_res:dict[str, pd.DataFrame],
    ligands:Iterable[str]
) -> tuple[float, float]:
    '''
    Construct and evaluate the ligand-target matrix. 
    Returns the matrices and the prediction scores

    Parameters
    ----------
    eval_res : dict[str, pd.DataFrame]
        The output of a call to `nichenetpy.parameter_optimization.evaluate_model`
    ligands : Iterable of str
        the ligands of interest

    Returns
    -------
    float
        target prediction score
    float
        ligand prediction score

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    performances_target_prediction_averaged = [
        e for e in (
            _average_performances(ligand, eval_res["performances_target_prediction"])
            for ligand in ligands
        ) if not np.isnan(e)
    ]
    if eval_res["performances_ligand_prediction"] is None:
        return (
            np.mean(performances_target_prediction_averaged),
            0
        )
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
        e for e in (
            _average_performances(ligand, performances_ligand_prediction_summary)
            for ligand in ligands
        ) if not np.isnan(e)
    ]
    return (
        np.mean(performances_target_prediction_averaged),
        (np.median(performances_ligand_prediction_averaged) + np.mean(performances_ligand_prediction_averaged)) / 2
    )

def construct_and_evaluate(
    source_weights:dict[str, float]|pd.DataFrame,
    lr_sig_hub:float,
    gr_hub:float,
    ltf_cutoff:float,
    damping_factor:float,
    lr_network:pd.DataFrame,
    gr_network:pd.DataFrame,
    sig_network:pd.DataFrame,
    settings:dict
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
    lr_sig_hub : float
        a number between 0 (no correction for hubiness) and 1 (maximal correction for hubiness)
    gr_hub : float
        a number between 0 (no correction for hubiness) and 1 (maximal correction for hubiness)
    ltf_cutoff : float
        ligand-tf scores beneath the "ltf_cutoff" quantile will be set to 0.
        Default: 0.99 such that only the 1 percent closest tfs will be considered as possible tfs downstream of the ligand of choice.
    damping_factor : float
        Only relevant when algorithm is PPR.
        In the PPR algorithm, the damping factor is the probability that the random walker will continue its walk on the graph;
        1-damping factor is the probability that the walker will return to the seed node.
    lr_network : pandas.DataFrame
        dataframe which contains ligand-receptor interactions
    gr_network : pandas.DataFrame
        dataframe which contains gene regulatory interactions
    sig_network : pandas.DataFrame
        dataframe which contains signaling interactions
    settings : dict
        A dictionary of dictionaries that have the following keys: 
        
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
    if sum(source_weights.values()) == 0:
        return (
            {
                "weighted networks": None,
                "grn matrix": None,
                "ltf matrix": None,
                "ligand-target matrix": None
            },
            0,
            0
        )
    ligands = extract_ligands_from_settings(settings)
    weighted_networks = construct_weighted_networks(
        lr_network,
        sig_network,
        gr_network,
        source_weights
    )
    if weighted_networks["lr_sig"].shape[0] > 0:
        weighted_networks["lr_sig"] = apply_hub_correction(weighted_networks["lr_sig"], hub=lr_sig_hub)
        weighted_networks["gr"] = apply_hub_correction(weighted_networks["gr"], hub=gr_hub)
        ligand2target, grn_matrix, ltf_matrix = construct_ligand_target_matrix(
            weighted_networks,
            lr_network,
            ligands,
            damping_factor=damping_factor,
            ltf_cutoff=ltf_cutoff,
            return_all_matrices=True
        )
    else:
        grn_matrix = construct_tf_target_matrix(
            weighted_networks,
            standalone_output=True
        )
        ligand2target = (grn_matrix[0].toarray(), grn_matrix[1], grn_matrix[2])
        ltf_matrix = None
    predictor = LigandActivityPredictor(*ligand2target)
    predictor.replace_zero_col_by_noisy_scores()
    scores = compute_evaluation_scores(
        evaluate_model(predictor, settings),
        extract_ligands_from_settings(settings, combination=True)
    )
    return (
        {
            "weighted networks": weighted_networks,
            "grn matrix": grn_matrix,
            "ltf matrix": ltf_matrix,
            "ligand-target matrix": ligand2target
        },
        scores[0],
        scores[1]
    )
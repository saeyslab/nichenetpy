from nichenetpy.model_construction import (
    construct_model_from_source_weights
)
from nichenetpy.evaluation import (
    EvaluationData,
    evaluate_single_importances_ligand_prediction
)
from nichenetpy.prediction import LigandActivityPredictor

from collections.abc import (
    Iterable,
    Callable
)

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
    return (
        performances_oi["auroc"].median(),
        performances_oi["aupr_corrected"].median()
    )

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
    evaluation_data:EvaluationData
):
    '''
    Evaluate the ligand-target matrix. 

    Parameters
    ----------
    predictor : LigandActivityPredictor
        The predictor that holds the ligand-target matrix to evaluate
    evaluation_data : EvaluationData
        The evaluation data

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
    if type(evaluation_data) is not EvaluationData:
        raise TypeError(f"settings should have type EvaluationData, was {type(evaluation_data)}")
    performances_target_prediction = {
        "setting": [],
        "ligand": [],
        "auroc": [],
        "pearson": [],
        "aupr": [],
        "aupr_corrected": []
    }
    for setting_id, setting in evaluation_data.items():
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
    all_ligands = evaluation_data.get_ligands(combination=False)
    ligand_importances = {
        "setting": [],
        "test_ligand": [],
        "true_ligand": [],
        "auroc": [],
        "pearson": [],
        "aupr": [],
        "aupr_corrected": []
    }
    for setting_id, setting in evaluation_data.items():
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
) -> tuple[float, float, float, float]:
    '''
    Construct and evaluate the ligand-target matrix. 

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
    performances_target_prediction_averaged_auroc, performances_target_prediction_averaged_aupr = zip(
        *(_average_performances(ligand, eval_res["performances_target_prediction"])
        for ligand in ligands
    ))
    performances_target_prediction_averaged_auroc = [e for e in performances_target_prediction_averaged_auroc if not np.isnan(e)]
    performances_target_prediction_averaged_aupr = [e for e in performances_target_prediction_averaged_aupr if not np.isnan(e)]
    if eval_res["performances_ligand_prediction"] is None:
        return (
            np.mean(performances_target_prediction_averaged_auroc),
            np.mean(performances_target_prediction_averaged_aupr),
            0,
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
    performances_ligand_prediction_averaged_auroc, performances_ligand_prediction_averaged_aupr = zip(
        *(_average_performances(ligand, performances_ligand_prediction_summary)
        for ligand in ligands
    ))
    performances_ligand_prediction_averaged_auroc = [e for e in performances_ligand_prediction_averaged_auroc if not np.isnan(e)]
    performances_ligand_prediction_averaged_aupr = [e for e in performances_ligand_prediction_averaged_aupr if not np.isnan(e)]
    return (
        np.mean(performances_target_prediction_averaged_auroc),
        np.mean(performances_target_prediction_averaged_aupr),
        (np.median(performances_ligand_prediction_averaged_auroc) + np.mean(performances_ligand_prediction_averaged_auroc)) / 2,
        (np.median(performances_ligand_prediction_averaged_aupr) + np.mean(performances_ligand_prediction_averaged_aupr)) / 2
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
    evaluation_data:EvaluationData
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
    evaluation_data : EvaluationData
        The evaluation data

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
            0,
            0,
            0
        )
    model = construct_model_from_source_weights(
        source_weights,
        lr_sig_hub,
        gr_hub,
        ltf_cutoff,
        damping_factor,
        lr_network,
        gr_network,
        sig_network,
        ligands=evaluation_data.get_ligands()
    )
    # make sure the ligand-target matrix is column-major, this will speed up the nichenet analysis which heavily relies on column indexing
    # the optimization as a whole is also faster despite the copy each trial
    ligand2target, row_names, col_names = model["ligand-target matrix"]
    if ligand2target.flags.c_contiguous:
        ligand2target = np.array(ligand2target, order="F")
    predictor = LigandActivityPredictor(ligand2target, row_names, col_names)
    predictor.replace_zero_col_by_noisy_scores()
    scores = compute_evaluation_scores(
        evaluate_model(predictor, evaluation_data),
        evaluation_data.get_ligands(combination=True)
    )
    return (
        model,
        scores[0],
        scores[1],
        scores[2],
        scores[3]
    )

def weighted_stress_function(
    w:float,
    d1:float=0.002,
    d2:float=0.008
) -> Callable[[float], float]:
    '''
    construct a weighted stress function

    Parameters
    ----------
    w : float
        the weight
    d1 : float
        a small correction
    d2 : float
        a small correction

    Returns
    -------
    Callable
        the weighted stress function
    '''
    a = 0.75 * (1 - w)**2 + 2*(1 - w) + d1
    b = a + 4*w - 2
    c = 1 - np.tan(np.pi*(w - 0.5) / (1 + d2)) / np.tan(-np.pi / (2*(1 + d2)))
    return lambda x : (
        (w / 2) * np.tan(-np.pi*(x - w) / b) + c
        if x <= w else
        c * (1 - np.tan(-np.pi*(x - w) / a) / np.tan(np.pi*(w - 1) / a))
    )

def choose_pareto_optimal_solution(
    objective_values:Iterable[Iterable[float]],
    weights:Iterable[float]
) -> int:
    '''
    Choose one solution from a set of pareto optimal solutions using the weighted stress function method

    Parameters
    ----------
    objective_values : Iterable of Iterable of float
        the values of the objectives for each solution
    weights : Iterable of float
        the preference weights of the objectives

    Returns
    -------
    int
        the chosen solution
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if not isinstance(objective_values, Iterable):
        raise TypeError(f"objective_values should be iterable, was {type(objective_values)}")
    if not isinstance(weights, Iterable):
        raise TypeError(f"weights should be iterable, was {type(weights)}")
    if type(weights) is np.ndarray:
        weights /= np.sum(weights)
    else:
        tw = sum(weights)
        weights = [weight/tw for weight in weights]
    fs = [weighted_stress_function(weight) for weight in weights]
    return np.argmin([
        sum(
            np.abs(fs[i](xs[i]) - fs[j](xs[j]))
            for j in range(len(xs))
            for i in range(j)
        )
        for xs in objective_values
    ])
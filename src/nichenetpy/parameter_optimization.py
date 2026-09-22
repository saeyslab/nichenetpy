from nichenetpy.model_construction import (
    construct_model_from_source_weights
)
from nichenetpy.evaluation import (
    EvaluationData,
    evaluate_single_importances_ligand_prediction
)
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.typing import gene_t

from collections.abc import (
    Iterable,
    Callable
)
from traceback import format_exc
from itertools import chain

import pandas as pd
import numpy as np
import warnings


def _average_performances(
    ligand_oi,
    performances,
    metrics=("auroc", "aupr_corrected")
):
    # true ligand must match with ligand(s) of interest
    performances_oi = performances[[
        any(
            ligand in true_ligand if type(true_ligand) is list else ligand == true_ligand
            for ligand in (
                (ligand_oi,) if isinstance(ligand_oi, gene_t) else ligand_oi
            )
        )
        for true_ligand in performances["ligand"]
    ]]
    return tuple((performances_oi[metric].median() for metric in metrics))

def _evaluate_single_importances_ligand_prediction(
    ligand_importances,
    group,
    ligand_evaluation_metrics:Iterable[str]=("aupr", "aupr_corrected", "auroc", "pearson"),
    target_evaluation_metrics:Iterable[str]=("aupr", "aupr_corrected", "auroc", "pearson")
):
    try:
        return evaluate_single_importances_ligand_prediction(
            ligand_importances,
            group,
            allow_nan=True,
            ligand_evaluation_metrics=ligand_evaluation_metrics,
            target_evaluation_metrics=target_evaluation_metrics
        )
    except Exception as ex:
        warnings.warn(f"Could not evaluate ligand importance scores for {group}:\n{ex}\n{format_exc()}")
        return None

def evaluate_model(
    predictor:LigandActivityPredictor,
    evaluation_data:EvaluationData,
    ligand_evaluation_metrics:Iterable[str]=("aupr", "aupr_corrected", "auroc", "pearson"),
    target_evaluation_metrics:Iterable[str]=("aupr", "aupr_corrected", "auroc", "pearson")
):
    '''
    Evaluate the ligand-target matrix. 

    Parameters
    ----------
    predictor : LigandActivityPredictor
        The predictor that holds the ligand-target matrix to evaluate
    evaluation_data : EvaluationData
        The evaluation data
    ligand_evaluation_metrics : Iterable of string
        the ligand prediction evaluation metrics to compute, must be a subset of ("aupr", "aupr_corrected", "auroc", "pearson", "map", "ndcg")
    target_evaluation_metrics : Iterable of string
        the target prediction evaluation metrics to compute, must be a subset of ("aupr", "aupr_corrected", "auroc", "pearson", "map", "ndcg")

    Returns
    -------
    dict
        A dictionary with keys 'performances_target_prediction' and 'performances_ligand_prediction'

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    
    Notes
    -----
    When the model can't be evaluated on a golden standard dataset, this particuler dataset is ignored. 
    For instance if the intersection between the genes in the ligand-target matrix and the genes in the
    GS set are genes that aren't expressed then the model can't be evaluated on this GS set. 
    '''
    if not isinstance(predictor, LigandActivityPredictor):
        raise TypeError(f"predictor should have type LigandActivityPredictor, was {type(predictor)}")
    if type(evaluation_data) is not EvaluationData:
        raise TypeError(f"settings should have type EvaluationData, was {type(evaluation_data)}")
    performances_target_prediction = {
        "setting": [],
        "ligand": []
    }
    for met in target_evaluation_metrics:
        performances_target_prediction[met] = []
    evaluation_data = EvaluationData((e[1] for e in evaluation_data.get_applicable_evaluation_datasets(predictor)))
    for setting_id, setting in evaluation_data.items():
        performances_target_prediction["setting"].append(setting_id)
        performances_target_prediction["ligand"].append(setting["from"])
        for k, v in predictor.evaluate_target_prediction(
            (
                setting["from"]
                if isinstance(setting["from"], gene_t) # potential BUG when using integers
                else "-".join(setting["from"])
            ),
            setting["response"],
            target_evaluation_metrics
        ).items():
            performances_target_prediction[k].append(v)
    performances_target_prediction = pd.DataFrame(performances_target_prediction)
    all_ligands = evaluation_data.get_ligands(combination=False)
    ligand_importances = {
        "setting": [],
        "test_ligand": [],
        "true_ligand": []
    }
    for met in target_evaluation_metrics:
        ligand_importances[met] = []
    for setting_id, setting in evaluation_data.items():
        for ligand in all_ligands:
            ligand_importances["setting"].append(setting_id)
            ligand_importances["test_ligand"].append(ligand)
            ligand_importances["true_ligand"].append(setting["from"])
            for k, v in predictor.evaluate_target_prediction(
                ligand,
                setting["response"],
                target_evaluation_metrics
            ).items():
                ligand_importances[k].append(v)
    ligand_importances = pd.DataFrame(ligand_importances)
    performances_ligand_prediction_single = [
        e for e in (
            _evaluate_single_importances_ligand_prediction(
                ligand_importances,
                group=setting_id,
                ligand_evaluation_metrics=ligand_evaluation_metrics,
                target_evaluation_metrics=target_evaluation_metrics
            )
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
    ligands:Iterable[gene_t],
    metric_score_f:Callable=lambda aupr_corrected, auroc : np.exp((np.log(aupr_corrected) + np.log(auroc)) / 2),
    objective_fs:dict[str, dict[str, Callable]]={
        "target_prediction": {
            "aupr_corrected": np.mean,
            "auroc": np.mean
        },
        "ligand_prediction": {
            "aupr_corrected": lambda x : np.mean(x) + np.median(x),
            "auroc": lambda x : np.mean(x) + np.median(x)
        }
    }
):
    '''
    Construct and evaluate the ligand-target matrix. 

    Parameters
    ----------
    eval_res : dict[str, pd.DataFrame]
        The output of a call to `nichenetpy.parameter_optimization.evaluate_model`
    ligands : Iterable of gene_t
        The ligands of interest
    metric_score_f : Callable
        Function that takes ligand prediction evaluation metrics as input and returns a score that can be used to rank target
        prediction evaluation metrics
    objective_fs : dict of dict[str, Callable]
        Dictionary which maps the keys "target_prediction" and "ligand_prediction" to dictionaries which map metrics to
        functions that aggregate values of said metric. These functions are used to compute the optimization objectives
        which are aggregated from metric values computed over multiple data sets. 

    Returns
    -------
    dict
        nested dictionary with keys "target_prediction" and "ligand_prediction", the nested dictionaries have metric names as keys

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(eval_res) is not dict:
        raise TypeError(f"eval_res should have type dict, was {type(eval_res)}")
    if not isinstance(ligands, Iterable):
        raise TypeError(f"ligands should have type Iterable, was {type(ligands)}")
    if not isinstance(metric_score_f, Callable):
        raise TypeError(f"metric_score_f should have type Callable, was {type(metric_score_f)}")
    if type(objective_fs) is not dict:
        raise TypeError(f"objective_fs should have type dict, was {type(objective_fs)}")
    if len(objective_fs) != 2 or "target_prediction" not in objective_fs.keys() or "ligand_prediction" not in objective_fs.keys():
        raise KeyError(f"objective_fs should have exactly 'target_prediction' and 'ligand_prediction' as keys, got {objective_fs.keys()}")
    target_evaluation_metrics = set(eval_res["performances_target_prediction"].columns)
    for e in ("setting", "ligand"):
        target_evaluation_metrics.remove(e)
    ligand_evaluation_metrics = set(eval_res["performances_ligand_prediction"].columns)
    for e in ("metric", "group", "ligand"):
        ligand_evaluation_metrics.remove(e)
    # median metric value per ligand for each metric
    performances_target_prediction_averaged = dict(zip(
        target_evaluation_metrics,
        zip(
            *(
                _average_performances(
                    ligand,
                    eval_res["performances_target_prediction"],
                    target_evaluation_metrics
                )
                for ligand in ligands
            )
        )
    ))
    for e in performances_target_prediction_averaged.keys():
        performances_target_prediction_averaged[e] = [e for e in performances_target_prediction_averaged[e] if not np.isnan(e)]
    if eval_res["performances_ligand_prediction"] is None:
        # only target prediction
        return {
            "target_prediction": {
                metric: f(performances_target_prediction_averaged[metric]) for metric, f in objective_fs["target_prediction"].items()
            },
            "ligand_prediction": {
                metric: 0 for metric in objective_fs["ligand_prediction"].keys()
            }
        }
    ligand_activity_performance_setting_summary = eval_res["performances_ligand_prediction"][
        list(chain(
            ("metric",),
            ligand_evaluation_metrics
        ))
    ].groupby("metric").mean()
    # compute a score for ligand prediction evaluation metrics (so we can rank them)
    ligand_activity_performance_setting_summary["score"] = [
        metric_score_f(**dict(zip(ligand_evaluation_metrics, mts))) # dictionary init takes ~1/6 as much time as geom avg? needs check... 
        for mts in zip(
            *(ligand_activity_performance_setting_summary[met] for met in ligand_evaluation_metrics)
        )
    ]
    ligand_activity_performance_setting_summary.reset_index(inplace=True)
    # find the best target prediction evaluation metric by comparing the geometric average of each metric
    best_metric = max(
        zip(
            ligand_activity_performance_setting_summary["metric"],
            ligand_activity_performance_setting_summary["score"]
        ),
        key=lambda x : x[1]
    )[0]
    # only keep the best target prediction evaluation metric
    performances_ligand_prediction_summary = eval_res["performances_ligand_prediction"][
        eval_res["performances_ligand_prediction"]["metric"] == best_metric
    ]
    # median metric value for each metric per ligand
    performances_ligand_prediction_averaged = dict(zip(
        ligand_evaluation_metrics,
        zip(
            *(
                _average_performances(
                    ligand,
                    performances_ligand_prediction_summary,
                    ligand_evaluation_metrics
                )
                for ligand in ligands
            )
        )
    ))
    for e in performances_ligand_prediction_averaged.keys():
        performances_ligand_prediction_averaged[e] = [e for e in performances_ligand_prediction_averaged[e] if not np.isnan(e)]
    # aggregate metrics over ligands
    return {
        metric_type: {
            metric: f(performances_target_prediction_averaged[metric]) for metric, f in fs.items()
        }
        for metric_type, fs in objective_fs.items()
    }

def _empty_solution(
    ligand_evaluation_metrics:Iterable[str],
    target_evaluation_metrics:Iterable[str]
):
    return (
        {
            "weighted networks": None,
            "grn matrix": None,
            "ltf matrix": None,
            "ligand-target matrix": None
        },
        {
            "target_prediction": {e: 0 for e in target_evaluation_metrics},
            "ligand_prediction": {e: 0 for e in ligand_evaluation_metrics}
        }
    )

def construct_and_evaluate(
    source_weights:dict[str, float]|dict[int, float]|pd.DataFrame,
    lr_sig_hub:float,
    gr_hub:float,
    ltf_cutoff:float,
    damping_factor:float,
    lr_network:pd.DataFrame,
    gr_network:pd.DataFrame,
    sig_network:pd.DataFrame,
    evaluation_data:EvaluationData,
    return_all_matrices:bool=True,
    return_weighted_networks:bool=True,
    split_direct:str="no",
    direct_coef:float=0,
    ligand_evaluation_metrics:Iterable[str]=("aupr", "aupr_corrected", "auroc", "pearson"),
    target_evaluation_metrics:Iterable[str]=("aupr", "aupr_corrected", "auroc", "pearson")
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
    return_all_matrices : bool
        whether or not to return the ligand-tf and tf-target matrices
    return_weighted_networks : bool
        whether or not to return the weighted networks
    split_direct : str
        Whether or not to split the matrix into direct and indirect submatrices and take a weighted average. 
        "no": don't split;
        "ltf": split the ltf matrix;
        "tft": split the tft matrix;
        "ltf-tft": split both the ltf and tft matrices;
        Default: "no"
    direct_coef : float
        The strength of direct links during matrix construction, should be between 0 and 1, not used when split_direct == 'no'
        note: a weighted average is computed between the RP originating from direct links and the RP originating from indirect links
    ligand_evaluation_metrics : Iterable of string
            the ligand prediction evaluation metrics to compute, must be a subset of ("aupr", "aupr_corrected", "auroc", "pearson", "map", "ndcg")
    target_evaluation_metrics : Iterable of string
        the target prediction evaluation metrics to compute, must be a subset of ("aupr", "aupr_corrected", "auroc", "pearson", "map", "ndcg")

    Returns
    -------
    dict
        A dictionary with keys 'weighted networks', 'grn matrix', 'ltf matrix' and 'ligand-target matrix'
    dict
        output of `compute_evaluation_scores`
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if (
        type(source_weights) is dict and sum(source_weights.values()) == 0
    ) or (
        type(source_weights) is pd.DataFrame and sum(source_weights["weight"]) == 0
    ):
        return _empty_solution(
            ligand_evaluation_metrics,
            target_evaluation_metrics
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
        ligands=evaluation_data.get_ligands(),
        return_all_matrices=return_all_matrices,
        return_weighted_networks=return_weighted_networks,
        split_direct=split_direct,
        direct_coef=direct_coef
    )
    # make sure the ligand-target matrix is column-major, this will speed up the nichenet analysis which heavily relies on column indexing
    # the optimization as a whole is also faster despite the copy each trial
    ligand2target, row_names, col_names = model["ligand-target matrix"]
    if ligand2target.flags.c_contiguous:
        ligand2target = np.array(ligand2target, order="F")
    if np.sum(ligand2target) == 0:
        return _empty_solution(
            ligand_evaluation_metrics,
            target_evaluation_metrics
        )
    predictor = LigandActivityPredictor(ligand2target, row_names, col_names)
    predictor.replace_zero_col_by_noisy_scores()
    scores = compute_evaluation_scores(
        evaluate_model(
            predictor,
            evaluation_data,
            ligand_evaluation_metrics,
            target_evaluation_metrics
        ),
        evaluation_data.get_ligands(combination=True)
    )
    return (
        model,
        scores
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

def choose_pareto_optimal_solution_normalized(
    objective_values:Iterable[Iterable[float]],
    weights:Iterable[float]
) -> int:
    '''
    Choose one solution from a set of pareto optimal solutions using the weighted stress function method on normalized objectives

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
    objective_values = list(zip(*objective_values))
    for i in range(len(objective_values)):
        vals = np.array(objective_values[i])
        vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
        objective_values[i] = vals
    return choose_pareto_optimal_solution(zip(*objective_values), weights)
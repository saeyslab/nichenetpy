import warnings
from collections.abc import Callable, Iterable, Sequence
from itertools import chain
from traceback import format_exc
from typing import Any, Literal

import numpy as np
import pandas as pd

from nichenetpy.evaluation import EvaluationData, evaluate_single_importances_ligand_prediction
from nichenetpy.model_construction import construct_model_from_source_weights
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.typing import (
    MetricScoreFunction,
    ObjectiveFunction,
    eval_metric_t,
    gene_t,
    ligand_eval_metric_t,
    target_eval_metric_t,
)

# Can only add docstrings to single line variable declarations (note: *.__doc__ is read-only)
objective_fs_dct: dict[
    str, dict[Literal["target_prediction"] | Literal["ligand_prediction"], dict[eval_metric_t, ObjectiveFunction]]
] = None  # will be overwritten
"""
sets of objective functions for the source weight optimization

NNv2
    The objective functions used in NicheNetV2.
    Only rank ligands using `aupr_corrected` and `auroc`. Use `aupr_corrected` and `auroc` to evaluate the ligand ranking.

map&ndcg
    Only rank ligands using `aupr_corrected` and `auroc`. Use `map` and `ndcg` to evaluate the ligand ranking.
"""

metric_score_f_dct: dict[str, MetricScoreFunction] = None  # will be overwritten
"""
the function that computes the metric score (which decides the best metric to rank the ligands with)

NNv2
    The metric score function used in NicheNetV2.
    Computes the geometric mean of `aupr_corrected` and `auroc`.

all_geometric_mean
    The metric score function used in NicheNetV2.
    Computes the geometric mean of all available target evaluation metrics.
"""

objective_fs_dct = {
    "NNv2": {
        "target_prediction": {"aupr_corrected": np.mean, "auroc": np.mean},
        "ligand_prediction": {
            "aupr_corrected": lambda x: (np.mean(x) + np.median(x)) / 2,
            "auroc": lambda x: (np.mean(x) + np.median(x)) / 2,
        },
    },
    "map&ndcg": {
        "target_prediction": {"aupr_corrected": np.mean, "auroc": np.mean},
        "ligand_prediction": {
            "map": lambda x: (np.mean(x) + np.median(x)) / 2,
            "ndcg": lambda x: (np.mean(x) + np.median(x)) / 2,
        },
    },
}

metric_score_f_dct = {
    "NNv2": lambda aupr_corrected, auroc, **_: np.exp((np.log(aupr_corrected) + np.log(auroc)) / 2),  # type: ignore # is compatible with **kwargs
    "all_geometric_mean": lambda **mts: np.exp(np.mean([np.log(mt) for mt in mts.values()])),
}


def _average_performances[GeneSymbol](
    ligand_oi: GeneSymbol | Sequence[GeneSymbol],
    performances: pd.DataFrame,
    metrics: Iterable[eval_metric_t] = ("auroc", "aupr_corrected"),
) -> Iterable[float]:
    # true ligand must match with ligand(s) of interest
    performances_oi = performances[
        [
            any(
                ligand in true_ligand
                if isinstance(true_ligand, Sequence) and type(true_ligand) is not str
                else ligand == true_ligand
                for ligand in ((ligand_oi,) if isinstance(ligand_oi, gene_t.__value__) else ligand_oi)  # type: ignore # is correct due to conditional
            )
            for true_ligand in performances["ligand"]
        ]
    ]
    return tuple(performances_oi[metric].median() for metric in metrics)


def _evaluate_single_importances_ligand_prediction(
    ligand_importances: pd.DataFrame,
    group: str,
    ligand_evaluation_metrics: Iterable[ligand_eval_metric_t] = ("aupr", "aupr_corrected", "auroc", "pearson"),
    target_evaluation_metrics: Iterable[target_eval_metric_t] = ("aupr", "aupr_corrected", "auroc", "pearson"),
) -> pd.DataFrame | None:
    try:
        return evaluate_single_importances_ligand_prediction(
            ligand_importances,
            group,
            allow_nan=True,
            ligand_evaluation_metrics=ligand_evaluation_metrics,
            target_evaluation_metrics=target_evaluation_metrics,
        )
    except Exception as ex:
        warnings.warn(f"Could not evaluate ligand importance scores for {group}:\n{ex}\n{format_exc()}", stacklevel=2)
        return None


def evaluate_model[GeneSymbol](
    predictor: LigandActivityPredictor[GeneSymbol],
    evaluation_data: EvaluationData[GeneSymbol],
    ligand_evaluation_metrics: Iterable[ligand_eval_metric_t] = ("aupr", "aupr_corrected", "auroc", "pearson"),
    target_evaluation_metrics: Iterable[target_eval_metric_t] = ("aupr", "aupr_corrected", "auroc", "pearson"),
) -> dict[str, pd.DataFrame | None]:
    """
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
        the target prediction evaluation metrics to compute, must be a subset of ("aupr", "aupr_corrected", "auroc", "pearson")

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
    """
    if not isinstance(predictor, LigandActivityPredictor):
        raise TypeError(f"predictor should have type LigandActivityPredictor, was {type(predictor)}")
    if type(evaluation_data) is not EvaluationData:
        raise TypeError(f"settings should have type EvaluationData, was {type(evaluation_data)}")
    performances_target_prediction = {"setting": [], "ligand": []}
    for met in target_evaluation_metrics:
        performances_target_prediction[met] = []
    # the field names of the evaluation data (they used to be "from" and "response", whatever the data used)
    ligand_name, de_genes_name = evaluation_data.ligand_name, evaluation_data.de_genes_name
    evaluation_data = EvaluationData(
        (e[1] for e in evaluation_data.get_applicable_evaluation_datasets(predictor)),
        key_name=evaluation_data.key_name,
        ligand_name=ligand_name,
        de_genes_name=de_genes_name,
    )
    for setting_id, setting in evaluation_data.items():
        performances_target_prediction["setting"].append(setting_id)
        performances_target_prediction["ligand"].append(setting[ligand_name])
        for k, v in predictor.evaluate_target_prediction(
            (
                setting[ligand_name]  # type: ignore
                if isinstance(setting[ligand_name], gene_t.__value__)  # potential BUG when using integers
                else "-".join(setting[ligand_name])
            ),
            setting[de_genes_name],
            target_evaluation_metrics,
        ).items():
            performances_target_prediction[k].append(v)
    performances_target_prediction = pd.DataFrame(performances_target_prediction)
    all_ligands = evaluation_data.get_ligands(combination=False)
    ligand_importances = {"setting": [], "test_ligand": [], "true_ligand": []}
    for met in target_evaluation_metrics:
        ligand_importances[met] = []
    for setting_id, setting in evaluation_data.items():
        for ligand in all_ligands:
            ligand_importances["setting"].append(setting_id)
            ligand_importances["test_ligand"].append(ligand)
            ligand_importances["true_ligand"].append(setting[ligand_name])
            for k, v in predictor.evaluate_target_prediction(
                ligand, setting[de_genes_name], target_evaluation_metrics
            ).items():
                ligand_importances[k].append(v)
    ligand_importances = pd.DataFrame(ligand_importances)
    performances_ligand_prediction_single = [
        e
        for e in (
            _evaluate_single_importances_ligand_prediction(
                ligand_importances,  # type: ignore # false positive
                group=setting_id,
                ligand_evaluation_metrics=ligand_evaluation_metrics,
                target_evaluation_metrics=target_evaluation_metrics,
            )
            for setting_id in set(ligand_importances["setting"])
        )
        if e is not None
    ]
    if len(performances_ligand_prediction_single) > 0:
        performances_ligand_prediction_single = pd.concat(performances_ligand_prediction_single)
    else:
        performances_ligand_prediction_single = None
    return {
        "performances_target_prediction": performances_target_prediction,  # type: ignore
        "performances_ligand_prediction": performances_ligand_prediction_single,  # type: ignore
    }


def compute_evaluation_scores[GeneSymbol](
    eval_res: dict[str, pd.DataFrame | None],
    ligands: Iterable[GeneSymbol | Sequence[GeneSymbol]],
    metric_score_f: MetricScoreFunction | None = None,
    objective_fs: dict[
        Literal["target_prediction"] | Literal["ligand_prediction"], dict[eval_metric_t, ObjectiveFunction]
    ]
    | None = None,
) -> dict[Literal["target_prediction"] | Literal["ligand_prediction"], dict[eval_metric_t, float]]:
    """
    Construct and evaluate the ligand-target matrix.

    Parameters
    ----------
    eval_res : dict[str, pd.DataFrame]
        The output of a call to `nichenetpy.parameter_optimization.evaluate_model`
    ligands : Iterable of gene_t
        The ligands of interest
    metric_score_f : Callable or None
        Function that takes ligand prediction evaluation metrics as input and returns a score that can be used to rank target
        prediction evaluation metrics. Note that all possible metrics are passed as arguments to the function.
        (tip: use `**kwargs` to make sure you catch them)
        If None, `metric_score_f_dct["NNv2"]` is used.
    objective_fs : dict of dict[str, Callable] or None
        Dictionary which maps the keys "target_prediction" and "ligand_prediction" to dictionaries which map metrics to
        functions that aggregate values of said metric. These functions are used to compute the optimization objectives
        which are aggregated from metric values computed over multiple data sets.
        If None, `objective_fs_dct["NNv2"]` is used.

    Returns
    -------
    dict
        nested dictionary with keys "target_prediction" and "ligand_prediction", the nested dictionaries have metric names as keys

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    """
    # the defaults are looked up here instead of in the signature, so the signature (and its documentation) doesn't
    # contain the dictionaries of functions themselves (note: the default is still the shared module-level object)
    if metric_score_f is None:
        metric_score_f = metric_score_f_dct["NNv2"]
    if objective_fs is None:
        objective_fs = objective_fs_dct["NNv2"]
    if type(eval_res) is not dict:
        raise TypeError(f"eval_res should have type dict, was {type(eval_res)}")
    if eval_res["performances_target_prediction"] is None:
        raise ValueError("eval_res['performances_target_prediction'] was None")
    if not isinstance(ligands, Iterable):
        raise TypeError(f"ligands should have type Iterable, was {type(ligands)}")
    if not isinstance(metric_score_f, Callable):  # type: ignore
        raise TypeError(f"metric_score_f should have type Callable, was {type(metric_score_f)}")
    if type(objective_fs) is not dict:
        raise TypeError(f"objective_fs should have type dict, was {type(objective_fs)}")
    if (
        len(objective_fs) != 2
        or "target_prediction" not in objective_fs.keys()
        or "ligand_prediction" not in objective_fs.keys()
    ):
        raise KeyError(
            f"objective_fs should have exactly 'target_prediction' and 'ligand_prediction' as keys, got {objective_fs.keys()}"
        )
    target_evaluation_metrics: set[target_eval_metric_t] = set(eval_res["performances_target_prediction"].columns)  # type: ignore # error at runtime
    for e in ("setting", "ligand"):
        target_evaluation_metrics.remove(e)  # type: ignore
    performances_prediction_averaged = {}
    # median metric value per ligand for each metric
    performances_target_prediction_averaged: dict[target_eval_metric_t, Sequence[float]] = dict(
        zip(
            target_evaluation_metrics,
            zip(
                *(
                    _average_performances(
                        ligand,
                        eval_res["performances_target_prediction"],
                        target_evaluation_metrics,
                    )
                    for ligand in ligands
                ),
                strict=True,
            ),
            strict=True,
        )
    )
    performances_prediction_averaged["target_prediction"] = performances_target_prediction_averaged
    for e in performances_target_prediction_averaged.keys():
        performances_target_prediction_averaged[e] = [
            e for e in performances_target_prediction_averaged[e] if not np.isnan(e)
        ]
    if eval_res["performances_ligand_prediction"] is None:
        # ligand prediction could not be evaluated on any data set: only score target prediction
        return {
            "target_prediction": {
                metric: f(performances_target_prediction_averaged[metric])  # type: ignore # target prediction keys should be fine
                for metric, f in objective_fs["target_prediction"].items()
            },
            "ligand_prediction": dict.fromkeys(objective_fs["ligand_prediction"].keys(), 0),
        }
    ligand_evaluation_metrics = set(eval_res["performances_ligand_prediction"].columns)
    for e in ("metric", "group", "ligand"):
        ligand_evaluation_metrics.remove(e)
    ligand_activity_performance_setting_summary = (
        eval_res["performances_ligand_prediction"][list(chain(("metric",), ligand_evaluation_metrics))]
        .groupby("metric")
        .mean()
    )
    # compute a score for ligand prediction evaluation metrics (so we can rank them)
    ligand_activity_performance_setting_summary["score"] = [
        metric_score_f(
            **dict(zip(ligand_evaluation_metrics, mts, strict=True))
        )  # dictionary init takes ~1/6 as much time as geom avg? needs check...
        for mts in zip(
            *(ligand_activity_performance_setting_summary[met] for met in ligand_evaluation_metrics), strict=True
        )
    ]
    ligand_activity_performance_setting_summary.reset_index(inplace=True)
    # find the best target prediction evaluation metric by comparing the geometric average of each metric;
    # a NaN score (e.g. the log of a negative aupr_corrected) is never the best, unless all scores are NaN
    scores = ligand_activity_performance_setting_summary["score"].to_numpy(dtype=np.float64)
    best_metric = ligand_activity_performance_setting_summary["metric"].iloc[
        0 if np.isnan(scores).all() else int(np.nanargmax(scores))
    ]
    # only keep the best target prediction evaluation metric
    performances_ligand_prediction_summary = eval_res["performances_ligand_prediction"][
        eval_res["performances_ligand_prediction"]["metric"] == best_metric
    ]
    # median metric value for each metric per ligand
    performances_ligand_prediction_averaged = dict(
        zip(
            ligand_evaluation_metrics,
            zip(
                *(
                    _average_performances(
                        ligand,
                        performances_ligand_prediction_summary,
                        ligand_evaluation_metrics,  # type: ignore # error at runtime
                    )
                    for ligand in ligands
                ),
                strict=True,
            ),
            strict=True,
        )
    )
    performances_prediction_averaged["ligand_prediction"] = performances_ligand_prediction_averaged
    for e in performances_ligand_prediction_averaged.keys():
        performances_ligand_prediction_averaged[e] = [  # pyright: ignore[reportArgumentType] # target prediction keys should be fine
            e for e in performances_ligand_prediction_averaged[e] if not np.isnan(e)
        ]
    # aggregate metrics over ligands
    return {
        metric_type: {metric: f(performances_prediction_averaged[metric_type][metric]) for metric, f in fs.items()}  # type: ignore
        for metric_type, fs in objective_fs.items()
    }


def _empty_solution(
    ligand_evaluation_metrics: Iterable[ligand_eval_metric_t], target_evaluation_metrics: Iterable[target_eval_metric_t]
) -> tuple[
    dict[str, None], dict[Literal["target_prediction"] | Literal["ligand_prediction"], dict[eval_metric_t, float]]
]:
    return (
        {"weighted networks": None, "grn matrix": None, "ltf matrix": None, "ligand-target matrix": None},
        {
            "target_prediction": dict.fromkeys(target_evaluation_metrics, 0),
            "ligand_prediction": dict.fromkeys(ligand_evaluation_metrics, 0),
        },
    )


def construct_and_evaluate[GeneSymbol](
    source_weights: dict[GeneSymbol, float] | pd.DataFrame,
    lr_sig_hub: float,
    gr_hub: float,
    ltf_cutoff: float,
    damping_factor: float,
    lr_network: pd.DataFrame,
    gr_network: pd.DataFrame,
    sig_network: pd.DataFrame,
    evaluation_data: EvaluationData[GeneSymbol],
    return_all_matrices: bool = True,
    return_weighted_networks: bool = True,
    split_direct: Literal["no"] | Literal["ltf"] | Literal["tft"] | Literal["ltf-tft"] = "no",
    direct_coef: float = 0,
    ligand_evaluation_metrics: Iterable[ligand_eval_metric_t] = ("aupr", "aupr_corrected", "auroc", "pearson"),
    target_evaluation_metrics: Iterable[target_eval_metric_t] = ("aupr", "aupr_corrected", "auroc", "pearson"),
    metric_score_f: MetricScoreFunction | None = None,
    objective_fs: dict[
        Literal["target_prediction"] | Literal["ligand_prediction"], dict[eval_metric_t, ObjectiveFunction]
    ]
    | None = None,
    random_state: int | np.random.Generator | None = None,
) -> tuple[
    dict[str, Any], dict[Literal["target_prediction"] | Literal["ligand_prediction"], dict[eval_metric_t, float]]
]:
    """
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
    metric_score_f : Callable or None
        Function that takes ligand prediction evaluation metrics as input and returns a score that can be used to rank target
        prediction evaluation metrics. Note that all possible metrics are passed as arguments to the function.
        (tip: use `**kwargs` to make sure you catch them)
        If None, `metric_score_f_dct["NNv2"]` is used.
    objective_fs : dict of dict[str, Callable] or None
        Dictionary which maps the keys "target_prediction" and "ligand_prediction" to dictionaries which map metrics to
        functions that aggregate values of said metric. These functions are used to compute the optimization objectives
        which are aggregated from metric values computed over multiple data sets.
        If None, `objective_fs_dct["NNv2"]` is used.
    random_state : int or numpy.random.Generator or None
        seed or generator for the noise that replaces the all-zero columns of the ligand-target matrix before the
        evaluation (see `LigandActivityPredictor.replace_zero_col_by_noisy_scores`); with the same seed the scores are
        reproducible. If None, numpy's global random state is used.

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

    Notes
    -----
    If all source weights are 0, or all ligand-receptor and signaling sources have weight 0 (no ligand-signaling
    network), no model is constructed and all prediction scores are 0.
    """
    weights = (
        source_weights
        if type(source_weights) is dict
        else dict(zip(source_weights["source"].tolist(), source_weights["weight"].tolist(), strict=True))  # type: ignore
    )
    # A model without ligand-signaling links (all ligand-receptor and signaling sources weighted 0 or not given)
    # can't propagate signals from ligands to transcription factors, so it is scored as an empty model instead of
    # being evaluated. Besides saving the evaluation, this avoids the ValueError from construct_ligand_tf_matrix
    # when none of the evaluation ligands are present in the remaining (gene regulatory) network.
    lr_sig_sources = set(lr_network["source"].unique()).union(sig_network["source"].unique())
    if sum(weights.values()) == 0 or all(weights.get(source, 0) == 0 for source in lr_sig_sources):
        return _empty_solution(ligand_evaluation_metrics, target_evaluation_metrics)
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
        direct_coef=direct_coef,
    )
    # make sure the ligand-target matrix is column-major, this will speed up the nichenet analysis which heavily relies on column indexing
    # the optimization as a whole is also faster despite the copy each trial
    ligand2target, row_names, col_names = model["ligand-target matrix"]
    if ligand2target.flags.c_contiguous:
        ligand2target = np.array(ligand2target, order="F")
    if np.sum(ligand2target) == 0:
        return _empty_solution(ligand_evaluation_metrics, target_evaluation_metrics)
    predictor = LigandActivityPredictor(ligand2target, row_names, col_names)
    predictor.replace_zero_col_by_noisy_scores(random_state=random_state)
    scores = compute_evaluation_scores(
        evaluate_model(
            predictor,
            evaluation_data,
            ligand_evaluation_metrics=ligand_evaluation_metrics,
            target_evaluation_metrics=target_evaluation_metrics,
        ),
        evaluation_data.get_ligands(combination=True),
        metric_score_f=metric_score_f,
        objective_fs=objective_fs,
    )
    return (model, scores)


def weighted_stress_function(w: float, d1: float = 0.002, d2: float = 0.008) -> Callable[[float], float]:
    """
    Construct a weighted stress function

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
    """
    a = 0.75 * (1 - w) ** 2 + 2 * (1 - w) + d1
    b = a + 4 * w - 2
    c = 1 - np.tan(np.pi * (w - 0.5) / (1 + d2)) / np.tan(-np.pi / (2 * (1 + d2)))
    return lambda x: (
        (w / 2) * np.tan(-np.pi * (x - w) / b) + c
        if x <= w
        else c * (1 - np.tan(-np.pi * (x - w) / a) / np.tan(np.pi * (w - 1) / a))
    )


def choose_pareto_optimal_solution(objective_values: Iterable[Sequence[float]], weights: Iterable[float]) -> int:
    """
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
    """
    if not isinstance(objective_values, Iterable):
        raise TypeError(f"objective_values should be iterable, was {type(objective_values)}")
    if not isinstance(weights, Iterable):
        raise TypeError(f"weights should be iterable, was {type(weights)}")
    # a new array: the caller's weights are not changed (and may be ints or a generator)
    weights = np.asarray(list(weights), dtype=np.float64)
    weights = weights / weights.sum()
    fs = [weighted_stress_function(weight) for weight in weights]
    return int(
        np.argmin(
            [
                sum(np.abs(fs[i](xs[i]) - fs[j](xs[j])) for j in range(len(xs)) for i in range(j))
                for xs in objective_values
            ]
        )
    )


def choose_pareto_optimal_solution_normalized(
    objective_values: Iterable[Iterable[float]], weights: Iterable[float]
) -> int:
    """
    Choose one solution from a set of pareto optimal solutions using the weighted stress function method on normalized objectives
    Objectives without variance (the same value for every solution) are not taken into account.

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
    """
    # one array per objective (the values of all solutions for that objective)
    objectives = [np.asarray(vals, dtype=float) for vals in zip(*objective_values, strict=True)]
    weights = list(weights)
    # objectives without variance (the same value for every solution) are not taken into account, neither is their
    # weight: they can't distinguish between solutions, and min-max normalization would divide by zero for them;
    # the remaining weights are renormalized in choose_pareto_optimal_solution
    keep = [i for i, vals in enumerate(objectives) if vals.max() > vals.min()]
    if len(keep) == 0:
        return 0  # no objective has any variance: all solutions are equivalent
    normalized = [(objectives[i] - objectives[i].min()) / (objectives[i].max() - objectives[i].min()) for i in keep]
    return choose_pareto_optimal_solution(zip(*normalized, strict=True), [weights[i] for i in keep])

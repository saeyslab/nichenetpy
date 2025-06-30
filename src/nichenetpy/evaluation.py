from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.metrics import calculate_metrics
from nichenetpy.utils import is_ligand_active

from collections.abc import Iterable
from itertools import repeat

import warnings
import pandas as pd


def convert_expression_settings_evaluation(setting:dict) -> dict:
    '''
    Converts expression settings to correct settings format for evaluation of target gene prediction.

    Parameters
    ----------
    setting : dict
        A dictionary with the following keys:

            name: the name of the setting

            from: the name of the ligand which is active in the setting of interest

            diffexp:

                gene:

                lfc: (log fold change treated vs untreated)

                qval: (fdr-corrected p-value)

    Returns
    -------
    dict
        a dictionary with the following keys:

            name: the name of the setting

            from: the name of the ligand which is active in the setting of interest

            response:   a logical vector indicating whether the gene's transcription was
                        influenced by the active ligand(s) in the setting of interest
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(setting) is not dict:
        raise TypeError(f"setting should have type dict, was {type(setting)}")
    diffexp = setting["diffexp"]
    diffexp = [1 if (abs(lfc) >= 1 and qval <= 0.1) else 0 for lfc, qval in zip(diffexp["lfc"], diffexp["qval"])]
    if sum(diffexp) == 0:
        warnings.warn(f"{setting["name"]}: No differentially expressed genes, remove this expression dataset")
    return {
        "name": setting["name"],
        "from": setting["from"],
        "response": dict(zip(setting["diffexp"]["gene"], diffexp))
    }

def convert_settings_ligand_prediction(
    settings:dict,
    all_ligands:Iterable[str]
) -> list[dict]:
    '''
    Converts settings to correct settings format for ligand activity prediction.In this prediction problem,
    ligands (out of a set of possibly active ligands) will be ranked based on feature importance scores.
    The format can be made suited for: 1) validation of ligand activity state prediction by calculating individual
    feature importane scores or 2) feature importance based on models with embedded feature importance determination;
    applications in which ligands need to be scores based on their possible upstream activity:
    3) by calculating individual feature importane scores or 4) feature importance based on models with embedded feature
    importance determination.

    Parameters
    ----------
    settings : dict
        A dictionary who's values have the following keys: 
        
            name: the name of the setting
        
            from: the name of the ligand which is active in the setting of interest
        
            response:   the observed target response, indicates for a gene whether
                        it was a target or not in the setting of interest
    all_ligands : Iterable of str
        the possible ligands that will be considered for the ligand activity state prediction

    Returns
    -------
    dict
        a dictionary with the following keys:

            name: the name of the setting

            ligand: the active ligand

            from: the ligand that will be tested for activity prediction

            response:   a logical vector indicating whether the gene's transcription was influenced by
                        the active ligand(s) in the setting of interest
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(settings) is not dict:
        raise TypeError(f"settings should have type dict, was {type(settings)}")
    if not isinstance(all_ligands, Iterable):
        raise TypeError(f"all_ligands should have type Iterable[str], was {type(all_ligands)}")
    return [
        {
            "name": v["name"],
            "ligand": v["from"],
            "from": ligand,
            "response": v["response"]
        }
        for v in settings.values() for ligand in all_ligands
    ]

def get_single_ligand_importances(
    predictor:LigandActivityPredictor,
    settings:Iterable[dict]
) -> pd.DataFrame:
    '''
    Get ligand importance measures for ligands based on how well a single, individual, ligand can predict
    an observed response. Assess how well every ligand of interest is able to predict the observed transcriptional
    response in a particular dataset, according to the ligand-target model. It can be assumed that the ligand
    that best predicts the observed response, is more likely to be the true ligand.

    Parameters
    ----------
    settings : dict
        An Iterable of dictionaries that have the following keys: 
        
            name: the name of the setting

            ligand: the name of the ligand which is known to be active in the setting of interest
        
            from:  the name of the ligand of which the predictive performance need to be assessed
        
            response:   the observed target response, indicates for a gene whether it was a target
                        or not in the setting of interest
    predictor : LigandActivityPredictor
        the ligand activity predictor

    Returns
    -------
    pandas.DataFrame
        A data frame with for each ligand - data set combination, classification evaluation metrics indicating
        how well the query ligand predicts the response in the particular dataset. Evaluation metrics are
        the same as in evaluate_target_prediction. In addition to the metrics, the name of the particular setting,
        the name of the query ligand(test_ligand), the name of the true active ligand (if known: ligand).
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(predictor) is not LigandActivityPredictor:
        raise TypeError(f"predictor should have type LigandActivityPredictor, was {type(predictor)}")
    if not isinstance(settings, Iterable):
        raise TypeError(f"settings should have type Iterable[dict], was {type(settings)}")
    # compute metrics for multiple prediction/response pairs and store them in a dataframe
    ligand_importances = pd.DataFrame(
        dict(zip(
            ("aupr", "aupr_corrected", "auroc", "pearson"),
            zip(*(
                list(zip(*sorted(
                    predictor.evaluate_target_prediction(setting["from"], setting["response"]).items(),
                    key=lambda x : x[0]
                )))[1]
                for setting in settings
            ))
        ))
    )
    ligand_importances["setting"], ligand_importances["test_ligand"], ligand_importances["true_ligand"] = (
        zip(*((setting["name"], setting["from"], setting["ligand"]) for setting in settings))
    )
    return ligand_importances

def evaluate_single_importances_ligand_prediction(
    importances:pd.DataFrame,
    group:str
) -> pd.DataFrame:
    '''
    Evaluate how well a single ligand importance score is able to predict the true activity state of a ligand.
    For this it is assumed, that ligand importance measures for truely active ligands will be higher than for
    non-active ligands. Several classification evaluation metrics for the prediction are calculated and variable
    importance scores can be extracted to rank the different importance measures in order of importance for ligand
    activity state prediction.

    Parameters
    ----------
    importances : pandas.DataFrame
        A data frame containing at least folowing variables: setting, test_ligand, ligand and one or more feature importance scores.
        test_ligand denotes the name of a possibly active ligand, true_ligand the name of the truely active ligand.
    group : str
        the setting of interest

    Returns
    -------
    pandas.DataFrame
        A data frame containing classification evaluation metrics for the ligand activity state prediction.
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(importances) is not pd.DataFrame:
        raise TypeError(f"importances should have type pandas.DataFrame, was {type(importances)}")
    if type(group) is not str:
        raise TypeError(f"group should have type str, was {type(group)}")
    importances = importances[importances["setting"] == group]
    metrics = ("aupr", "aupr_corrected", "auroc", "pearson")
    added = is_ligand_active(importances)
    # compute metrics for multiple prediction/response pairs and store them in a dataframe
    output = pd.DataFrame(
        dict(zip(
            metrics,
            zip(*(
                list(zip(*sorted(
                    calculate_metrics(list(importances[metric]), added).items(),
                    key=lambda x : x[0]
                )))[1]
                for metric in metrics
            ))
        ))
    )
    output["metric"] = metrics
    output["group"] = list(repeat(group, len(metrics)))
    output["ligand"] = list(repeat(importances["true_ligand"].iloc[1], len(metrics)))
    return output
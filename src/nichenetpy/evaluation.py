from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.metrics import calculate_metrics

from collections.abc import Iterable

import warnings
import pandas as pd


def convert_expression_settings_evaluation(setting:dict):
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
):
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
    settings:dict
):
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
    '''ligand_importances["test_ligand"] = [setting["from"] for setting in settings]
    ligand_importances["true_ligand"] = [setting["ligand"] for setting in settings]
    ligand_importances["setting"] = [setting["name"] for setting in settings]'''
    ligand_importances["setting"], ligand_importances["test_ligand"], ligand_importances["true_ligand"] = (
        zip(*((setting["name"], setting["from"], setting["ligand"]) for setting in settings))
    )
    return ligand_importances

def evaluate_single_importances_ligand_prediction(
    importances:pd.DataFrame
):
    added = importances["test_ligand"], importances["test_ligand"] == importances["true_ligand"]
    # compute metrics for multiple prediction/response pairs and store them in a dataframe
    return pd.DataFrame(
        dict(zip(
            ("aupr", "aupr_corrected", "auroc", "pearson"),
            zip(*(
                list(zip(*sorted(
                    calculate_metrics(list(pred), list(added)).items(),
                    key=lambda x : x[0]
                )))[1]
                for pred in (
                    importances["aupr"],
                    importances["aupr_corrected"],
                    importances["auroc"],
                    importances["pearson"]
                )
            ))
        ))
    )
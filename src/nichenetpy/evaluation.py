from nichenetpy.prediction import LigandActivityPredictor

from collections.abc import Iterable

import warnings


def convert_expression_settings_evaluation(setting:dict):
    diffexp = setting["diffexp"]
    diffexp = [1 if (lfc >= 1 and qval <= 0.1) else 0 for lfc, qval in zip(diffexp["lfc"], diffexp["qval"])]
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
    return {
        k: {
            "name": v["name"],
            "ligand": v["from"],
            "from": ligand,
            "response": v["response"]
        }
        for k, v in settings.items() for ligand in all_ligands
    }
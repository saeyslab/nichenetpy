# type: ignore

"""
Results whose order came from iterating a set of strings (which changes between Python processes, with the hash seed)
are in a reproducible order: the order of first appearance, of an argument, or sorted.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from anndata import AnnData

from nichenetpy.evaluation import EvaluationData
from nichenetpy.extraction import average_expression
from nichenetpy.model_construction import construct_model_from_source_weights
from nichenetpy.prediction import LigandActivityPredictor, assess_rf_class_probabilities
from nichenetpy.prioritization import get_avg_exp
from nichenetpy.utils import combine_dicts
from nichenetpy.visualization import assign_ligands_to_celltype

_CELLTYPES = ["T", "B", "NK", "Mono", "DC", "B", "T", "Mono", "NK", "DC"]
_GENES = ["L3", "L1", "L4", "L2", "g0"]


def _ann():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 10, size=(len(_CELLTYPES), len(_GENES))).astype(float)
    # each of L3, L1 and L4 is clearly highest in one cell type, L2 in two (a general ligand)
    counts[[0, 6], 0] = 100
    counts[[1, 5], 1] = 100
    counts[[2, 8], 2] = 100
    counts[[3, 7, 4, 9], 3] = 100
    obs = pd.DataFrame({"celltype": _CELLTYPES}, index=[f"c{i}" for i in range(len(_CELLTYPES))])
    ann = AnnData(X=counts, obs=obs, var=pd.DataFrame(index=_GENES))
    ann.layers["counts"] = counts
    ann.layers["data"] = np.log1p(counts)
    return ann


def _model_ligands():
    lr = pd.DataFrame({"from": ["L3", "L1", "L4", "L2", "L1"], "to": ["R1", "R2", "R1", "R2", "R1"], "source": "lr"})
    sig = pd.DataFrame({"from": ["R1", "R2"], "to": ["T1", "T2"], "source": "sig"})
    gr = pd.DataFrame({"from": ["T1", "T2"], "to": ["g0", "g1"], "source": "gr"})
    model = construct_model_from_source_weights({"lr": 1, "sig": 1, "gr": 1}, 0, 0, 0, 0.5, lr, gr, sig)
    return list(model["ligand-target matrix"][2])


def _rf_predictions():
    # a set of ligands: the order of the random forest's features
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(40)]
    ligands = [f"L{i}" for i in range(8)]
    predictor = LigandActivityPredictor(np.asfortranarray(rng.random((len(genes), len(ligands)))), genes, ligands)
    res = assess_rf_class_probabilities(
        2, set(genes[:15]), set(genes[15:]), set(ligands), predictor, 20, random_state=0
    )
    return res.sort_values("gene")["prediction"].round(10).tolist()


def _orders():
    ann = _ann()
    evaluation = EvaluationData([{"name": "s1", "from": "L1", "response": {"g0": 1}}])
    return {
        "construct_model_from_source_weights": _model_ligands(),
        "EvaluationData.to_dataframe": evaluation.to_dataframe().columns.tolist(),
        "combine_dicts": list(combine_dicts(dict.fromkeys("zyxwvu", 1), dict.fromkeys("uvwxyz", 2))),
        "average_expression": average_expression(ann, "celltype", layer="counts").columns.tolist(),
        "average_expression keys": average_expression(
            ann, "celltype", keys=["NK", "Z", "T", "B"], layer="counts"
        ).columns.tolist(),
        "get_avg_exp": get_avg_exp(ann, "celltype")["celltype"].unique().tolist(),
        "get_avg_exp features": get_avg_exp(ann, "celltype", features={"L3", "g0", "L1"})["gene"].unique().tolist(),
        "assess_rf_class_probabilities": _rf_predictions(),
        "assign_ligands_to_celltype": assign_ligands_to_celltype(ann, ["L2", "L4", "L1", "L3"]).values.tolist(),
    }


def test_orders():
    orders = _orders()
    assert orders["construct_model_from_source_weights"] == ["L3", "L1", "L4", "L2"]
    assert orders["EvaluationData.to_dataframe"] == ["name", "from", "response"]
    assert orders["combine_dicts"] == list("zyxwvu")
    assert orders["average_expression"] == ["T", "B", "NK", "Mono", "DC"]
    assert orders["average_expression keys"] == ["NK", "T", "B"]
    assert orders["get_avg_exp"] == ["T", "B", "NK", "Mono", "DC"]
    # a set of features is sorted
    assert orders["get_avg_exp features"] == ["L1", "L3", "g0"]
    # cell types sorted, the ligands (general ligands last) in the order of ligands
    assert orders["assign_ligands_to_celltype"] == [["B", "L1"], ["NK", "L4"], ["T", "L3"], ["General", "L2"]]


def test_orders_independent_of_hash_seed():
    # the same orders in Python processes with a different hash seed (a set of strings has another order in each)
    code = "import json, deterministic_order_test as t; print(json.dumps(t._orders()))"
    outputs = []
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
        )
        outputs.append(json.loads(res.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1] == json.loads(json.dumps(_orders()))

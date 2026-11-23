import argparse
import os
import pickle
from functools import reduce
from operator import and_

import numpy as np
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from pareto import Pareto, ParetoSet
from scipy.sparse import csr_matrix

from nichenetpy.model_construction import (
    apply_hub_correction,
    construct_ligand_target_matrix,
    construct_weighted_networks,
)
from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork
from nichenetpy.parameter_optimization import choose_pareto_optimal_solution_normalized
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.utils import read_csv_cols, read_csv_rows, read_network_file

HYPERPARAMETERS = ("lr_sig_hub", "gr_hub", "ltf_cutoff", "damping_factor")


def fixed_source_names(sources, source_annotations, var_database, excluded_database, included_database):
    # the sources whose weights weren't optimized, selected as in run_optimization_v2.py
    df = pd.DataFrame({"source": sorted(sources)}).merge(source_annotations, on="source", how="inner")
    databases = df["database"].tolist()
    source_names = set(df["source"].tolist())
    if len(excluded_database) > 0:
        source_names = set(df[[db not in excluded_database for db in databases]]["source"])
    elif len(included_database) > 0:
        source_names = set(df[[db in included_database for db in databases]]["source"])
    bool_v = reduce(
        and_,
        (df["database"] != db for db in var_database),
        np.array([e in source_names for e in df["source"].tolist()])
        if len(excluded_database) > 0 or len(included_database) > 0
        else np.ones(df.shape[0], dtype=bool),
    )
    return set(df[bool_v]["source"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="constructs a nichenet model from the optimization logs (optuna)")
    parser.add_argument("log_path", help="path to the log file")
    parser.add_argument("--lr_network", help="path to the ligand receptor network file", action="append", default=[])
    parser.add_argument("--gr_network", help="path to the gene regulatory network file", action="append", default=[])
    parser.add_argument("--sig_network", help="path to the signaling network file", action="append", default=[])
    parser.add_argument(
        "--model_path", help="path to the file where the model should be saved", action="append", default=[]
    )
    parser.add_argument(
        "--split_direct",
        help="""
            "Whether or not to split the matrix into direct and indirect submatrices and take a weighted average.
            "no": don't split; "ltf": split the ltf matrix; "tft": split the tft matrix; "ltf-tft":
            split both the ltf and tft matrices; Default: "no""
        """,
        choices=("no", "ltf", "tft", "ltf-tft"),
        default="no",
    )
    # the values that were fixed during the optimization (the same arguments as run_optimization_v2.py), they're only
    # used for the parameters that aren't in the chosen trial
    for name in HYPERPARAMETERS:
        parser.add_argument(
            f"--{name}", help=f"the value of {name} if it was fixed during the optimization", type=float
        )
    parser.add_argument(
        "--direct_coef",
        help="the value of direct_coef if it was fixed during the optimization (it is when split_direct is 'no')",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--semantic_similarity_metric",
        help="the semantic similarity metric that filtered the signaling network during the optimization (ss_cutoff)",
        default=None,
    )
    parser.add_argument(
        "--source_path",
        help="the directory with optimized_source_weights.csv and annotation_data_sources.csv, for the source weights "
        "that were fixed during the optimization (--var_database)",
        default=None,
    )
    parser.add_argument("--var_database", help="as in run_optimization_v2.py", action="append", default=[])
    parser.add_argument("--excluded_database", help="as in run_optimization_v2.py", action="append", default=[])
    parser.add_argument("--included_database", help="as in run_optimization_v2.py", action="append", default=[])
    args = parser.parse_args()
    lock_obj = JournalFileOpenLock(args.log_path)
    storage = JournalStorage(JournalFileBackend(args.log_path, lock_obj))
    study = storage.get_all_studies()[0]
    # some trials in the log were never completed due to the time or memory limit being exceeded
    trials = [trial for trial in storage.get_all_trials(study._study_id) if trial.values is not None]
    pareto_set = ParetoSet()
    for trial in trials:
        pareto_set.add((Pareto(tuple(trial.values)), trial.params))
    pareto_set = list(pareto_set)
    pws = (1, 1, 1, 1)
    chosen_solution = pareto_set[choose_pareto_optimal_solution_normalized((objs for objs, _ in pareto_set), pws)][1]
    # separate the hyperparameters from the source weights, a parameter that isn't in the trial was fixed
    params = dict(chosen_solution)
    hyperparameters = {}
    for name in HYPERPARAMETERS:
        hyperparameters[name] = params.pop(name, getattr(args, name))
        if hyperparameters[name] is None:
            raise ValueError(f"{name} isn't in the chosen trial, pass the value of the optimization with --{name}")
    direct_coef = params.pop("direct_coef", args.direct_coef)
    ss_cutoff = params.pop("ss_cutoff", None)
    if ss_cutoff is not None and args.semantic_similarity_metric is None:
        raise ValueError(
            "the signaling network was filtered on semantic similarity (ss_cutoff) during the optimization, "
            "pass the metric with --semantic_similarity_metric"
        )
    optimized_source_weights = source_annotations = None
    if args.source_path is not None and len(args.var_database) > 0:
        weights = tuple(
            zip(*read_csv_rows(os.path.join(args.source_path, "optimized_source_weights.csv"))[1], strict=True)
        )
        optimized_source_weights = dict(zip(weights[0], (float(e) for e in weights[1]), strict=True))
        source_annotations = pd.DataFrame(read_csv_cols(os.path.join(args.source_path, "annotation_data_sources.csv")))
    # build the model
    for lr, gr, sig, out in zip(args.lr_network, args.gr_network, args.sig_network, args.model_path, strict=True):
        lr_network = read_network_file(lr)
        gr_network = read_network_file(gr)
        sig_network = read_network_file(sig)
        source_weights = dict(params)
        if optimized_source_weights is not None:
            sources = set(gr_network["source"].tolist()).union(
                lr_network["source"].tolist(), sig_network["source"].tolist()
            )
            for source in fixed_source_names(
                sources, source_annotations, args.var_database, args.excluded_database, args.included_database
            ):
                if source in optimized_source_weights:
                    source_weights[source] = optimized_source_weights[source]
        if ss_cutoff is not None:
            metric = args.semantic_similarity_metric
            similarity = np.array([0 if e == "" else float(e) for e in sig_network[metric].tolist()])
            sig_network = sig_network[similarity >= ss_cutoff]
        sig_network = sig_network[["from", "to", "source"]]
        weighted_networks = construct_weighted_networks(lr_network, sig_network, gr_network, source_weights)
        weighted_networks["lr_sig"] = apply_hub_correction(
            weighted_networks["lr_sig"], hub=hyperparameters["lr_sig_hub"]
        )
        weighted_networks["gr"] = apply_hub_correction(weighted_networks["gr"], hub=hyperparameters["gr_hub"])
        lt_matrix, ltf_matrix, grn_matrix = construct_ligand_target_matrix(
            weighted_networks,
            lr_network,
            sorted(set(lr_network["from"].tolist())),
            damping_factor=hyperparameters["damping_factor"],
            ltf_cutoff=hyperparameters["ltf_cutoff"],
            return_all_matrices=True,
            split_direct=args.split_direct,
            direct_coef=direct_coef,
        )
        mat, row_names, col_names = lt_matrix
        # convert to column-major memory layout
        predictor = LigandActivityPredictor(np.array(mat, order="F"), row_names, col_names)
        ltf_matrix = (csr_matrix(ltf_matrix[0]), ltf_matrix[1], ltf_matrix[2])
        out_dir = os.path.split(out)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out, "wb") as file:
            file.write(
                pickle.dumps(
                    {
                        "predictor": predictor,
                        "lr_network": LigandReceptorNetwork(lr_network),
                        "lr_sig": WeightedNetwork(weighted_networks["lr_sig"]),
                        "gr": WeightedNetwork(weighted_networks["gr"]),
                        "ltf_matrix": ltf_matrix,
                        "grn_matrix": grn_matrix,
                        "source_weights": source_weights,
                    }
                )
            )

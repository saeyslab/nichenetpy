from nichenetpy.utils import (
    read_csv_cols,
)
from nichenetpy.model_construction import (
    construct_weighted_networks,
    apply_hub_correction,
    construct_ligand_target_matrix
)
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.network import LigandReceptorNetwork, WeightedNetwork
from nichenetpy.model import NicheNet

from optuna.storages import JournalStorage
from optuna.storages.journal import (
    JournalFileBackend,
    JournalFileOpenLock
)
from itertools import chain
from scipy.sparse import csr_matrix

import pandas as pd
import numpy as np
import os
import pickle
import argparse

from pareto import Pareto, ParetoSet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="constructs a nichenet model from the optimization logs (optuna)"
    )
    parser.add_argument(
        "log_dir",
        help="path to the directory which contains the logs"
    )
    parser.add_argument(
        "--lr_network",
        help="path to the ligand receptor network file",
        action="append",
        default=[]
    )
    parser.add_argument(
        "--gr_network",
        help="path to the gene regulatory network file",
        action="append",
        default=[]
    )
    parser.add_argument(
        "--sig_network",
        help="path to the signaling network file",
        action="append",
        default=[]
    )
    parser.add_argument(
        "--model_path",
        help="path to the file where the model should be saved",
        action="append",
        default=[]
    )
    parser.add_argument(
        "--aggregation_method",
        help="average or median, default: average",
        type=str,
        choices=("average", "median"),
        default="average"
    )
    args = parser.parse_args()
    pareto_set = dict()
    for log_file in os.listdir(args.log_dir):
        log_file = os.path.join(args.log_dir, log_file)
        lock_obj = JournalFileOpenLock(log_file)
        storage = JournalStorage(
            JournalFileBackend(log_file, lock_obj)
        )
        study = storage.get_all_studies()[0]
        # some trials in the log were never completed due to the time or memory limit being exceeded
        trials = [trial for trial in storage.get_all_trials(study._study_id) if trial.values is not None]
        id = os.path.split(log_file)[1]
        pareto_set[id] = ParetoSet()
        for trial in trials:
            pareto_set[id].add((Pareto(tuple(trial.values)), trial.params))
    ranked_solutions = list(chain(*(
        sorted(
            ((np.exp(np.mean(np.log(np.array(scores)))), scores, sol) for scores, sol in ps),
            key=lambda x : x[0],
            reverse=True
        )[:25]
        for ps in pareto_set.values()
    )))
    # make sure each source is present in all solutions
    present_sources = set(ranked_solutions[0][2].keys()).union(*(sw.keys() for _, _, sw in ranked_solutions[1:]))
    for _, _, sw in ranked_solutions:
        for source in present_sources:
            if source not in sw:
                sw[source] = 0
    # aggregate the solutions
    keys = sorted(ranked_solutions[0][2].keys())
    aggregated_solution = (
        dict(
            zip(
                keys,
                (# aggregation function
                    np.median if args.aggregation_method == "median" else np.mean
                )(# aggregation function arguments
                    [list(zip(*sorted(sol.items(), key=lambda x : x[0])))[1] for _, _, sol in ranked_solutions],
                    axis=0
                )
            )
        )
    )
    # separate the hyperparameters from the source weights
    lr_sig_hub = aggregated_solution.pop("lr_sig_hub")
    gr_hub = aggregated_solution.pop("gr_hub")
    ltf_cutoff = aggregated_solution.pop("ltf_cutoff")
    damping_factor = aggregated_solution.pop("damping_factor")
    # build the model
    for lr, gr, sig, out in zip(args.lr_network, args.gr_network, args.sig_network, args.model_path):
        lr_network = pd.DataFrame(read_csv_cols(lr))
        gr_network = pd.DataFrame(read_csv_cols(gr))
        sig_network = pd.DataFrame(read_csv_cols(sig))
        weighted_networks = construct_weighted_networks(
            lr_network,
            sig_network,
            gr_network,
            aggregated_solution
        )
        weighted_networks["lr_sig"] = apply_hub_correction(weighted_networks["lr_sig"], hub=lr_sig_hub)
        weighted_networks["gr"] = apply_hub_correction(weighted_networks["gr"], hub=gr_hub)
        lt_matrix, ltf_matrix, grn_matrix = construct_ligand_target_matrix(
            weighted_networks,
            lr_network,
            set(lr_network["from"]),
            damping_factor=damping_factor,
            ltf_cutoff=ltf_cutoff,
            return_all_matrices=True
        )
        mat, row_names, col_names = lt_matrix
        # convert to column-major memory layout
        predictor = LigandActivityPredictor(
            np.array(mat, order="F"),
            row_names,
            col_names
        )
        ltf_matrix = (csr_matrix(ltf_matrix[0]), ltf_matrix[1], ltf_matrix[2])
        out_dir = os.path.split(out)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out, "wb") as file:
            file.write(pickle.dumps({
                "predictor": predictor,
                "lr_network": LigandReceptorNetwork(lr_network),
                "lr_sig": WeightedNetwork(weighted_networks["lr_sig"]),
                "gr": WeightedNetwork(weighted_networks["gr"]),
                "ltf_matrix": ltf_matrix,
                "grn_matrix": grn_matrix,
                "source_weights": aggregated_solution
            }))
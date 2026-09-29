from nichenetpy.utils import (
    read_network_file
)
from nichenetpy.model_construction import (
    construct_weighted_networks,
    apply_hub_correction,
    construct_ligand_target_matrix
)
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.network import (
    LigandReceptorNetwork,
    WeightedNetwork
)
from nichenetpy.parameter_optimization import choose_pareto_optimal_solution_normalized

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
        "log_path",
        help="path to the log file"
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
        "--split_direct",
        help=
        '''
            "Whether or not to split the matrix into direct and indirect submatrices and take a weighted average.
            "no": don't split; "ltf": split the ltf matrix; "tft": split the tft matrix; "ltf-tft":
            split both the ltf and tft matrices; Default: "no""
        ''',
        default="no"
    )
    args = parser.parse_args()
    lock_obj = JournalFileOpenLock(args.log_path)
    storage = JournalStorage(
        JournalFileBackend(args.log_path, lock_obj)
    )
    study = storage.get_all_studies()[0]
    # some trials in the log were never completed due to the time or memory limit being exceeded
    trials = [trial for trial in storage.get_all_trials(study._study_id) if trial.values is not None]
    pareto_set = ParetoSet()
    for trial in trials:
        pareto_set.add((Pareto(tuple(trial.values)), trial.params))
    pareto_set = list(pareto_set)
    pws = (1, 1, 1, 1)
    chosen_solution = pareto_set[choose_pareto_optimal_solution_normalized((objs for objs, _ in pareto_set), pws)][1]
    # separate the hyperparameters from the source weights
    lr_sig_hub = chosen_solution.pop("lr_sig_hub")
    gr_hub = chosen_solution.pop("gr_hub")
    ltf_cutoff = chosen_solution.pop("ltf_cutoff")
    damping_factor = chosen_solution.pop("damping_factor")
    # build the model
    for lr, gr, sig, out in zip(args.lr_network, args.gr_network, args.sig_network, args.model_path):
        lr_network = read_network_file(lr)
        gr_network = read_network_file(gr)
        sig_network = read_network_file(sig)
        weighted_networks = construct_weighted_networks(
            lr_network,
            sig_network,
            gr_network,
            chosen_solution
        )
        weighted_networks["lr_sig"] = apply_hub_correction(weighted_networks["lr_sig"], hub=lr_sig_hub)
        weighted_networks["gr"] = apply_hub_correction(weighted_networks["gr"], hub=gr_hub)
        lt_matrix, ltf_matrix, grn_matrix = construct_ligand_target_matrix(
            weighted_networks,
            lr_network,
            set(lr_network["from"]),
            damping_factor=damping_factor,
            ltf_cutoff=ltf_cutoff,
            return_all_matrices=True,
            split_direct=args.split_direct,
            direct_coef=chosen_solution.pop("direct_coef")
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
                "source_weights": chosen_solution
            }))
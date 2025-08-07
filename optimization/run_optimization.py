from nichenetpy.utils import (
    read_csv_cols
)
from nichenetpy.parameter_optimization import (
    construct_and_evaluate
)

from optuna import (
    create_study,
    load_study,
    Study
)
from optuna.trial import Trial
from optuna.samplers import (
    TPESampler,
    NSGAIISampler
)
from optuna.storages import JournalStorage
from optuna.storages.journal import (
    JournalFileBackend,
    JournalFileOpenLock
)
from optuna.samplers.nsgaii import (
    BaseCrossover
)
from itertools import chain
from pickle import dumps
from joblib import Parallel, delayed
from time import time

import pandas as pd
import json
import numpy as np
import argparse
import os


class FlatCrossover(BaseCrossover):
    n_parents = 2

    def crossover(
        self,
        parents_params:np.ndarray,
        rng:np.random.RandomState,
        study:Study,
        search_space_bounds:np.ndarray
    ):
        n_params = parents_params.shape[1]
        return parents_params[0, :] + rng.rand(n_params) * (parents_params[1, :] - parents_params[0, :])


@delayed
def optimize(study_name, storage, sampler):
    load_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler
    ).optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=1
    )

if __name__ == "__main__":
    t = time()
    parser = argparse.ArgumentParser(
        description="optimize the source weights and hyperparameters"
    )
    parser.add_argument(
        "out_file",
        help="path to the output file"
    )
    parser.add_argument(
        "gr_network_file",
        help="path to the gr_network file"
    )
    parser.add_argument(
        "lr_network_file",
        help="path to the lr_network file"
    )
    parser.add_argument(
        "sig_network_file",
        help="path to the sig_network file"
    )
    parser.add_argument(
        "--settings_file",
        help="path to the settings file for training",
        action="append",
        default=[]
    )
    parser.add_argument(
        "--n_trials",
        help="the amount of times to create and evaluate a model per process",
        type=int,
        default=50
    )
    parser.add_argument(
        "--n_process",
        help="the amount of processes",
        type=int,
        default=16
    )
    parser.add_argument(
        "-c",
        help="continue running the optimization from already existing log files",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--algorithm",
        help="the optimization algorithm to use",
        type=str,
        choices=("TPE", "NSGA-II"),
        default="TPE"
    )
    args = parser.parse_args()
    if len(args.lr_network_file) == 0:
        raise ValueError("at least one settings file needs to be provided")
    gr_network = pd.DataFrame(read_csv_cols(args.gr_network_file))
    lr_network = pd.DataFrame(read_csv_cols(args.lr_network_file))
    sig_network = pd.DataFrame(read_csv_cols(args.sig_network_file))
    parallel = Parallel(n_jobs=-1)
    optimal_parameters = dict()
    for settings_file in args.settings_file:
        with open(settings_file, "rb") as file:
            settings_CV = json.loads(file.read())
        settings = settings_CV["settings"]
        gr_network = gr_network[
            ~ (
                (gr_network["database"] == "NicheNet_LT") &
                np.array([fr not in settings_CV["forbidden_ligands_nichenet"] for fr in gr_network["from"]])
            )
            &
            ~ (
                (gr_network["database"] == "CytoSig") &
                np.array([fr not in settings_CV["forbidden_ligands_cytosig"] for fr in gr_network["from"]])
            )
        ]
        source_names = sorted(set(chain(gr_network["source"], lr_network["source"], sig_network["source"])))

        def objective(trial:Trial):
            source_weights = dict(
                (
                    source_name,
                    trial.suggest_float(
                        name=source_name,
                        low=0,
                        high=1
                    )
                ) for source_name in source_names
            )
            lr_sig_hub = trial.suggest_float(
                name="lr_sig_hub",
                low=0,
                high=1
            )
            gr_hub = trial.suggest_float(
                name="gr_hub",
                low=0,
                high=1
            )
            ltf_cutoff = trial.suggest_float(
                name="ltf_cutoff",
                low=0.9,
                high=0.999
            )
            damping_factor = trial.suggest_float(
                name="damping_factor",
                low=0.01,
                high=0.99
            )
            res = construct_and_evaluate(
                source_weights,
                lr_sig_hub,
                gr_hub,
                ltf_cutoff,
                damping_factor,
                lr_network,
                gr_network,
                sig_network,
                settings
            )
            return (res[1], res[2])

        name = settings_file.split("/")[-1][:-5]
        if not os.path.exists("log"):
            os.mkdir("log")
        log_file = f"./log/{name}_{args.algorithm}.log"
        with open(log_file, "a" if args.c else "w"):
            pass
        lock_obj = JournalFileOpenLock(log_file)
        storage = JournalStorage(
            JournalFileBackend(log_file, lock_obj)
        )
        if args.algorithm == "TPE":
            sampler = TPESampler()
        elif args.algorithm == "NSGA-II":
            sampler = NSGAIISampler(
                crossover=FlatCrossover(),
                crossover_prob=1
            )
        else:
            raise ValueError(f"{args.algorithm} is not a supported optimization algorithm, supported algorithms are 'TPE' and 'NSGA-II'")
        study = create_study(
            sampler=sampler,
            directions=["maximize", "maximize"],
            study_name=name,
            storage=storage,
            load_if_exists=args.c
        )
        parallel(optimize(name, storage, sampler) for _ in range(args.n_process))
        optimal_parameters[name] = [trial.params for trial in study.best_trials]
    with open(args.out_file, "wb") as file:
        file.write(dumps(optimal_parameters))
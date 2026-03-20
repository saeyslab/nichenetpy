from nichenetpy.utils import (
    read_csv_cols,
    read_csv_rows,
    read_network_file
)
from nichenetpy.parameter_optimization import (
    construct_and_evaluate
)
from nichenetpy.evaluation import EvaluationData
from nichenetpy.typing import gene_t

from optuna import (
    create_study,
    load_study,
    Study
)
from optuna.trial import Trial
from optuna.samplers import (
    TPESampler,
    NSGAIISampler,
    GPSampler
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
from joblib import (
    Parallel,
    delayed,
    cpu_count
)
from functools import reduce
from operator import and_

import pandas as pd
import json
import numpy as np
import argparse
import os
import requests
import pickle


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
    parser = argparse.ArgumentParser(
        description="optimize the source weights and hyperparameters"
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
        "settings_file",
        help="path to the settings file for training",
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
        default=-1
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
        choices=("TPE", "NSGA-II", "GP"),
        default="TPE"
    )
    parser.add_argument(
        "--source_path",
        help="the path to the directory where the source files (optimized_source_weights.csv and annotation_data_sources.csv) should be stored",
        type=str,
        default=None
    )
    parser.add_argument(
        "--lr_sig_hub",
        help="a number between 0 (no correction for hubiness) and 1 (maximal correction for hubiness)",
        type=float,
        default=None
    )
    parser.add_argument(
        "--gr_hub",
        help="a number between 0 (no correction for hubiness) and 1 (maximal correction for hubiness)",
        type=float,
        default=None
    )
    parser.add_argument(
        "--ltf_cutoff",
        help="ligand-tf scores beneath the 'ltf_cutoff' quantile will be set to 0.",
        type=float,
        default=None
    )
    parser.add_argument(
        "--damping_factor",
        help="In the PPR algorithm, the damping factor is the probability that the random walker will continue its walk on the graph; 1 - damping factor is the probability that the walker will return to the seed node.",
        type=float,
        default=None
    )
    parser.add_argument(
        "--var_database",
        help="Databases that have their source weights optimized. Ignored if source_path is not provided. By default all databases have their source weights optimized.",
        action="append",
        default=[]
    )
    parser.add_argument(
        "--excluded_database",
        help="databases to exclude from the optimization",
        action="append",
        default=[]
    )
    parser.add_argument(
        "--included_database",
        help="databases to include in the optimization",
        action="append",
        default=[]
    )
    parser.add_argument(
        "--excluded_source",
        help="sources to exclude from the optimization",
        action="append",
        default=[]
    )
    parser.add_argument(
        "--included_source",
        help="sources to include in the optimization",
        action="append",
        default=[]
    )
    parser.add_argument(
        "--log_dir",
        help="Directory in which to store the log of the optimization run. ",
        default="./log"
    )
    parser.add_argument(
        "--id",
        help="Identifier of the log",
        default=""
    )
    args = parser.parse_args()
    if len(args.included_database) > 0 and len(args.excluded_database) > 0:
        raise ValueError("included_database and excluded_database are incompatible with eachother")
    source_path = os.path.normpath("./source_files/")
    if args.source_path is not None:
        if not os.path.exists(args.source_path):
            os.makedirs(args.source_path)
        for filename in (
            "optimized_source_weights.csv",
            "annotation_data_sources.csv"
        ):
            file_path = os.path.join(args.source_path, filename)
            if not os.path.exists(file_path):
                res = requests.get(f"https://zenodo.org/records/14929618/files/{filename}")
                with open(file_path, "wb") as file:
                    file.write(res.content)
        optimized_source_weights = tuple(zip(*read_csv_rows(os.path.join(source_path, "optimized_source_weights.csv"))[1]))
        optimized_source_weights = dict(zip(optimized_source_weights[0], (float(e) for e in optimized_source_weights[1])))
        source_annotations = pd.DataFrame(read_csv_cols(os.path.join(source_path, "annotation_data_sources.csv")))
    if len(args.lr_network_file) == 0:
        raise ValueError("at least one settings file needs to be provided")
    gr_network = read_network_file(args.gr_network_file)
    lr_network = read_network_file(args.lr_network_file)
    sig_network = read_network_file(args.sig_network_file)
    parallel = Parallel(n_jobs=args.n_process)
    file_ext = args.settings_file.split(".")[-1]
    if file_ext == "json":
        # old way
        with open(args.settings_file, "rb") as file:
            settings_CV = json.loads(file.read())
        evaluation_data = EvaluationData(settings_CV["settings"])
        gr_network = gr_network[
            ~ (
                (gr_network["database"] == "NicheNet_LT") &
                np.array([fr in settings_CV["forbidden_ligands_nichenet"] for fr in gr_network["from"]])
            )
            &
            ~ (
                (gr_network["database"] == "CytoSig") &
                np.array([fr in settings_CV["forbidden_ligands_cytosig"] for fr in gr_network["from"]])
            )
        ]
    elif file_ext == "pkl":
        # new way
        with open(args.settings_file, "rb") as file:
            eval = pickle.loads(file.read())
        evaluation_data = eval["data"]
        # all ligands from a specific database present in the evaluation data have their links (in this database)
        # removed from the gene regulatory network to avoid data leakage
        forbidden_ligands = eval["forbidden_ligands"]
        gr_network = gr_network[
            ~ (
                (gr_network["database"] == "NicheNet_LT") &
                np.array([fr in forbidden_ligands["NicheNet"] for fr in gr_network["from"]])
            )
            &
            ~ (
                (gr_network["database"] == "CytoSig") &
                np.array([fr in forbidden_ligands["CytoSig"] for fr in gr_network["from"]])
            )
            &
            ~ (
                (gr_network["database"] == "Lignature") &
                np.array([fr in forbidden_ligands["Lignature"] for fr in gr_network["from"]])
            )
        ]
    else:
        raise ValueError(f"the training data should be a json (.json) or pickle (.pkl) file")
    # define the source weights that should be updated
    if len(args.included_source) > 0:
        source_names = sorted(set(args.included_source))
    elif len(args.excluded_source) > 0:
        source_names = sorted(
            set(
                chain(gr_network["source"], lr_network["source"], sig_network["source"])
            ).difference(args.excluded_source)
        )
    else:
        source_names = sorted(set(chain(gr_network["source"], lr_network["source"], sig_network["source"])))
        if args.source_path is not None: # code for old pbs scripts where I filtered on databases
            df = pd.DataFrame(
                {"source": source_names}
            ).merge(
                source_annotations,
                on="source",
                how="inner"
            )
            init_v = np.array([True for _ in range(df.shape[0])])
            if len(args.excluded_database) > 0:
                source_names = set(df[
                    [db not in args.excluded_database for db in df["database"]]
                ]["source"])
            elif len(args.included_database) > 0:
                source_names = set(df[
                    [db in args.included_database for db in df["database"]]
                ]["source"])
            if len(args.var_database) > 0:
                bool_v = reduce(
                    and_,
                    (df["database"] != db for db in args.var_database),
                    np.array([e in source_names for e in df["source"]]) if len(args.excluded_database) > 0 or len(args.included_database) > 0 else init_v
                )
                source_names_fixed = set(df[bool_v]["source"])
                source_names_var = set(df[~bool_v]["source"])
    # database column is no longer required so remove to save memory
    lr_network.drop("database", axis=1, inplace=True)
    gr_network.drop("database", axis=1, inplace=True)
    sig_network.drop("database", axis=1, inplace=True)
    # map strings to integers to save a lot of memory in the subprocesses
    syms = sorted(set(chain(
        lr_network["from"],
        lr_network["to"],
        lr_network["source"],
        gr_network["from"],
        gr_network["to"],
        gr_network["source"],
        sig_network["from"],
        sig_network["to"],
        sig_network["source"],
        evaluation_data.get_ligands(combination=False),
        set(chain(
            k for e in evaluation_data.values() for k in e[evaluation_data._de_genes_name].keys()
        ))
    )))
    sym2id = dict(zip(syms, range(len(syms))))
    lr_network["from"] = [sym2id[e] for e in lr_network["from"]]
    lr_network["to"] = [sym2id[e] for e in lr_network["to"]]
    lr_network["source"] = [sym2id[e] for e in lr_network["source"]]
    gr_network["from"] = [sym2id[e] for e in gr_network["from"]]
    gr_network["to"] = [sym2id[e] for e in gr_network["to"]]
    gr_network["source"] = [sym2id[e] for e in gr_network["source"]]
    sig_network["from"] = [sym2id[e] for e in sig_network["from"]]
    sig_network["to"] = [sym2id[e] for e in sig_network["to"]]
    sig_network["source"] = [sym2id[e] for e in sig_network["source"]]
    for dct in evaluation_data.values():
        ligand = dct[evaluation_data._ligand_name]
        dct[evaluation_data._ligand_name] = sym2id[ligand] if isinstance(ligand, gene_t) else tuple(sym2id[e] for e in ligand)
        dct[evaluation_data._de_genes_name] = {sym2id[k]: v for k, v in dct[evaluation_data._de_genes_name].items()}

    def objective(trial:Trial):
        # define source weights
        if args.source_path is not None and len(args.var_database) > 0:
            # some source weights have been fixed a priori
            source_weights = dict(
                (
                    source_name,
                    trial.suggest_float(
                        name=source_name,
                        low=0,
                        high=1
                    )
                ) for source_name in source_names_var
            )
            for source_name in source_names_fixed:
                if source_name in optimized_source_weights:
                    source_weights[source_name] = optimized_source_weights[source_name]
        else:
            source_weights = {
                source_name: trial.suggest_float(
                    name=source_name,
                    low=0,
                    high=1
                ) for source_name in source_names
            }
        # define hyperparameters
        lr_sig_hub = trial.suggest_float(
            name="lr_sig_hub",
            low=0,
            high=1
        ) if args.lr_sig_hub is None else args.lr_sig_hub
        gr_hub = trial.suggest_float(
            name="gr_hub",
            low=0,
            high=1
        ) if args.gr_hub is None else args.gr_hub
        ltf_cutoff = trial.suggest_float(
            name="ltf_cutoff",
            low=0.9,
            high=0.999
        ) if args.ltf_cutoff is None else args.ltf_cutoff
        damping_factor = trial.suggest_float(
            name="damping_factor",
            low=0.01,
            high=0.99
        ) if args.damping_factor is None else args.damping_factor
        # construct the model from the source weights and compute the objectives
        res = construct_and_evaluate(
            dict((sym2id[s], w) for s, w in source_weights.items()),
            lr_sig_hub,
            gr_hub,
            ltf_cutoff,
            damping_factor,
            lr_network,
            gr_network,
            sig_network,
            evaluation_data,
            return_all_matrices=False,
            return_weighted_networks=False
        )
        return res[1:]

    name = args.settings_file.split("/")[-1][:-5]
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    log_file = os.path.join(args.log_dir, f"{args.id}_{name}_{args.algorithm}.log")
    with open(log_file, "a" if args.c else "w"):
        pass # the file is created, if not args.c the file is emptied if it already existed
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
    elif args.algorithm == "GP":
        sampler = GPSampler(deterministic_objective=False)
    study = create_study(
        sampler=sampler,
        directions=["maximize", "maximize", "maximize", "maximize"],
        study_name=name,
        storage=storage,
        load_if_exists=args.c
    )
    parallel(optimize(name, storage, sampler) for _ in range(cpu_count() if parallel.n_jobs == -1 else parallel.n_jobs))
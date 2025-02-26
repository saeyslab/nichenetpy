from nichenetpy.utils import subset_matrix
from nichenetpy.graph import dijkstra_spl

from numbers import Number
from itertools import chain
from collections.abc import Iterable
from scipy.sparse import csr_matrix
from sknetwork.ranking import PageRank
from sknetwork.path import get_distances

import pandas as pd
import numpy as np


def _sum_weights(df, source_weights):
    return (
        df
            .merge(source_weights, on="source", how="inner")[["from", "to", "weight"]]
            .groupby(["from", "to"])
            .aggregate("sum")
    )

def construct_weighted_networks(
    lr_network:pd.DataFrame,
    sig_network:pd.DataFrame,
    gr_network:pd.DataFrame,
    source_weights:dict[str, float]|pd.DataFrame,
    n_output_networks:int=2
) -> dict[str, pd.DataFrame]:
    if type(lr_network) is not pd.DataFrame:
        raise TypeError(f"lr_network should have type pandas.DataFrame, was {type(lr_network)}")
    if type(sig_network) is not pd.DataFrame:
        raise TypeError(f"sig_network should have type pandas.DataFrame, was {type(sig_network)}")
    if type(gr_network) is not pd.DataFrame:
        raise TypeError(f"gr_network should have type pandas.DataFrame, was {type(gr_network)}")
    if type(source_weights) is dict:
        for source, weight in source_weights.items():
            if weight < 0 or weight > 1:
                raise ValueError(f"{source} weight was {weight}, should be in the interval [0, 1]")
        source_weights = pd.DataFrame(dict(zip(("source", "weight"), zip(*source_weights.items()))))
    elif type(source_weights) is pd.DataFrame:
        for source, weight in zip(source_weights["source"], source_weights["weight"]):
            if weight < 0 or weight > 1:
                raise ValueError(f"{source} weight was {weight}, should be in the interval [0, 1]")
    else:
        raise TypeError(f"source_weights should have type pandas.DataFrame or dict[str, float], was {type(source_weights)}")
    if type(n_output_networks) is not int:
        raise TypeError(f"n_output_networks should have type int, was {type(n_output_networks)}")
    if n_output_networks != 2 and n_output_networks != 3:
        raise ValueError("n_output_networks should be 2 or 3")
    source_weights = source_weights[source_weights["weight"] > 0]
    gr_network_w = _sum_weights(gr_network, source_weights)
    if n_output_networks == 2:
        ligand_signaling_w = _sum_weights(pd.concat((lr_network, sig_network)), source_weights)
        return {
            "lr_sig": ligand_signaling_w,
            "gr": gr_network_w
        }
    else:
        lr_network_w = _sum_weights(lr_network, source_weights)
        sig_network_w = _sum_weights(sig_network, source_weights)
        return {
            "lr": lr_network_w,
            "sig": sig_network_w,
            "gr": gr_network_w
        }

def apply_hub_correction(
    df:pd.DataFrame,
    hub:float
) -> pd.DataFrame:
    if type(df) is not pd.DataFrame:
        raise TypeError(f"df should haver type pandas.DataFrame, was {type(df)}")
    if not isinstance(hub, Number):
        raise TypeError(f"hub should have type float, was {type(hub)}")
    if hub < 0 or hub > 1:
        raise ValueError("hub should be in the interval [0, 1]")
    if hub == 0:
        return df.copy()
    to_count = df.groupby("to").aggregate("count")
    to_count.rename(columns={"weight": "n"}, inplace=True)
    to_count.reset_index(inplace=True)
    df.reset_index(inplace=True)
    df = df.merge(to_count, on="to", how="inner")
    df["weight"] = df["weight"] / (df["n"] ** hub)
    df.drop(columns="n", inplace=True)
    return df

def _quantile_clip(mat, cutoff):
    if cutoff > 0:
        for i in range(mat.shape[0]):
            row = mat[i, :]
            qt = np.quantile(row, cutoff)
            for j in range(len(row)):
                if row[j] <= qt:
                    mat[i, j] = 0


def construct_ligand_tf_matrix(
    weighted_networks:dict[str, pd.DataFrame],
    ligands:Iterable[Iterable[str]],
    ltf_cutoff:float=0.99,
    algorithm:str="PPR",
    damping_factor:float=0.5,
    ligands_as_cols:bool=False
) -> tuple[list[str], list[str], np.ndarray|csr_matrix]:
    if type(weighted_networks) is not dict:
        return TypeError(f"weighted_networks should have type dict[str, pandas.DataFrame], was {type(weighted_networks)}")
    if not isinstance(ligands, Iterable):
        return TypeError(f"ligands should have type Iterable[str], was {type(ligands)}")
    if not isinstance(ltf_cutoff, Number):
        return TypeError(f"ltf_cutoff should have type float, was {type(ltf_cutoff)}")
    if type(algorithm) is not str:
        return TypeError(f"algorithm should have type str, was {type(algorithm)}")
    if not isinstance(damping_factor, Number):
        return TypeError(f"damping_factor should have type float, was {type(damping_factor)}")
    if type(ligands_as_cols) is not bool:
        return TypeError(f"ligands_as_cols should have type bool, was {type(ligands_as_cols)}")
    if ltf_cutoff < 0 or ltf_cutoff > 1:
        raise ValueError(f"ltf_cutoff should be between 0 and 1, was {ltf_cutoff}")
    if damping_factor < 0 or damping_factor > 1:
        raise ValueError(f"damping_factor should be between 0 and 1, was {damping_factor}")
    lr_sig = weighted_networks["lr_sig"]
    gr = weighted_networks["gr"]
    all_genes = sorted(set(chain(lr_sig["from"], lr_sig["to"], gr["from"], gr["to"])))
    gene2id = dict(zip(all_genes, range(len(all_genes))))
    if algorithm == "PPR":
        # the adjancy matrix (and adjacency graph)
        lr_sig_mat = csr_matrix(
            (
                lr_sig["weight"],
                (
                    [gene2id[e] for e in lr_sig["from"]],
                    [gene2id[e] for e in lr_sig["to"]]
                )
            )
        )
        # preference vector
        pv = np.zeros(shape=lr_sig_mat.shape[0])
        pr = PageRank(damping_factor=damping_factor)
        complete_matrix = []
        for _ligands in ligands:
            partial_matrix = []
            for ligand in _ligands:
                pv[gene2id[ligand]] = 1 # TODO: check if this needs a reset
                partial_matrix.append(pr.fit_predict(lr_sig_mat, weights=pv))
            ppr_matrix = np.array(partial_matrix)
            ppr_matrix = ppr_matrix.reshape((
                int(sum(len(x) for x in partial_matrix)/len(pv)),
                len(pv)
            ))
            if damping_factor == 0:
                ltf_cutoff = 0
            _quantile_clip(ppr_matrix, ltf_cutoff)
            complete_matrix.append(ppr_matrix.mean(axis=0))
    elif algorithm == "SPL":
        # the adjancy matrix (and adjacency graph)
        lr_sig_mat = csr_matrix(
            (
                [1/e for e in lr_sig["weight"]],
                (
                    [gene2id[e] for e in lr_sig["from"]],
                    [gene2id[e] for e in lr_sig["to"]]
                )
            )
        )
        complete_matrix = []
        for _ligands in ligands:
            spl_matrix = np.array([dijkstra_spl(graph=lr_sig_mat, src=gene2id[src]) for src in _ligands])
            for i in range(spl_matrix.shape[0]):
                for j in range(spl_matrix.shape[1]):
                    if spl_matrix[i, j] == np.inf:
                        spl_matrix[i, j] = -1
            max_dist = spl_matrix.max()
            for i in range(spl_matrix.shape[0]):
                for j in range(spl_matrix.shape[1]):
                    if spl_matrix[i, j] == -1:
                        spl_matrix[i, j] = 0
                    else:
                        spl_matrix[i, j] = max_dist - spl_matrix[i, j]
            _quantile_clip(spl_matrix, ltf_cutoff)
            complete_matrix.append(spl_matrix.mean(axis=0))
    elif algorithm == "direct":
        raise NotImplementedError("direct is not supported yet")
    else:
        raise ValueError(f"algorithm should be 'PPR', 'SPL' or direct', was {algorithm}")
    ltf_matrix = np.array(complete_matrix)
    row_names = ["-".join(_ligands) for _ligands in ligands]
    col_names = all_genes
    if ligands_as_cols:
        return (col_names, row_names, ltf_matrix.transpose())
    return (row_names, col_names, ltf_matrix)

def construct_tf_target_matrix(
    weighted_networks:dict[str, pd.DataFrame],
    tfs_as_cols:bool=False,
    standalone_output:bool=False
) -> tuple[list[str], list[str], np.ndarray|csr_matrix]:
    if type(weighted_networks) is not dict:
        return TypeError(f"weighted_networks should have type dict[str, pandas.DataFrame], was {type(weighted_networks)}")
    if type(tfs_as_cols) is not bool:
        return TypeError(f"tfs_as_cols should have type bool, was {type(tfs_as_cols)}")
    if type(standalone_output) is not bool:
        return TypeError(f"standalone_output should have type bool, was {type(standalone_output)}")
    lr_sig = weighted_networks["lr_sig"]
    gr = weighted_networks["gr"]
    all_genes = sorted(set(chain(lr_sig["from"], lr_sig["to"], gr["from"], gr["to"])))
    gene2id = dict(zip(all_genes, range(len(all_genes))))
    fr = [gene2id[e] for e in gr["from"]]
    grn_matrix = csr_matrix(
        (
            gr["weight"],
            (
                fr,
                [gene2id[e] for e in gr["to"]]
            )
        )
    )
    row_names = all_genes
    col_names = all_genes
    if standalone_output:
        # keep only regulators with gene regulatory interactions
        rows = sorted(set(fr))
        grn_matrix = subset_matrix(
            grn_matrix,
            rows=rows
        )
        row_names = [row_names[i] for i in rows]
    if tfs_as_cols:
        return (col_names, row_names, grn_matrix.transpose())
    return (row_names, col_names, grn_matrix)

def _set_min(mat):
    for i in range(len(mat)):
        if mat[i] == 0:
            mat[i] = np.inf
    m = min(mat)
    for i in range(len(mat)):
        if mat[i] == np.inf:
            mat[i] = m

def construct_ligand_target_matrix(
    weighted_networks:dict[str, pd.DataFrame],
    lr_network:pd.DataFrame,
    ligands:Iterable[str],
    ltf_cutoff:float=0.99,
    algorithm:str="PPR",
    damping_factor:float=0.5,
    secondary_targets:bool=False,
    ligands_as_cols:bool=True,
    remove_direct_links:str="no"
) -> tuple[list[str], list[str], np.ndarray|csr_matrix]:
    if type(weighted_networks) is not dict:
        return TypeError(f"weighted_networks should have type dict[str, pandas.DataFrame], was {type(weighted_networks)}")
    if type(lr_network) is not pd.DataFrame:
        raise TypeError(f"lr_network should have type pandas.DataFrame, was {type(lr_network)}")
    if not isinstance(ligands, Iterable):
        return TypeError(f"ligands should have type Iterable[str], was {type(ligands)}")
    if not isinstance(ltf_cutoff, Number):
        return TypeError(f"ltf_cutoff should have type float, was {type(ltf_cutoff)}")
    if type(algorithm) is not str:
        return TypeError(f"algorithm should have type str, was {type(algorithm)}")
    if not isinstance(damping_factor, Number):
        return TypeError(f"damping_factor should have type float, was {type(damping_factor)}")
    if type(secondary_targets) is not bool:
        raise TypeError(f"secondary_targets should have type bool, was {type(secondary_targets)}")
    if type(ligands_as_cols) is not bool:
        return TypeError(f"ligands_as_cols should have type bool, was {type(ligands_as_cols)}")
    if type(remove_direct_links) is not str:
        raise TypeError(f"remove_direct_links should have type str, was {type(remove_direct_links)}")
    if ltf_cutoff < 0 or ltf_cutoff > 1:
        raise ValueError(f"ltf_cutoff should be between 0 and 1, was {ltf_cutoff}")
    if damping_factor < 0 or damping_factor > 1:
        raise ValueError(f"damping_factor should be between 0 and 1, was {damping_factor}")
    if remove_direct_links not in ("no", "ligand", "ligand_receptor"):
        raise ValueError(f"remove_direct_links should be in ['no', 'ligand', 'receptor'], was {remove_direct_links}")
    if remove_direct_links == "ligand":
        rm_set = set(lr_network["from"])
        weighted_networks["gr"][weighted_networks["gr"]["from"].apply(lambda x : x not in rm_set)]
    elif remove_direct_links == "ligand_receptor":
        rm_set = set(chain(lr_network["from"], lr_network["to"]))
        weighted_networks["gr"][weighted_networks["gr"]["from"].apply(lambda x : x not in rm_set)]
    ltf_rows, _, ltf_matrix = construct_ligand_tf_matrix(weighted_networks, ligands, ltf_cutoff, algorithm, damping_factor)
    _, grn_cols, grn_matrix = construct_tf_target_matrix(weighted_networks)
    ligand2target = ltf_matrix * grn_matrix
    if secondary_targets:
        _quantile_clip(ligand2target, ltf_cutoff)
        ligand2target_secondary = ligand2target * grn_matrix
        _set_min(ligand2target)
        _set_min(ligand2target_secondary)
        ligand2target = (ligand2target**-1 + ligand2target_secondary**-1)**-1
    if ligands_as_cols:
        return (grn_cols, ltf_rows, ligand2target.transpose())
    return (ltf_rows, grn_cols, ligand2target)
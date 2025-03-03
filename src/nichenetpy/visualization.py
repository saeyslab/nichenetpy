from nichenetpy.utils import subset_matrix
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.network import WeightedNetwork
from nichenetpy.graph import walk_graph

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from collections.abc import Iterable, Collection
from numbers import Number
from scipy.sparse import csr_matrix
from itertools import chain

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import pandas as pd


def reorder_labels(
    mat:np.ndarray,
    row_labels:list[str],
    col_labels:list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    '''
    Reorders the rows and columns of the matrix along with the corresponding labels based on hierarchic clustering. 

    Parameters
    ----------
    mat : numpy.ndarray
        the matrix to reorder
    row_labels : list or tuple of str
        the row labels
    col_labels : list or tuple of str
        the column labels

    Returns
    -------
    numpy.ndarray
        the reordered matrix
    list[str]
        the reordered row labels
    list[str]
        the reordered column labels

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(mat) is not np.ndarray:
        raise TypeError(f"mat should have type numpy.ndarray, was {type(mat)}")
    if type(row_labels) is not list and type(row_labels) is not tuple:
        raise TypeError(f"row_labels should have type list[str] or tuple[str], was {type(row_labels)}")
    if type(col_labels) is not list and type(col_labels) is not tuple:
        raise TypeError(f"col_labels should have type list[str] or tuple[str], was {type(col_labels)}")
    nrows, ncols = mat.shape
    if nrows > 1 and ncols > 1:
        corr = np.corrcoef(mat, rowvar=False)
        corr = 1 - corr
        dist = sc.spatial.distance_matrix(corr, corr)
        clust = sc.cluster.hierarchy.ward(sc.spatial.distance.squareform(dist))
        order_cols = sc.cluster.hierarchy.leaves_list(clust)
        corr = np.corrcoef(mat, rowvar=True)
        corr = 1 - corr
        dist = sc.spatial.distance_matrix(corr, corr)
        clust = sc.cluster.hierarchy.ward(sc.spatial.distance.squareform(dist))
        order_rows = sc.cluster.hierarchy.leaves_list(clust)
        mat = subset_matrix(mat, order_rows, order_cols)
        row_labels = [row_labels[i] for i in order_rows]
        col_labels = [col_labels[i] for i in order_cols]
    return (mat, row_labels, col_labels)

def prepare_ligand_target_visualization(
    predictor:LigandActivityPredictor,
    ligand_target_links:Iterable[tuple[str, str, float]],
    cutoff:float=0.25
) -> tuple[np.ndarray, list[str], list[str]]:
    '''
    Compute the data for a heatmap of regulatory potential. 

    Parameters
    ----------
    predictor : LigandActivityPredictor
        the ligand activity predictor which contains the required ligand-target prior model
    ligand_target_links : Iterable of tuples
        Iterable of (ligand, target, regulatory_potential_scores) tuples
    cutoff : float
        quantile cutoff on the ligand-target scores of the input weighted ligand-target network, scores under this cutoff will be set to 0

    Returns
    -------
    numpy.ndarray
        a matrix giving the ligand-target regulatory potential scores between ligands of interest and their targets genes part of the gene set of interest
    list[str]
        the row labels of the matrix
    list[str]
        the column labels of the matrix

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(predictor) is not LigandActivityPredictor:
        raise TypeError(f"predictor should have type LigandActivityPredictor, was {type(predictor)}")
    if not isinstance(ligand_target_links, Iterable):
        raise TypeError(f"ligand_target_links should have type Iterable, was {type(ligand_target_links)}")
    if not isinstance(cutoff, Number):
        raise TypeError(f"cutoff should have type float, was {type(cutoff)}")
    ligands, targets, weights = zip(*ligand_target_links)
    ligands = sorted(set(ligands))
    targets = sorted(set(targets))
    # select ligands and targets that appear in ligand_target_links
    ligand_target_vis = subset_matrix(
        predictor.ligand_target_matrix,
        [predictor.gene2index[target] for target in targets],
        [predictor.ligand2index[ligand] for ligand in ligands]
    )
    ligand2index = dict(zip(ligands, range(len(ligands))))
    target2index = dict(zip(targets, range(len(targets))))
    # define a cutoff on the ligand-target links
    cutoff = np.quantile(weights, [cutoff])[0]
    nrows, ncols = ligand_target_vis.shape
    ligand_target_vis = np.array([
        [ligand_target_vis[r, c] if ligand_target_vis[r, c] >= cutoff else 0 for c in range(ncols)]
        for r in range(nrows)
    ])
    # keep only rows and columns that contain at least one non-zero element
    ligands = [ligand for ligand in ligands if any(ligand_target_vis[:, ligand2index[ligand]])]
    targets = [target for target in targets if any(ligand_target_vis[target2index[target], :])]
    ligand_target_vis = subset_matrix(
        ligand_target_vis,
        [target2index[target] for target in targets],
        [ligand2index[ligand] for ligand in ligands]
    )
    return reorder_labels(ligand_target_vis, targets, ligands)

def prepare_ligand_receptor_visualization(ligand_receptor_links:WeightedNetwork) -> tuple[np.ndarray, list[str], list[str]]:
    '''
    Compute the data for a heatmap of prior interaction potential. 

    Parameters
    ----------
    ligand_receptor_links : WeigthedNetwork
        the weighted ligand-receptor links between a possible ligand and its receptors

    Returns
    -------
    numpy.ndarray
        a matrix giving the ligand-receptor prior interaction potential scores between a possible ligand and its receptors
    list[str]
        the row labels of the matrix
    list[str]
        the column labels of the matrix

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(ligand_receptor_links) is not WeightedNetwork:
        raise TypeError(f"ligand_receptor_links should have type WeightedNetwork, was {type(ligand_receptor_links)}")
    ligands = sorted(ligand_receptor_links.get_ligands())
    receptors = sorted(ligand_receptor_links.get_receptors())
    ligand2index = dict(zip(ligands, range(len(ligands))))
    receptor2index = dict(zip(receptors, range(len(receptors))))
    mat = np.zeros(shape=(len(ligands), len(receptors)))
    for ligand, receptor, weight in ligand_receptor_links:
        mat[ligand2index[ligand], receptor2index[receptor]] = weight
    return reorder_labels(mat, ligands, receptors)

def heatmap_1d(
    vals:Iterable[float],
    labels:Collection[str],
    title:str=None,
    cbar_label:str=None,
    cmap:str="Greys",
    figsize:tuple[float]=(8, 8)
) -> tuple[Figure, Axes]:
    '''
    Create a 1d heatmap using matplotlib. 

    Parameters
    ----------
    vals : Iterable of float
        the values to plot
    labels : Collection of str
        the labels of the values
    title : str
        the title of the plot
    cbar_label : str
        the label of the color bar
    cmap : str
        the name of the color map
    figsize : tuple of float
        the size of figure

    Returns
    -------
    Figure
        the matplotlib figure
    Axes
        the matplotlib axes
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if not isinstance(vals, Iterable):
        raise TypeError(f"vals should have type Iterable, was {type(vals)}")
    if not isinstance(labels, Collection):
        raise TypeError(f"labels should have type Collection, was {type(labels)}")
    if type(cmap) is not str:
        raise TypeError(f"cmap should have type str, was {type(cmap)}")
    if type(figsize) is not tuple:
        raise TypeError(f"figsize should have type tuple, was {type(figsize)}")
    fig, ax = plt.subplots(figsize=figsize)
    ys = range(len(labels)+1)
    im = ax.pcolormesh([0, 1], ys, [[val] for val in vals], cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.set_yticks(np.arange(len(labels))+0.5, labels=labels)
    if title is not None:
        if type(title) is not str:
            raise TypeError(f"title should have type str, was {type(title)}")
        ax.set_title(title)
    fig.tight_layout()
    if cbar_label is not None:
        if type(cbar_label) is not str:
            raise TypeError(f"cbar_label should have type str, was {type(cbar_label)}")
        plt.colorbar(im, label=cbar_label)
    return (fig, ax)

def heatmap_2d(
    mat:list[list[float]]|np.ndarray,
    xlabels:Collection[str],
    ylabels:Collection[str],
    xtitle:str=None,
    ytitle:str=None,
    cbar_label:str=None,
    cbar_position:str="top",
    cbar_orientation:str="horizontal",
    cmap:str="Greys",
    figsize:tuple[float]=(5, 5)
) -> tuple[Figure, Axes]:
    '''
    Create a 2d heatmap using matplotlib. 

    Parameters
    ----------
    mat : numpy.ndarray or list of list of float
        a matrix of values to plot
    xlabels : Collection of str
        the labels of the x values
    ylabels : Collection of str
        the labels of the y values
    xtitle : str
        the title of the x-axis
    ytitle : str
        the title of the y-axis
    cbar_label : str
        the label of the color bar
    cbar_position : str
        the position of the color bar ("top", "bottom", "left" or "right")
    cbar_orientation : str
        the orientation of the color bar ("horizontal", "vertical")
    cmap : str
        the name of the color map
    figsize : tuple of float
        the size of figure

    Returns
    -------
    Figure
        the matplotlib figure
    Axes
        the matplotlib axes

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(mat) is not np.ndarray and type(mat) is not list and type(mat) is not tuple:
        raise TypeError(f"mat should have type numpy.ndarray or list or tuple, was {type(mat)}")
    if not isinstance(xlabels, Collection):
        raise TypeError(f"xlabels should have type Collection, was {type(xlabels)}")
    if not isinstance(ylabels, Collection):
        raise TypeError(f"ylabels should have type Collection, was {type(ylabels)}")
    if type(cmap) is not str:
        raise TypeError(f"cmap should have type str, was {type(cmap)}")
    if type(figsize) is not tuple:
        raise TypeError(f"figsize should have type tuple, was {type(figsize)}")
    if type(cbar_position) is not str:
        raise TypeError(f"cbar_position should have type str, was {type(cbar_position)}")
    if type(cbar_orientation) is not str:
        raise TypeError(f"cbar_orientation should have type str, was {type(cbar_orientation)}")
    fig, ax = plt.subplots(figsize=figsize)
    xs = range(len(xlabels))
    ys = range(len(ylabels))
    im = ax.pcolormesh(xs, ys, mat, cmap=cmap)
    xts = np.arange(len(xlabels))
    yts = np.arange(len(ylabels))
    ax.set_xticks(xts, labels=xlabels)
    ax.set_yticks(yts, labels=ylabels)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    trans = mtrans.Affine2D().translate(-7, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
    fig.tight_layout()
    if cbar_label is not None:
        if type(cbar_label) is not str:
            raise TypeError(f"cbar_label should have type str, was {type(cbar_label)}")
        plt.colorbar(im, fraction=0.05, orientation=cbar_orientation, location=cbar_position, label=cbar_label)
    if xtitle is not None:
        if type(xtitle) is not str:
            raise TypeError(f"xtitle should have type str, was {type(xtitle)}")
        plt.xlabel(xtitle)
    if ytitle is not None:
        if type(ytitle) is not str:
            raise TypeError(f"ytitle should have type str, was {type(ytitle)}")
        plt.ylabel(ytitle)
    plt.hlines([y + 0.5 for y in ys[:-1]], xs[0]-0.5, xs[-1]+0.5, color="white")
    plt.vlines([x + 0.5 for x in xs[:-1]], ys[0]-0.5, ys[-1]+0.5, color="white")
    return (fig, ax)

def construct_ligand_signaling_df(
    ligands_oi:Iterable[str],
    targets_oi:Iterable[str],
    all_ligands:Iterable[str],
    all_targets:Iterable[str],
    gr:pd.DataFrame,
    ltf_matrix:csr_matrix,
    k:int
) -> pd.DataFrame:
    pd.options.mode.chained_assignment = None # false positive warnings removal
    target2id = dict(zip(all_targets, range(len(all_targets))))
    dfs = []
    for ligand in ligands_oi:
        ltf_vis = pd.DataFrame(data=ltf_matrix[:, target2id[ligand]].toarray(), columns=["weight"], index=all_ligands)
        ltf_vis.index.name = "TF"
        ltf_vis.reset_index(inplace=True)
        ltf_vis = ltf_vis[ltf_vis["weight"] > 0]
        ltf_vis["ligand"] = [ligand for _ in range(len(ltf_vis))]
        for target in targets_oi:
            gr_filtered = gr[gr["to"] == target]
            gr_filtered.rename(columns={"from": "TF", "weight": "weight_grn"}, inplace=True)
            combined_df = ltf_vis.merge(gr_filtered, on="TF")
            combined_df["total_weight"] = combined_df["weight"] * combined_df["weight_grn"]
            combined_df.sort_values(by="total_weight", ascending=False, inplace=True)
            combined_df = combined_df.iloc[0:min(k, len(combined_df))]
            dfs.append(combined_df)
    return pd.concat(dfs)

def _minmax_scaling(df):
    weight = np.array(df["weight"])
    mn = min(weight)
    mx = max(weight)
    df["weight"] = (weight - mn) / (mx - mn) + 0.75

def get_ligand_signaling_path(
    ltf_matrix:csr_matrix,
    ligands_oi:Iterable[str],
    targets_oi:Iterable[str],
    all_ligands:Iterable[str],
    all_targets:Iterable[str],
    lr_sig:pd.DataFrame,
    gr:pd.DataFrame,
    top_n_regulators:int=4,
    minmax_scaling:bool=False
) -> tuple[pd.DataFrame]:
    combined_df = construct_ligand_signaling_df(
        ligands_oi,
        targets_oi,
        all_ligands,
        all_targets,
        gr,
        ltf_matrix,
        top_n_regulators
    )
    all_genes = sorted(set(chain(lr_sig["from"], lr_sig["to"], gr["from"], gr["to"])))
    gene2id = dict(zip(all_genes, range(len(all_genes))))
    lr_sig_mat = csr_matrix(
        (
            [1/e for e in lr_sig["weight"]],
            (
                [gene2id[e] for e in lr_sig["from"]],
                [gene2id[e] for e in lr_sig["to"]]
            )
        )
    )
    tfs = set()
    for ligand in ligands_oi:
        ligand_signaling = combined_df[combined_df["ligand"] == ligand]
        ligand_id = gene2id[ligand]
        tfs = tfs.union(set(ligand_signaling["TF"]).intersection(walk_graph(lr_sig_mat, src=ligand_id)))
        tfs.remove(ligand_id)
    tfs = {all_genes[id] for id in tfs}
    tf_signaling = lr_sig[[(fr in ligands_oi or fr in tfs) and to in tfs for fr, to in zip(lr_sig["from"], lr_sig["to"])]]
    tf_signaling = tf_signaling.groupby(("from", "to")).sum()
    tf_regulatory = gr[[fr in combined_df and to in targets_oi for fr, to in zip(gr["from"], gr["to"])]]
    if minmax_scaling:
        _minmax_scaling(tf_signaling)
        _minmax_scaling(tf_regulatory)
    return (tf_signaling, tf_regulatory)
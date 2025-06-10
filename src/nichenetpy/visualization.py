from nichenetpy.utils import (
    subset_matrix,
    get_ties
)
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.network import WeightedNetwork
from nichenetpy.graph import get_reachable_nodes
from nichenetpy.ann_utils import subset_ann

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    Rectangle,
    Wedge
)
from matplotlib.collections import PatchCollection
from matplotlib.text import Text
from matplotlib.colors import colorConverter
from matplotlib import colormaps as cm
from matplotlib.patheffects import withStroke
from math import (
    isnan,
    sqrt
)
from collections.abc import (
    Iterable,
    Collection
)
from numbers import Number
from scipy.sparse import csr_matrix
from itertools import chain, repeat
from anndata import AnnData
from nichenetpy.ann_utils import subset_ann

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import pandas as pd
import networkx as nx


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
        [predictor.gene2index(target) for target in targets],
        [predictor.ligand2index(ligand) for ligand in ligands]
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
    title:str|None=None,
    cbar_label:str|None=None,
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
    title : str or None
        the title of the plot
    cbar_label : str or None
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
    xtitle:str|None=None,
    ytitle:str|None=None,
    cbar_label:str|None=None,
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
    xtitle : str or None
        the title of the x-axis
    ytitle : str or None
        the title of the y-axis
    cbar_label : str or None
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

def _construct_ligand_signaling_df(
    ligands_oi:Collection[str],
    targets_oi:Collection[str],
    all_ligands:Collection[str],
    all_targets:Collection[str],
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
    mn = weight.min()
    mx = weight.max()
    df["weight"] = (weight - mn) / (mx - mn) + 0.75

def get_ligand_signaling_path(
    ltf_matrix:csr_matrix,
    ligands_oi:Collection[str],
    targets_oi:Collection[str],
    all_ligands:Collection[str],
    all_targets:Collection[str],
    lr_sig:pd.DataFrame,
    gr:pd.DataFrame,
    top_n_regulators:int=4,
    minmax_scaling:bool=False
) -> tuple[pd.DataFrame]:
    '''
    Extract possible signaling paths between a ligand and target gene of interest. The most highly weighted path(s) will be extracted.

    Parameters
    ----------
    ltf_matrix : csr_matrix
        a row-major parse matrix of ligand-regulator probability scores
    ligands_oi : Collection of str
        the ligands of interest
    targets_oi : Collection of str
        the target genes of interest
    all_ligands : Collection of str
        all ligands (row labels of ltf_matrix)
    all_targets : Collection of str
        all targets (column labels of ltf_matrix)
    lr_sig : pandas.DataFrame
        dataframe which contains weighted ligand-receptor and signaling interactions (from, to, weight)
    gr : pandas.DataFrame
        dataframe which contains weighted gene regulatory interactions (from, to, weight)
    top_n_regulators : int
        The number of top regulators that should be included in the ligand-target signaling network.
        Top regulators are regulators that score both high for being upstream of the target gene(s) and high for being downstream of the ligand.
        Default: 4
    minmax_scaling : bool
        Indicate whether the weights of both dataframes should be min-max scaled between 0.75 and 1.
        Default: FALSE

    Returns
    -------
    tuple of pandas.DataFrame
        the integrated weighted ligand-signaling and gene regulatory network data frames

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(ltf_matrix) is not csr_matrix:
        raise TypeError(f"ltf_matrix should have type csr_matrix, was {type(ltf_matrix)}")
    if not isinstance(ligands_oi, Collection):
        raise TypeError(f"ligands_oi should have type Collection[str], was {type(ligands_oi)}")
    if not isinstance(targets_oi, Collection):
        raise TypeError(f"targets_oi should have type Collection[str], was {type(targets_oi)}")
    if not isinstance(all_ligands, Collection):
        raise TypeError(f"all_ligands should have type Collection[str], was {type(all_ligands)}")
    if not isinstance(all_targets, Collection):
        raise TypeError(f"all_targets should have type Collection[str], was {type(all_targets)}")
    if type(lr_sig) is not pd.DataFrame:
        raise TypeError(f"lr_sig should have type pandas.DataFrame, was {type(lr_sig)}")
    if type(gr) is not pd.DataFrame:
        raise TypeError(f"gr should have type pandas.DataFrame, was {type(gr)}")
    if type(top_n_regulators) is not int:
        raise TypeError(f"top_n_regulators should have type int, was {type(top_n_regulators)}")
    if type(minmax_scaling) is not bool:
        raise TypeError(f"minmax_scaling should have type bool, was {type(minmax_scaling)}")
    combined_df = _construct_ligand_signaling_df(
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
        tfs.update(set(gene2id[e] for e in ligand_signaling["TF"]).intersection(get_reachable_nodes(lr_sig_mat, src=ligand_id)))
        try:
            tfs.remove(ligand_id)
        except KeyError:
            pass # if it's not in there, that's great!
    tfs = {all_genes[id] for id in tfs}
    tf_signaling = lr_sig[[(fr in ligands_oi or fr in tfs) and to in tfs for fr, to in zip(lr_sig["from"], lr_sig["to"])]]
    tf_signaling = tf_signaling.groupby(["from", "to"], as_index=False).sum()
    combined_df_tf = set(combined_df["TF"])
    tf_regulatory = gr[[fr in combined_df_tf and to in targets_oi for fr, to in zip(gr["from"], gr["to"])]]
    if minmax_scaling:
        _minmax_scaling(tf_signaling)
        _minmax_scaling(tf_regulatory)
    return (tf_signaling, tf_regulatory)

def visualize_ligand_signaling_graph(
    tf_signaling:pd.DataFrame,
    tf_regulatory:pd.DataFrame,
    ligands_oi:Collection[str],
    targets_oi:Collection[str],
    node_size:int=1300,
    arrow_size:int=10,
    label_size:int=7,
    seed:int|None=0
):
    '''
    Visualize extracted ligand-target signaling network. 

    Parameters
    ----------
    tf_signaling : pandas.DataFrame
        dataframe which contains weighted ligand-receptor and signaling interactions  (from, to, weight)
    tf_regulatory : pandas.DataFrame
        dataframe which contains weighted gene regulatory interactions (from, to, weight)
    ligands_oi : Collection of str
        the ligands of interest
    targets_oi : Collection of str
        the target genes of interest
    node_size : int
        the size of the nodes in the visualized network
    arrow_size : int
        the size of the arrows in the visualized network
    label_size : int
        the size of the node labels in the visualized network
    seed : int or None
        the random seed

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    '''
    if type(tf_signaling) is not pd.DataFrame:
        raise TypeError(f"tf_signaling should have type pandas.DataFrame, was {type(tf_signaling)}")
    if type(tf_regulatory) is not pd.DataFrame:
        raise TypeError(f"tf_regulatory should have type pandas.DataFrame, was {type(tf_regulatory)}")
    if not isinstance(ligands_oi, Collection):
        raise TypeError(f"ligands_oi should have type Collection[str], was {type(ligands_oi)}")
    if not isinstance(targets_oi, Collection):
        raise TypeError(f"targets_oi should have type Collection[str], was {type(targets_oi)}")
    if type(node_size) is not int:
        raise TypeError(f"node_size should have type int, was {type(node_size)}")
    if type(arrow_size) is not int:
        raise TypeError(f"arrow_size should have type int, was {type(arrow_size)}")
    if type(label_size) is not int:
        raise TypeError(f"font_size should have type int, was {type(label_size)}")
    if type(seed) is not int:
        raise TypeError(f"seed should have type int, was {type(seed)}")
    graph = nx.DiGraph()
    for fr, to, w, c in chain(
        zip(tf_signaling["from"], tf_signaling["to"], tf_signaling["weight"], repeat("red")),
        zip(tf_regulatory["from"], tf_regulatory["to"], tf_regulatory["weight"], repeat("blue"))
    ):
        graph.add_edge(fr, to, weight=w, color=c)
    pos = nx.arf_layout(graph, seed=seed)
    node2color = dict((node, ("red" if node in ligands_oi else "blue" if node in targets_oi else "grey")) for node in graph.nodes)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=[node2color[node] for node in graph.nodes],
        node_size=node_size
    )
    nx.draw_networkx_labels(graph, pos, labels=dict(zip(graph.nodes, graph.nodes)), font_size=label_size, font_color="white")
    nx.draw_networkx_edges(
        graph,
        pos,
        arrowstyle="->",
        arrowsize=arrow_size,
        arrows=True,
        label=graph.nodes,
        node_size=node_size,
        edge_color=[e[2]["color"] for e in graph.edges.data()],
        width=[e[2]["weight"] for e in graph.edges.data()]
    )

def assign_ligands_to_celltype(
    ann:AnnData,
    ligands:Collection[str],
    celltype_col:str="celltype",
    condition_oi:str|None=None,
    condition_col:str|None=None,
    layer:str="data"
):
    '''
    Assign ligands to a sender cell type, based on the strongest expressing cell type of that ligand.
    Ligands are only assigned to a cell type if that cell type is the only one to show an expression
    that is higher than the average + SD. Otherwise, it is assigned to "General".

    Parameters
    ----------
    ann : anndata.AnnData
        the AnnData object
    ligands : Collection of str
        the ligands to assign to cell types
    celltype_col : str
        the name of the column in ann.obs which contains the cell types
    condition_oi : str or None
        the condition of interest
    condition_col : str or None
        the name of the column in ann.obs which contains the conditions
    layer : str
        the layer in ann to use

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    
    Returns
    -------
    pandas.DataFrame
        A data frame with two columns, the cell type the ligand has been assigned to (ligand_type) and the ligand name (ligand)
    '''
    if type(ann) is not AnnData:
        raise TypeError(f"ann should have type anndata.AnnData, was {type(ann)}")
    if not isinstance(ligands, Collection):
        raise TypeError(f"ligands should have type Collection[str], was {type(ligands)}")
    if type(celltype_col) is not str:
        raise TypeError(f"celltype_col should have type str, was {type(celltype_col)}")
    if condition_oi is not None and type(condition_oi) is not str:
        raise TypeError(f"condition_oi should have type str, was {type(condition_oi)}")
    if condition_col is not None and type(condition_col) is not str:
        raise TypeError(f"condition_col should have type str, was {type(condition_col)}")
    if type(layer) is not str:
        raise TypeError(f"layer should have type str, was {type(layer)}")
    if condition_col is None:
        ann_sub = subset_ann(
            ann,
            genes=ligands,
            layers=[layer]
        )
    else:
        ann_sub = subset_ann(
            ann,
            genes=ligands,
            val_col=condition_col,
            val=condition_oi,
            layers=[layer]
        )
    celltypes = sorted(set(ann_sub.obs[celltype_col]))
    avg_expression_ligands = []
    for celltype in celltypes:
        ann_celltype = subset_ann(
            ann_sub,
            val_col="celltype",
            val=celltype,
            layers=[layer]
        )
        mat = ann_celltype.layers[layer]
        if layer == "data":
            mat = np.expm1(mat)
        mat = mat.tocsr()
        avg_expression_ligands.append(mat.mean(axis=0).A.reshape((-1,)))
    avg_expression_ligands = np.array(avg_expression_ligands)
    assig = np.mean(avg_expression_ligands, axis=0)
    assig += np.std(avg_expression_ligands, mean=assig, axis=0)
    sender_ligand_assignment = {
        celltype: {
            ann_sub.var_names[j]
            for j, e in enumerate(avg_expression_ligands[i, :])
            if e > assig[j]
        }
        for i, celltype in enumerate(celltypes)
    }
    count = dict()
    for e in chain(*sender_ligand_assignment.values()):
        if e in count:
            count[e] += 1
        else:
            count[e] = 1
    unique_ligands = {k for k, v in count.items() if v == 1}
    general_ligands = (
        ligands if type(ligands) is set else set(ligands)
    ).difference(unique_ligands)
    for e in sender_ligand_assignment.values():
        e.difference_update(general_ligands)
    return pd.DataFrame({
        "ligand_type": chain(
            chain(
                *(repeat(k, len(sender_ligand_assignment[k])) for k in sender_ligand_assignment.keys())
            ),
            repeat("General", len(general_ligands))
        ),
        "ligand": chain(
            chain(*sender_ligand_assignment.values()),
            general_ligands
        )
    })

def _plot_ties(
    ties,
    x_col,
    max_ligands,
    r_tie,
    top_offset,
    ax,
    ymax,
    tie_color
):
    for group in ties.values():
        if len(group) > 1 and all((e < max_ligands for e in group)):
            start = min(group) + 1
            end = max(group) + 1
            ax.add_collection(
                PatchCollection(
                    (
                        Rectangle(
                            xy=(x_col - r_tie, (start + top_offset)/ymax),
                            width=2*r_tie,
                            height=(end-start)/ymax
                        ),
                        Wedge(
                            (x_col, (start + top_offset)/ymax),
                            r_tie,
                            theta1=180,
                            theta2=360
                        ),
                        Wedge(
                            (x_col, (end + top_offset)/ymax),
                            r_tie,
                            theta1=0,
                            theta2=180
                        )
                    ),
                    fc=tie_color
                )
            )

def create_barcode_plot(
    ligand_activities_agnostic:Iterable,
    ligand_activities_focused:Iterable,
    x_col_barcode:float=0.1,
    x_col_rank:float=0.4,
    x_col_agnostic:float=0.6,
    x_col_focused:float=0.9,
    r:float=0.005,
    max_ligands:int=20,
    figsize:tuple[float, float]=(10, 10)
) -> tuple[Figure, Axes]:
    '''
    Creates a slope graph combined with a barcode plot. This visualization is suitable for comparison between
    the sender-focused ranking and the sender-agnostic ranking. 

    Parameters
    ----------
    ligand_activities_agnostic : Iterable
        the ligand activity metrics for the sender-agnostic approach
    ligand_activities_focused : Iterable
        the ligand activity metrics for the sender-focused approach
    x_col_barcode : float
        the relative x-coordinate of the barcode
    x_col_rank : float
        the relative x-coordinate of the rank labels
    x_col_agnostic : float
        the relative x-coordinate of the sender-agnostic ranking
    layer : float
        the relative x-coordinate of the sender-focused ranking
    r : float
        the radius of the dots in the slope graph
    max_ligands : int
        the maximum amount of ligands
    figsize : tuple of floats
        the size of the figure

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    
    Returns
    -------
    Figure
        the figure
    Axes
        the axes
    '''
    if type(ligand_activities_agnostic) is dict:
        ligand_activities_agnostic = ligand_activities_agnostic.items()
    elif not isinstance(ligand_activities_agnostic, Iterable):
        raise TypeError(f"ligand_activities_agnostic should be a dictionary or an iterable of (key, value) tuples, was {type(ligand_activities_agnostic)}")
    if type(ligand_activities_focused) is dict:
        ligand_activities_focused = ligand_activities_focused.items()
    elif not isinstance(ligand_activities_focused, Iterable):
        raise TypeError(f"ligand_activities_focused should be a dictionary or an iterable of (key, value) tuples, was {type(ligand_activities_focused)}")
    if not isinstance(x_col_barcode, Number):
        raise TypeError(f"x_col_barcode should have type float, was {type(x_col_barcode)}")
    if not isinstance(x_col_rank, Number):
        raise TypeError(f"x_col_rank should have type float, was {type(x_col_rank)}")
    if not isinstance(x_col_agnostic, Number):
        raise TypeError(f"x_col_agnostic should have type float, was {type(x_col_agnostic)}")
    if not isinstance(x_col_focused, Number):
        raise TypeError(f"x_col_focused should have type float, was {type(x_col_focused)}")
    if not isinstance(r, Number):
        raise TypeError(f"r should have type float, was {type(r)}")
    if type(max_ligands) is not int:
        raise TypeError(f"max_ligands should have type iny, was {type(max_ligands)}")
    columns = ("rank", "ligand")
    scores1 = sorted(
        ((e[0], e[1]["aupr_corrected"]) for e in ligand_activities_agnostic),
        key=lambda x : (-x[1], x[0])
    )
    ranking1 = pd.DataFrame(
        zip(
            range(1, len(scores1)),
            (e[0] for e in scores1)
        ),
        columns=columns
    )
    scores2 = sorted(
        ((e[0], e[1]["aupr_corrected"]) for e in ligand_activities_focused),
        key=lambda x : (-x[1], x[0])
    )
    ranking2 = pd.DataFrame(
        zip(
            range(1, len(scores2)),
            (e[0] for e in scores2)
        ),
        columns=columns
    )
    ranking = ranking1.merge(
        ranking2,
        on="ligand",
        how="outer",
        suffixes=("_agnostic", "_focused")
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    top_offset = max_ligands/6
    bot_offset = 0.5
    ymax = max_ligands + top_offset + bot_offset
    ax.set_ylim(
        ymin=0,
        ymax=1
    )
    ax.invert_yaxis()
    ax.add_artist(Text(
        x_col_rank,
        top_offset/ymax,
        "Rank",
        horizontalalignment="center",
        size=12
    ))
    ax.add_artist(Text(
        x_col_agnostic,
        top_offset/ymax,
        "Agnostic",
        horizontalalignment="center",
        size=12
    ))
    ax.add_artist(Text(
        x_col_focused,
        top_offset/ymax,
        "Focused",
        horizontalalignment="center",
        size=12
    ))
    # add marks and lines
    for ligand, rank_agnostic, rank_focused in zip(ranking["ligand"], ranking["rank_agnostic"], ranking["rank_focused"]):
        xy1 = (x_col_agnostic, (rank_agnostic + top_offset)/ymax)
        xy2 = (x_col_focused, (rank_focused + top_offset)/ymax)
        if isnan(rank_focused):
            color="red"
        else:
            color="black"
            if rank_focused <= max_ligands:
                ax.add_line(
                    Line2D(
                        (xy1[0], xy2[0]),
                        (xy1[1], xy2[1]),
                        color=color
                    )
                )
                ax.add_patch(Circle(xy2, r, fc=color))
                ax.add_artist(Text(
                    xy2[0]+0.01,
                    xy2[1],
                    ligand,
                    color=color,
                    verticalalignment="center"
                ))
        if rank_agnostic <= max_ligands:
            ax.add_artist(Text(
                x_col_rank,
                (rank_agnostic + top_offset)/ymax,
                rank_agnostic,
                horizontalalignment="center",
                verticalalignment="center"
            ))
            ax.add_patch(Circle(xy1, r, fc=color))
            ax.add_artist(Text(
                xy1[0]-0.01,
                xy1[1],
                ligand,
                horizontalalignment="right",
                verticalalignment="center",
                color=color
            ))
    # add ties
    tie_color = colorConverter.to_rgba("0.3", alpha=0.5)
    r_tie = r * 1.6
    _plot_ties(
        get_ties(
            scores1,
            key=lambda x : x[1]
        ),
        x_col=x_col_agnostic,
        max_ligands=max_ligands,
        r_tie=r_tie,
        top_offset=top_offset,
        ax=ax,
        ymax=ymax,
        tie_color=tie_color
    )
    _plot_ties(
        get_ties(
            scores2,
            key=lambda x : x[1]
        ),
        x_col=x_col_focused,
        max_ligands=max_ligands,
        r_tie=r_tie,
        top_offset=top_offset,
        ax=ax,
        ymax=ymax,
        tie_color=tie_color
    )
    # add legend
    legend_width = 0.28
    legend_height = 0.07
    legend_center = ((x_col_rank + x_col_focused)/2, top_offset/ymax * 0.35)
    legend_xy = (legend_center[0] - legend_width/2, legend_center[1] - legend_height/2)
    legend_r = legend_height*0.12
    ax.add_patch(Circle(
        (legend_xy[0] + legend_r, legend_center[1]),
        radius=legend_r,
        fc="red"
    ))
    ax.add_artist(Text(
        legend_xy[0] + legend_r*2.5,
        legend_center[1],
        "Only agnostic",
        horizontalalignment="left",
        verticalalignment="center"
    ))
    ax.add_patch(Rectangle(
        (legend_center[0] + 0.06*legend_width, legend_center[1] - legend_r),
        width=legend_r*2,
        height=legend_r*2,
        fc=tie_color
    ))
    ax.add_artist(Text(
        legend_center[0] + 0.06*legend_width + legend_r*2.5,
        legend_center[1],
        "Tied",
        horizontalalignment="left",
        verticalalignment="center"
    ))
    # add barcode plot
    top_offset = ranking.shape[0]/6
    ymax = ranking.shape[0] + top_offset + bot_offset
    runs = []
    run_length = 1
    group = isnan(ranking["rank_focused"][0])
    ranking.sort_values(by="rank_agnostic", ascending=True, inplace=True)
    for rank in ranking["rank_focused"][1:]:
        if isnan(rank) == group:
            run_length += 1
        else:
            runs.append((group, run_length))
            group = isnan(rank)
            run_length = 1
    pos = 0
    for group, run_length in runs:
        segment_height = run_length/ymax
        ax.add_patch(Rectangle(
            (x_col_barcode, top_offset/ymax + pos),
            width=x_col_rank - x_col_barcode - 0.1,
            height=segment_height,
            color="red" if group else "black"
        ))
        pos += segment_height
    return (fig, ax)

def gradient_image(ax, extent, direction=0.3, cmap_range=(0, 1), **kwargs):
    """
    Draw a gradient image based on a colormap.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    extent
        The extent of the image as (xmin, xmax, ymin, ymax).
        By default, this is in Axes coordinates but may be
        changed using the *transform* keyword argument.
    direction : float
        The direction of the gradient. This is a number in
        range 0 (=vertical) to 1 (=horizontal).
    cmap_range : float, float
        The fraction (cmin, cmax) of the colormap that should be
        used for the gradient, where the complete colormap is (0, 1).
    **kwargs
        Other parameters are passed on to `.Axes.imshow()`.
        In particular useful is *cmap*.
    
    Notes
    -----
    source: https://matplotlib.org/3.5.1/gallery/lines_bars_and_markers/gradient_bar.html
    """
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    im = ax.imshow(X, extent=extent, interpolation='bicubic',
                   vmin=0, vmax=1, **kwargs)
    return im

_mushroom_plot_translate_keywords = {
    "lfc": "LFC",
    "p": "pval",
    "val": "",
    "prod": "product",
    "avg": "mean",
    "adj": "adjusted",
    "exprs": "expression"
}

def _mushroomplot_label(
    prefix,
    suffix=None
):
    return f"{
        " ".join(
            _mushroom_plot_translate_keywords[keyword]
            if keyword in _mushroom_plot_translate_keywords
            else keyword
            for keyword in prefix.split("_")
        )
    }{
        "" if suffix is None else f" {suffix}"
    }"

def _mushroomplot_colorbar(
    ax,
    extent,
    figsize_scale,
    cmap,
    label,
    text_offset=0.28
):
    ax.add_artist(Text(
        (extent[0] + extent[1]) / 2,
        extent[2] - text_offset,
        label,
        horizontalalignment="center",
        verticalalignment="center",
        size=figsize_scale*3,
        clip_on=False
    ))
    gradient_image(
        ax,
        extent=extent,
        direction=1,
        cmap=cmap,
        clip_on=False
    )
    ax.add_artist(Text(
        extent[0],
        extent[3] + text_offset,
        0,
        horizontalalignment="center",
        verticalalignment="center",
        size=figsize_scale*3,
        clip_on=False
    ))
    ax.add_artist(Text(
        extent[1],
        extent[3] + text_offset,
        1,
        horizontalalignment="center",
        verticalalignment="center",
        size=figsize_scale*3,
        clip_on=False
    ))

def create_mushroom_plot(
    df:pd.DataFrame,
    figsize_scale:float=4,
    top_n:int=30,
    use_absolute_rank:bool=False,
    size_prefix:str="scaled_avg_exprs",
    color_prefix:str="scaled_pval_adapted",
    max_rows:int=20,
):
    '''
    Creates a plot in which each glyph consists of two semicircles corresponding to ligand- and receptor- information.
    The size of the semicircle is the percentage of cells that express the protein, while the saturation corresponds
    to the scaled average expression value.

    Parameters
    ----------
    df : pandas.DataFrame
        the dataframe which contains the required data
    figsize_scale : float
        the size of figure
    top_n : int
        the amount of ligand-receptor pairs to use
    use_absolute_rank : bool
        whether to use the absolute or relative prioritization rank to filter the top_n ligand-receptor pairs
    size_prefix : str
        the prefix of the size column (the suffices are ligand and receptor)
    color_prefix : str
        the prefix of the color column (the suffices are ligand and receptor)
    max_rows : int
        the maximum amount of rows to show

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    
    Returns
    -------
    Figure
        the figure
    Axes
        the axes
    '''
    if type(df) is not pd.DataFrame:
        raise TypeError(f"df should have type pandas.DataFrame, was {type(df)}")
    if not isinstance(figsize_scale, Number):
        raise TypeError(f"figsize_scale should have type float, was {type(figsize_scale)}")
    if type(top_n) is not int:
        raise TypeError(f"top_n should have type int, was {type(top_n)}")
    if type(use_absolute_rank) is not bool:
        raise TypeError(f"use_absolute_rank should have type bool, was {type(use_absolute_rank)}")
    if type(size_prefix) is not str:
        raise TypeError(f"size_prefix should have type str, was {type(size_prefix)}")
    if type(color_prefix) is not str:
        raise TypeError(f"color_prefix should have type str, was {type(color_prefix)}")
    if type(max_rows) is not int:
        raise TypeError(f"max_rows should have type int, was {type(max_rows)}")
    if use_absolute_rank:
        df["show"] = df["prioritization_rank"] <= top_n
    else:
        df.sort_values(by="prioritization_rank", inplace=True)
        df["show"] = list(chain(repeat(True, top_n), repeat(False, df.shape[0] - top_n)))
    interactions = df.drop_duplicates(subset=["ligand", "receptor"])
    if max_rows > interactions.shape[0]:
        max_rows = interactions.shape[0]
    interactions = interactions.iloc[:max_rows]
    interactions = [f"{ligand} - {receptor}" for ligand, receptor in zip(interactions["ligand"], interactions["receptor"])]
    interaction2index = dict(zip(interactions, range(1, len(interactions)+1)))
    senders = sorted(set(df["sender"]))
    sender2index = dict(zip(senders, range(len(senders))))
    xmin = 0
    ymin = 0
    xmax = len(senders) + 1
    ymax = len(interactions) + 1
    fig, ax = plt.subplots(figsize=(figsize_scale, figsize_scale*ymax/xmax))
    ax.set_xlim(
        xmin=xmin,
        xmax=xmax
    )
    ax.set_ylim(
        ymin=ymin,
        ymax=ymax
    )
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("sender celltypes")
    ax.set_ylabel("ligand-receptor interaction")
    xs = np.array(range(1, xmax + 1))
    ys = np.array(range(1, ymax + 1))
    ax.set_xticks(
        ticks=xs[:-1],
        labels=senders
    )
    ax.set_yticks(
        ticks=ys[:-1],
        labels=interactions
    )
    leftout_color = "0.9"
    ax.hlines(ys-0.5, xmin=xmin, xmax=xmax, colors=leftout_color)
    ax.vlines(xs-0.5, ymin=ymin, ymax=ymax, colors=leftout_color)
    ligand_cm = cm["Blues"]
    receptor_cm = cm["Reds"]
    max_wedge_size = 0.49
    for ligand, receptor, sender, size_ligand, size_receptor, color_ligand, color_receptor, rank, show in zip(
        df["ligand"],
        df["receptor"],
        df["sender"],
        df[f"{size_prefix}_ligand"],
        df[f"{size_prefix}_receptor"],
        df[f"{color_prefix}_ligand"],
        df[f"{color_prefix}_receptor"],
        df["prioritization_rank"],
        df["show"]
    ):
        try:
            x = sender2index[sender] + 1
            y = interaction2index[f"{ligand} - {receptor}"]
        except KeyError:
            continue
        wedge_size_ligand = max_wedge_size * sqrt(size_ligand)
        wedge_size_receptor = max_wedge_size * sqrt(size_receptor)
        wedge_color_ligand = ligand_cm(color_ligand)
        wedge_color_receptor = receptor_cm(color_receptor)
        if show:
            ax.add_patch(Wedge(
                (x, y),
                wedge_size_ligand,
                theta1=90,
                theta2=270,
                fc=wedge_color_ligand 
            ))
            ax.add_patch(Wedge(
                (x, y),
                wedge_size_receptor,
                theta1=270,
                theta2=90,
                fc=wedge_color_receptor
            ))
            txt = Text(
                x,
                y,
                int(rank),
                horizontalalignment="center",
                verticalalignment="center",
                size=figsize_scale*3,
                color="white"
            )
            txt.set_path_effects([withStroke(linewidth=1.5, foreground='black')])
            ax.add_artist(txt)
        else:
            ax.add_patch(Wedge(
                (x, y),
                wedge_size_ligand,
                theta1=90,
                theta2=270,
                fc=leftout_color
            ))
            ax.add_patch(Wedge(
                (x, y),
                wedge_size_receptor,
                theta1=270,
                theta2=90,
                fc=leftout_color
            ))
    # legend
    legend_margin = 1.5
    size_legend_width = 8*max_wedge_size
    size_legend_height = 4
    size_legend_center = (
        xmax + legend_margin + size_legend_width / 2,
        1.5
    )
    x_pos = size_legend_center[0] - size_legend_width / 3
    y_pos = size_legend_center[1]
    ax.add_artist(Text(
        x_pos + legend_margin,
        y_pos - 0.7,
        _mushroomplot_label(size_prefix),
        horizontalalignment="center",
        verticalalignment="center",
        size=figsize_scale*3,
        clip_on=False
    ))
    for size in [0.25, 0.5, 0.75, 1.0]:
        ax.add_patch(Wedge(
            (x_pos, y_pos),
            max_wedge_size * sqrt(size),
            theta1=90,
            theta2=270,
            fc="black",
            clip_on=False
        ))
        ax.add_artist(Text(
            x_pos,
            y_pos+1,
            size,
            horizontalalignment="center",
            verticalalignment="center",
            size=figsize_scale*3,
            clip_on=False
        ))
        x_pos += 2*max_wedge_size
    color_legend_width = size_legend_width
    color_legend_height = 1
    ligand_color_legend_center = (
        size_legend_center[0],
        size_legend_center[1] + (size_legend_height + color_legend_height) / 2
    )
    _mushroomplot_colorbar(
        ax,
        extent=[
            ligand_color_legend_center[0] - color_legend_width/2,
            ligand_color_legend_center[0] + color_legend_width/2,
            ligand_color_legend_center[1] - color_legend_height/2,
            ligand_color_legend_center[1] + color_legend_height/2
        ],
        figsize_scale=figsize_scale,
        cmap=ligand_cm,
        label=_mushroomplot_label(color_prefix, "ligand")
    )
    receptor_color_legend_center = (
        ligand_color_legend_center[0],
        ligand_color_legend_center[1] + (size_legend_height + color_legend_height) / 2
    )
    _mushroomplot_colorbar(
        ax,
        extent=[
            receptor_color_legend_center[0] - color_legend_width/2,
            receptor_color_legend_center[0] + color_legend_width/2,
            receptor_color_legend_center[1] - color_legend_height/2,
            receptor_color_legend_center[1] + color_legend_height/2
        ],
        figsize_scale=figsize_scale,
        cmap=receptor_cm,
        label=_mushroomplot_label(color_prefix, "receptor")
    )
    return (fig, ax)
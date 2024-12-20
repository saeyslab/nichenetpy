from nichenetpy.utils import subset_matrix
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.network import WeightedNetwork

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans


def reorder_labels(mat:np.ndarray, row_labels:list[str], col_labels:list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    '''
    Reorders the rows and columns of the matrix along with the corresponding labels based on hierarchic clustering. 

    Parameters
    ----------
    mat : numpy.ndarray
        the matrix to reorder
    row_labels : list or tuple of float
        the row labels
    col_labels : list or tuple of float
        the column labels

    Returns
    -------
    numpy.ndarray
        the reordered matrix
    list[str]
        the reordered row labels
    list[str]
        the reordered column labels
    '''
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
    ligand_target_links:list[tuple[str, str, float]],
    cutoff:float=0.25
) -> tuple[np.ndarray, list[str], list[str]]:
    '''
    Compute the data for a heatmap of regulatory potential. 

    Parameters
    ----------
    predictor : LigandActivityPredictor
        the ligand activity predictor which contains the required ligand-target prior model
    ligand_target_links : list of tuples
        list of (ligand, target, regulatory_potential_scores) tuples
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
    '''
    ligands, targets, weights = zip(*ligand_target_links)
    # TODO: there is most certainly a faster way of doing this
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
    '''
    ligands = sorted(ligand_receptor_links.get_ligands())
    receptors = sorted(ligand_receptor_links.get_receptors())
    ligand2index = dict(zip(ligands, range(len(ligands))))
    receptor2index = dict(zip(receptors, range(len(receptors))))
    mat = np.zeros(shape=(len(ligands), len(receptors)))
    for ligand, receptor, weight in ligand_receptor_links:
        mat[ligand2index[ligand], receptor2index[receptor]] = weight
    return reorder_labels(mat, ligands, receptors)

def heatmap_1d(
    vals:list[float]|np.ndarray,
    labels:list[str],
    title:str=None,
    cbar_label:str=None,
    cmap:str="Greys",
    figsize:tuple[float]=(8, 8)
) -> tuple[Figure, Axes]:
    '''
    Create a 1d heatmap using matplotlib. 

    Parameters
    ----------
    vals : list of float
        the values to plot
    labels : list of str
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
    '''
    fig, ax = plt.subplots(figsize=figsize)
    ys = range(len(labels)+1)
    im = ax.pcolormesh([0, 1], ys, [[val] for val in vals], cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.set_yticks(np.arange(len(labels))+0.5, labels=labels)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    if cbar_label is not None:
        plt.colorbar(im, label=cbar_label)
    return (fig, ax)

def heatmap_2d(
    mat:list[list[float]]|np.ndarray,
    xlabels:list[str],
    ylabels:list[str],
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
    xlabels : list of str
        the labels of the x values
    ylabels : list of str
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
    '''
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
        plt.colorbar(im, fraction=0.05, orientation=cbar_orientation, location=cbar_position, label=cbar_label)
    if xtitle is not None:
        plt.xlabel(xtitle)
    if ytitle is not None:
        plt.ylabel(ytitle)
    plt.hlines([y + 0.5 for y in ys[:-1]], xs[0]-0.5, xs[-1]+0.5, color="white")
    plt.vlines([x + 0.5 for x in xs[:-1]], ys[0]-0.5, ys[-1]+0.5, color="white")
    return (fig, ax)
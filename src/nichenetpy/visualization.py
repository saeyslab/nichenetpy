from nichenetpy.utils import subset_matrix
from nichenetpy.prediction import LigandActivityPredictor

import numpy as np

def prepare_ligand_target_visualization(
    predictor:LigandActivityPredictor,
    ligand_target_links:list[tuple[str, str, float]],
    cutoff:float=0.25
) -> tuple[np.ndarray, list[str], list[str]]:
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
    return (ligand_target_vis, targets, ligands)
    '''
    # TODO: check if these dictionaries are used
    ligand2index = dict(zip(ligands, range(len(ligands))))
    target2index = dict(zip(targets, range(len(targets))))
    nrows, ncols = ligand_target_vis.shape
    if nrows > 1 and ncols > 1:
        #corr = np.corrcoef(np.transpose(ligand_target_vis))
        corr = np.corrcoef(ligand_target_vis, rowvar=False)
        nrows, ncols = corr.shape
        corr = 1 - corr
        dist = sc.spatial.distance_matrix(corr, corr)
        clust = sc.cluster.hierarchy.ward(sc.spatial.distance.squareform(dist))
        '''
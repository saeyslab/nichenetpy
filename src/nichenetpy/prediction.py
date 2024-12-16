from nichenetpy.metrics import calculate_metrics
from nichenetpy.utils import subset_matrix

import numpy as np


class LigandActivityPredictor:
    def __init__(
        self,
        ligand_target_matrix:np.ndarray,
        row_names:list[str],
        col_names:list[str],
    ) -> None:
        self.ligand_target_matrix = ligand_target_matrix
        self.row_names = row_names
        self.col_names = col_names
        self.ligand2index = dict(zip(self.col_names, range(len(self.col_names))))
        self.gene2index = dict(zip(self.row_names, range(len(self.row_names))))

    def predict_ligand_activities(
        self,
        geneset:list[str],
        background_expressed_genes:list[str],
        potential_ligands:list[str]
    ) -> dict[str, dict[str, float]]:
        output = dict()

        # create the expected gene expression response vector
        response = dict((gene, 0) for gene in background_expressed_genes if gene not in geneset)
        for gene in geneset:
            response[gene] = 1
        
        # create the prediction model vector
        predictions = ([
            dict(zip(self.row_names, self.ligand_target_matrix[:, self.ligand2index[ligand]]))
            for ligand in potential_ligands
        ])

        # compute the metrics for each ligand
        for ligand, prediction in zip(potential_ligands, predictions):
            # we need to match the predictions with the responses so we intersect and sort by key
            common_keys = prediction.keys() & response.keys()
            pred = [tup[1] for tup in sorted(((key, prediction[key]) for key in common_keys), key=lambda x : x[0])]
            resp = [tup[1] for tup in sorted(((key, response[key]) for key in common_keys), key=lambda x : x[0])]
            output[ligand] = calculate_metrics(pred, resp)
        return output
    
    def get_weighted_ligand_target_links(self, ligand:str, geneset:set[str], n:int=250) -> dict[str, list]:
        targets = set(
            e[0] for e in sorted(
                zip(self.row_names, self.ligand_target_matrix[:, self.ligand2index[ligand]]),
                key=lambda x : x[1],
                reverse=True
            )[:n]
        ).intersection(geneset)
        if len(targets) == 0:
            return {
                "ligand": ligand,
                "target": None,
                "weight": None
            }
        else:
            return {
                "ligand": ligand,
                "target": targets,
                "weight": [self.ligand_target_matrix[self.gene2index[target]][self.ligand2index[ligand] ]for target in targets]
            }
    
    def prepare_ligand_target_visualization(self, ligand_target_links:list[tuple[str, str, float]], cutoff:float=0.25):
        ligands, targets, weights = zip(*ligand_target_links)
        # TODO: there is most certainly a faster way of doing this
        ligands = sorted(set(ligands))
        targets = sorted(set(targets))
        # select ligands and targets that appear in ligand_target_links
        ligand_target_vis = subset_matrix(
            self.ligand_target_matrix,
            [self.gene2index[target] for target in targets],
            [self.ligand2index[ligand] for ligand in ligands]
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
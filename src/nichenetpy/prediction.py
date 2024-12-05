from nichenetpy.metrics import calculate_metrics
from scipy.sparse import hstack
from anndata import AnnData
import numpy as np

class LigandActivityPredictor:
    def __init__(
        self,
        ligand_target_matrix:np.ndarray,
        row_names:list[str],
        col_names:list[str],
        ligands_position:str="cols"
    ) -> None:
        self.ligand_target_matrix = ligand_target_matrix
        self.ligands_position = ligands_position
        self.row_names = row_names
        self.col_names = col_names
        if self.ligands_position == "cols":
            self.ligand2index = dict(zip(self.col_names, range(len(self.col_names))))
            self.gene2index = dict(zip(self.row_names, range(len(self.row_names))))
        else:
            self.ligand2index = dict(zip(self.row_names, range(len(self.row_names))))
            self.gene2index = dict(zip(self.col_names, range(len(self.col_names))))

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
            ] if self.ligands_position == "cols" else [
                dict(zip(self.col_names, self.ligand_target_matrix[self.ligand2index[ligand], :]))
                for ligand in potential_ligands
            ]
        )

        # compute the metrics for each ligand
        for ligand, prediction in zip(potential_ligands, predictions):
            # we need to match the predictions with the responses so we intersect and sort by key
            common_keys = prediction.keys() & response.keys()
            pred = [tup[1] for tup in sorted(((key, prediction[key]) for key in common_keys), key=lambda x : x[0])]
            resp = [tup[1] for tup in sorted(((key, response[key]) for key in common_keys), key=lambda x : x[0])]
            output[ligand] = calculate_metrics(pred, resp)
        return output
        
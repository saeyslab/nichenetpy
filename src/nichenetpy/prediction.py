from nichenetpy.metrics import calculate_metrics
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
            self.ligand2index = dict(zip(self.col_names, len(self.col_names)))
            self.gene2index = dict(zip(self.row_names, len(self.row_names)))
        else:
            self.ligand2index = dict(zip(self.row_names, len(self.row_names)))
            self.gene2index = dict(zip(self.col_names, len(self.col_names)))

    def predict_ligand_activities(
        self,
        geneset:list[str],
        background_expressed_genes:list[str],
        potential_ligands:list[str]
    ):
        output = []
        response = [(gene, False) for gene in background_expressed_genes if gene not in geneset] + [(gene, True) for gene in geneset]
        response.sort(key=lambda x : x[0])
        if self.ligands_position == "cols":
            for ligand in potential_ligands:
                prediction = sorted(zip(self.row_names, self.ligand_target_matrix[:, self.ligand2index[ligand]]), key=lambda x : x[0])
                output.append((ligand, calculate_metrics(prediction, response)))
        else:
            for ligand in potential_ligands:
                prediction = sorted(zip(self.col_names, self.ligand_target_matrix[self.ligand2index[ligand], :]), key=lambda x : x[0])
        return output
        
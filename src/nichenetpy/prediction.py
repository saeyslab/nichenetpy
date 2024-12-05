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

class LigandReceptorNetwork:
    def __init__(self, filename:str=None) -> None:
        if filename is not None:
            with open(filename) as file:
                lines = file.readlines()
            self._mapping = sorted(
                (tuple(word.strip("\"\'") for word in line.rstrip().split(",")) for line in lines[1:]),
                key=lambda x : x[0]
            )
        self._index = dict()
        for i, item in enumerate(self._mapping):
            if item[0] in self._index:
                self._index[item[0]][1] += 1
            else:
                self._index[item[0]] = [i, 1]
    
    def __str__(self) -> str:
        return self._mapping.__str__()

    def __getitem__(self, key:str) -> list[str]:
        start, count = self._index[key]
        return set(item[1] for item in self._mapping[start:start+count])
    
    def __iter__(self):
        return self._mapping.__iter__()

    def key_iter(self):
        return (self._mapping[start][0] for start, _ in self._index.values())

    def item_iter(self):
        return ((key, self[key]) for key in self.key_iter())
    
    def get_ligands(self):
        return set(key_iter)
    
    def get_receptors(self):
        return set(receptor for _, receptor in self._mapping)
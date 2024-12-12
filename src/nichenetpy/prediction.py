from nichenetpy.metrics import calculate_metrics
import numpy as np
import scipy as sc


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
        # define a cutoff on the ligand-target links
        cutoff = np.quantile(weights, [cutoff])[0]
        nrows, ncols = self.ligand_target_matrix.shape
        ligand_target_matrix_oi = np.array([
            [self.ligand_target_matrix[r, c] if self.ligand_target_matrix[r, c] >= cutoff else 0 for c in range(ncols)]
            for r in range(nrows)
        ])
        # TODO: there is most certainly a faster way of doing this
        ligands = sorted(set(ligands))
        targets = sorted(set(targets))
        # keep only rows and columns that contain at least one non-zero element
        ligands = [ligand for ligand in ligands if any(ligand_target_matrix_oi[:, self.ligand2index[ligand]])]
        targets = [target for target in targets if any(ligand_target_matrix_oi[self.gene2index[target], :])]
        ligand_target_vis = ligand_target_matrix_oi[
            [[self.gene2index[target]] for target in targets],
            [self.ligand2index[ligand] for ligand in ligands]
        ]
        return (ligand_target_matrix_oi, ligand_target_vis)
        nrows, ncols = ligand_target_vis.shape
        if nrows > 1 and ncols > 1:
            #corr = np.corrcoef(np.transpose(ligand_target_vis))
            corr = np.corrcoef(ligand_target_vis, rowvar=False)
            nrows, ncols = corr.shape
            corr = np.array([
                [1 - corr[r, c] for c in range(ncols)]
                for r in range(nrows)
            ])
            dist = sc.spatial.distance_matrix(corr, corr)


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
        return set(self.key_iter())
    
    def get_receptors(self):
        return set(receptor for _, receptor in self._mapping)
from nichenetpy.prediction import LigandActivityPredictor
from nichenetpy.network import (
    Network,
    WeightedNetwork,
    LigandReceptorNetwork
)
from nichenetpy.typing import (
    nichenet_matrix,
    gene_t
)

from itertools import chain
from collections.abc import (
    Iterable,
    Collection
)

import pandas as pd
import numpy as np


class NicheNet:
    '''
    This class combines the components of NicheNet and internally maps gene symbols to integers to save memory

    Parameters
    ----------
    predictor : LigandActivityPredictor
        the ligand activity predictor (wrapper for ligand-target matrix)
    lr_network : pandas.DataFrame
        the ligand-receptor network
    weighted_networks : dict
        the weighted networks
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    
    Attributes
    ----------
    predictor : LigandActivityPredictor
        the ligand activity predictor (wrapper for ligand-target matrix)
    lr_network : LigandReceptorNetwork
        the ligand-receptor network
    lr_sig : WeightedNetwork
        the ligand-receptor-signaling weighted network
    gr : WeightedNetwork
        gene regulatory weighted network
    
    NOTES
    -----
    This class introduces significant overhead and uses more memory, don't use it
    '''
    def __init__(
        self,
        predictor:LigandActivityPredictor,
        lr_network:pd.DataFrame,
        weighted_networks:dict[str, pd.DataFrame]
    ) -> None:
        if type(predictor) is not LigandActivityPredictor:
            raise TypeError(f"predictor should have type LigandActivityPredictor, was {type(predictor)}")
        if type(weighted_networks) is not dict:
            raise TypeError(f"weighted_networks should have type dict, was {type(weighted_networks)}")
        # do not change the input
        lr_network = lr_network.copy()
        lr_sig = weighted_networks["lr_sig"].copy()
        gr = weighted_networks["gr"].copy()
        self._syms = sorted(set(chain(
            lr_network["from"],
            lr_network["to"],
            lr_sig["from"],
            lr_sig["to"],
            gr["from"],
            gr["to"],
            predictor.row_names,
            predictor.col_names
        )))
        self._sym2id = dict(zip(self._syms, range(len(self._syms))))
        lr_network["from"] = [self._sym2id[e] for e in lr_network["from"]]
        lr_network["to"] = [self._sym2id[e] for e in lr_network["to"]]
        lr_sig["from"] = [self._sym2id[e] for e in lr_sig["from"]]
        lr_sig["to"] = [self._sym2id[e] for e in lr_sig["to"]]
        gr["from"] = [self._sym2id[e] for e in gr["from"]]
        gr["to"] = [self._sym2id[e] for e in gr["to"]]
        self.predictor = NicheNet._LigandActivityPredictor(
            ligand_target_matrix=predictor.ligand_target_matrix.copy(order="F"),
            row_names=tuple(self._sym2id[e] for e in predictor.row_names),
            col_names=tuple(self._sym2id[e] for e in predictor.col_names),
            model=self
        )
        self.lr_network = NicheNet._LigandReceptorNetwork(
            mapping=lr_network,
            model=self
        )
        self.lr_sig = NicheNet._WeightedNetwork(
            lr_sig,
            model=self
        )
        self.gr = NicheNet._WeightedNetwork(
            gr,
            model=self
        )
    
    class _LigandActivityPredictor(LigandActivityPredictor):
        def __init__(
            self,
            ligand_target_matrix:nichenet_matrix,
            row_names:list[gene_t]|tuple[gene_t],
            col_names:list[gene_t]|tuple[gene_t],
            model
        ):
            super().__init__(
                ligand_target_matrix,
                row_names,
                col_names
            )
            self._model = model
        
        def _sym2id(self, sym:gene_t) -> int:
            # if sym is an integer it doesn't need to be converted
            return self._model._sym2id[sym] if type(sym) is str else sym

        def _id2sym(self, id:int) -> str:
            return self._model._syms[id]
        
        def ligand2index(self, ligand:gene_t):
            return super().ligand2index(self._sym2id(ligand))

        def gene2index(self, gene:gene_t):
            return super().gene2index(self._sym2id(gene))
        
        def get_ligands(self):
            return {self._id2sym(e) for e in self._ligand2index.keys()}
        
        def get_genes(self):
            return {self._id2sym(e) for e in self._gene2index.keys()}
        
        def get_col(
            self,
            index:int|gene_t,
            is_index:bool=True
        ):
            return super().get_col(self._sym2id(index), is_index)
        
        def evaluate_target_prediction(
            self,
            ligand:gene_t,
            response:dict[gene_t, int]
        ):
            return super().evaluate_target_prediction(
                self._sym2id(ligand),
                {
                    self._sym2id(k): v
                    for k, v in response.items()
                    if k in self._model._syms
                }
            )

        def predict_ligand_activities(
            self,
            geneset:Collection[gene_t],
            background_expressed_genes:Iterable[gene_t],
            potential_ligands:Iterable[gene_t]
        ):
            return {
                self._id2sym(k): v
                for k, v in super().predict_ligand_activities(
                    {self._sym2id(e) for e in geneset},
                    (self._sym2id(e) for e in background_expressed_genes),
                    (self._sym2id(e) for e in potential_ligands)
                ).items()
            }
        
        def predict_single_cell_ligand_activities(
            self,
            cells:Collection[str],
            expression_scaled:np.ndarray,
            expression_scaled_rows:Iterable[str],
            expression_scaled_cols:Iterable[gene_t],
            potential_ligands:Collection[gene_t],
            quantile_cutoff:float=0.975,
            calc_aupr:bool=True,
            calc_auroc:bool=True,
            calc_pearson:bool=True
        ):
            output = super().predict_single_cell_ligand_activities(
                cells,
                expression_scaled,
                expression_scaled_rows,
                (self._sym2id(e) for e in expression_scaled_cols),
                {self._sym2id(e) for e in potential_ligands},
                quantile_cutoff,
                calc_aupr,
                calc_auroc,
                calc_pearson
            )
            output["ligand"] = [self._id2sym(e) for e in output["ligand"]]
            return output
        
        def get_weighted_ligand_target_links(
            self,
            ligand:gene_t,
            geneset:Iterable[gene_t],
            n:int=250
        ):
            output = super().get_weighted_ligand_target_links(
                self._sym2id(ligand),
                (self._sym2id(e) for e in geneset),
                n
            )
            output["ligand"] = self._id2sym(output["ligand"])
            output["targets"] = [self._id2sym(e) for e in output["targets"]]
            return output

        def gene_presence(self, genes:Iterable[gene_t]):
            return super().gene_presence((self._sym2id(e) for e in genes))
        
        def ligand_presence(self, ligands:Iterable[gene_t]):
            return super().ligand_presence((self._sym2id(e) for e in ligands))
    
    class _Network(Network):
        def __init__(
            self,
            mapping:list|pd.DataFrame|None=None,
            filename:str|None=None,
            model=None
        ):
            super().__init__(mapping, filename)
            self._model = model
        
        def _sym2id(self, sym:gene_t) -> int:
            # if sym is an integer it doesn't need to be converted
            return self._model._sym2id[sym] if type(sym) is str else sym

        def _id2sym(self, id:int) -> str:
            return self._model._syms[id]
        
        def __iter__(self):
            return ((self._id2sym(fr), self._id2sym(to)) for fr, to in super().__iter__())
        
        def __contains__(self, item:tuple):
            return super().contains((self._sym2id(item[0]), self._sym2id(item[1])))
        
        def key_iter(self):
            return (self._id2sym(e) for e in super().key_iter())
        
        def item_iter(self):
            return ((self._id2sym(fr), self._id2sym(to)) for fr, to in super().item_iter())
        
        def mapping_iter(self, key):
            return (self._id2sym(e) for e in super().mapping_iter(self._sym2id(key)))
        
        def get_all(self) -> set:
            return set((self._id2sym(fr), self._id2sym(to)) for fr, to in chain(*zip(*self._mapping)))
        
        def subset(self, from_to:Collection[tuple[gene_t, gene_t]]):
            if not isinstance(from_to, Collection):
                raise TypeError(f"from_to should be a Collection of tuple[gene_t, gene_t], was {type(from_to)}")
            from_to = tuple((self._sym2id(fr), self._sym2id(to)) for fr, to in from_to)
            return type(self)(
                mapping=[tup for tup in self._mapping if (tup[0], tup[1]) in from_to],
                model=self._model
            )

        def subset_sep(self, fr:Collection[gene_t]|None=None, to:Collection[gene_t]|None=None):
            if fr is None:
                fr = set(self.key_iter())
            else:
                fr = {self._sym2id(e) for e in fr}
            if to is None:
                to = set(e for _, e in self._mapping)
            else:
                to = {self._sym2id(e) for e in to}
            if not isinstance(fr, Collection):
                raise TypeError(f"fr should be a Collection of gene_t, was {type(fr)}")
            if not isinstance(to, Collection):
                raise TypeError(f"to should be a Collection of gene_t, was {type(to)}")
            return type(self)(
                mapping=[tup for tup in self._mapping if tup[0] in fr and tup[1] in to],
                model=self._model
            )

    
    class _LigandReceptorNetwork(_Network, LigandReceptorNetwork):
        def get_receptors(self):
            return {self._id2sym(e) for e in super().get_receptors()}
    
    class _WeightedNetwork(_Network, WeightedNetwork):
        def __init__(
            self,
            mapping:list|pd.DataFrame|None=None,
            filename:str|None=None,
            model=None
        ):
            super().__init__(mapping, filename, model)
        
        def _sym2id(self, sym:gene_t) -> int:
            # if sym is an integer it doesn't need to be converted
            return self._model._sym2id[sym] if type(sym) is str else sym

        def _id2sym(self, id:int) -> str:
            return self._model._syms[id]
        
        def __getitem__(self, key:gene_t):
            return {
                self._id2sym(k): v
                for k, v in super().__get_item__(self._sym2id(key)).items()
            }
        
        def get_receptors(self):
            return {self._id2sym(e) for e in super().get_receptors()}
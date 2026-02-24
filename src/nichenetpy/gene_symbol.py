from nichenetpy.network import Network

from anndata import AnnData
from collections.abc import Iterable

import os


_root = os.path.dirname(__file__)

class GeneAliasInfo:
    '''
    This class facilitates gene alias conversion. 

    Parameters
    ----------
    arg : str or dict
        name of the file to read gene alias information from or a dict which contains the alias mappings: alias -> (symbol, entrez)
    
    Raises
    ------
    TypeError
        if arg is not of the correct type
    '''
    def __init__(self, arg:str|dict[str, tuple[str, int|None]]) -> None:
        if type(arg) is str:
            with open(arg) as file:
                lines = file.readlines()
            symbol, entrez, alias = zip(*([word.strip("\"\'") for word in line.rstrip().split(",")] for line in lines[1:]))
            self._mapping = dict(zip(alias, zip(symbol, (None if e == "" else int(e) for e in entrez))))
        elif type(arg) is dict:
            self._mapping = arg
        else:
            raise TypeError(f"arg should have type str or dict, was {type(arg)}")
    
    def __str__(self) -> str:
        return self._mapping.__str__()

    def __getitem__(self, key:str) -> tuple[str, int|None]:
        try:
            return self._mapping[key]
        except KeyError:
            raise KeyError(f"{key} is not a recognized gene")
    
    def __iter__(self):
        return self._mapping.__iter__()
    
    def __contains__(self, item):
        return item in self._mapping
    
    def add(self, items:Iterable[tuple[str, tuple[str, int|None]]]):
        '''
        Adds new mappings but will not overwrite existing mappings. 

        Parameters
        ----------
        items : Iterable
            the new mappings
        '''
        for k, v in items:
            if k not in self:
                self._mapping[k] = v
    
    def alias_to_symbol(self, obj:Iterable[str]|AnnData) -> list[str]|None:
        '''
        Converts gene aliases to gene symbols. 

        Parameters
        ----------
        obj : Iterable or AnnData or None
            an object that contains gene symbols
        
        Returns
        -------
        list of str or None
            a list of gene names post gene alias conversion
        
        Raises
        ------
        TypeError
            if obj is not of the correct type
        
        Notes
        -----
        if obj is of type Iterable[str], this function returns a list of gene symbols
        if obj is of type AnnData, this function returns None
        '''
        if type(obj) is list or type(obj) is tuple:
            output = []
            mapped_genes = dict()
            for gene in obj:
                ngene = self[gene][0] if gene in self else gene
                output.append(ngene)
                if ngene in mapped_genes:
                    mapped_genes[ngene].add(gene)
                else:
                    mapped_genes[ngene] = {gene}
            # revert duplicate symbols back to the original symbols
            doubles = [i for i, ngene in enumerate(output) if len(mapped_genes[ngene]) > 1]
            for i in doubles:
                output[i] = obj[i]
            return output
        elif type(obj) is AnnData:
            obj.var_names = self.alias_to_symbol(obj.var_names)
            if "gene" in obj.var:
                obj.var["gene"] = self.alias_to_symbol(obj.var["gene"])
        elif isinstance(obj, Iterable):
            return self.alias_to_symbol(list(obj))
        else:
            raise TypeError(f"expected type of obj argument to be Iterable[str] or AnnData, got {type(obj)}")

mouse_alias_info = GeneAliasInfo(os.path.join(_root, "../../data/gene_info/geneinfo_alias_mouse.csv"))
'''
gene alias info for mice
'''

human_alias_info = GeneAliasInfo(os.path.join(_root, "../../data/gene_info/geneinfo_alias_human.csv"))
'''
gene alias info for humans
'''

class GeneInfo:
    '''
    This class facilitates gene conversion between mouse and human symbols. 

    Parameters
    ----------
    arg : str or Iterable
        name of the file to read gene alias information from or an Iterable of (human_symbol, mouse_symbol) tuples
    
    Raises
    ------
    TypeError
        if arg is not of the correct type
    '''
    def __init__(self, arg:str|Iterable[tuple[str, str]]):
        if type(arg) is str:
            with open(arg) as file:
                lines = file.readlines()
            symbol, _, _, symbol_mouse = zip(*([word.strip("\"\'") for word in line.rstrip().split(",")] for line in lines[1:]))
            self.__init__(list(zip(symbol, symbol_mouse)))
        elif isinstance(arg, Iterable):
            # one to many
            self._human2mouse = Network(
                sorted(
                    ((sh, sm) for sh, sm in arg if sh != "NA" and sm != "NA"),
                    key=lambda x : x[0]
                )
            )
            # one to one
            self._mouse2human = dict(((sm, sh) for sh, sm in arg))
        else:
            raise TypeError(f"arg should have type str or dict, was {type(arg)}")

    def __getitem__(self, key:str) -> tuple[str, str]:
        if key in self._human2mouse._index:
            return self._human2mouse[key]
        elif key in self._mouse2human._index:
            return self._mouse2human[key]
        else:
            raise KeyError(f"{key} is not a recognized gene")
    
    def __contains__(self, item):
        return item in self._human2mouse._index or item in self._mouse2human._index
    
    def convert_human_to_mouse_symbols(self, symbols:Iterable[str]) -> Iterable[str]:
        '''
        Converts human gene symbols to their mouse one-to-one orthologs

        Parameters
        ----------
        symbols : Iterable of str
            the human gene symbols to convert
        
        Returns
        -------
        Iterable of str
            the mouse gene symbols
        
        Raises
        ------
        KeyError
            if a gene symbol isn't recognized
        '''
        # this does the same as the NicheNetR equivalent but it makes no sense to me (Victor)
        for symbol in symbols:
            if symbol in self._human2mouse._index:
                mapping = self._human2mouse.mapping_iter(symbol)
                yield next(mapping)
            else:
                yield None
    
    def convert_mouse_to_human_symbols(self, symbols:Iterable[str]) -> Iterable[str]:
        '''
        Converts mouse gene symbols to their human one-to-one orthologs

        Parameters
        ----------
        symbols : Iterable of str
            the mouse gene symbols to convert
        
        Returns
        -------
        Iterable of str
            the human gene symbols
        
        Raises
        ------
        KeyError
            if a gene symbol isn't recognized
        '''
        for symbol in symbols:
            try:
                yield self._mouse2human[symbol]
            except KeyError:
                yield None

gene_info = GeneInfo(os.path.join(_root, "../../data/gene_info/geneinfo.csv"))
'''
mapper between mouse and human symbols
'''
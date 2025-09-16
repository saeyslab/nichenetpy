from nichenetpy.network import Network

from anndata import AnnData
from collections.abc import Iterable
from itertools import chain

import os


_root = os.path.dirname(__file__)

class GeneAliasInfo:
    '''
    This class facilitates gene alias conversion. 

    Parameters
    ----------
    filename : str
        name of the file to read gene alias information from
    
    Raises
    ------
    TypeError
        if filename is not of the correct type
    '''
    def __init__(self, filename:str) -> None:
        if type(filename) is not str:
            raise TypeError(f"filename should have type str, was {type(filename)}")
        with open(filename) as file:
            lines = file.readlines()
        symbol, entrez, alias = zip(*([word.strip("\"\'") for word in line.rstrip().split(",")] for line in lines[1:]))
        self._mapping = dict(zip(alias, zip(symbol, (int(e) for e in entrez))))
    
    def __str__(self) -> str:
        return self._mapping.__str__()

    def __getitem__(self, key:str) -> tuple[str, str]:
        try:
            return self._mapping[key]
        except KeyError:
            raise KeyError(f"{key} is not a recognized gene")
    
    def __iter__(self):
        return self._mapping.__iter__()
    
    def __contains__(self, item):
        return item in self._mapping
    
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
        if type(obj) is list:
            output = []
            counts = dict()
            for gene in obj:
                ngene = self[gene][0] if gene in self else gene
                output.append(ngene)
                if ngene in counts:
                    counts[ngene] += 1
                else:
                    counts[ngene] = 1
            # revert duplicate symbols back to the original symbols
            doubles = [i for i, ngene in enumerate(output) if counts[ngene] > 1]
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
    filename : str
        name of the file to read gene alias information from
    
    Raises
    ------
    TypeError
        if filename is not of the correct type
    '''
    def __init__(self, filename:str):
        if type(filename) is not str:
            raise TypeError(f"filename should have type str, was {type(filename)}")
        with open(filename) as file:
            lines = file.readlines()
        symbol, _, _, symbol_mouse = zip(*([word.strip("\"\'") for word in line.rstrip().split(",")] for line in lines[1:]))
        entries = [(sh, sm) for sh, sm in zip(symbol, symbol_mouse) if sh != "NA" and sm != "NA"]
        self._human2mouse = Network(
            sorted(
                entries,
                key=lambda x : x[0]
            )
        )
        self._mouse2human = Network(
            sorted(
                ((sm, sh) for sh, sm in entries),
                key=lambda x : x[0]
            )
        )

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
        for symbol in symbols:
            if symbol in self._human2mouse._index:
                mapping = self._human2mouse[symbol]
                if len(mapping) > 1:
                    lwr = f"{symbol[0]}{symbol[1:].lower()}"
                    if lwr in mapping:
                        yield lwr
                    else:
                        yield None
                else:
                    yield next(iter(mapping))
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
            if symbol in self._mouse2human._index:
                mapping = self._mouse2human[symbol]
                if len(mapping) > 1:
                    upr = symbol.upper()
                    if upr in mapping:
                        yield upr
                    else:
                        yield None
                else:
                    yield next(iter(mapping))
            else:
                yield None

gene_info = GeneInfo(os.path.join(_root, "../../data/gene_info/geneinfo.csv"))
'''
mapper between mouse and human symbols
'''
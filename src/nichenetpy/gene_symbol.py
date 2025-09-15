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
        symbol = [None if e == "NA" else e for e in symbol]
        symbol_mouse = [None if e == "NA" else e for e in symbol_mouse]
        count_mouse = dict()
        for sm in symbol_mouse:
            if sm in count_mouse:
                count_mouse[sm] += 1
            else:
                count_mouse[sm] = 1
        unambiguous_mouse_genes = []
        ambiguous_mouse_genes = []
        for i, sm in enumerate(symbol_mouse):
            if sm is not None:
                (ambiguous_mouse_genes if count_mouse[sm] > 1 else unambiguous_mouse_genes).append(i)
        print([(symbol[i], symbol_mouse[i]) for i in unambiguous_mouse_genes if symbol[i] == "BRCC3"])
        print([(symbol[i], symbol_mouse[i]) for i in ambiguous_mouse_genes if symbol[i] == "BRCC3"])
        self._human2mouse = dict(
            (symbol[i], symbol_mouse[i])
            for i in chain(unambiguous_mouse_genes, (j for j in ambiguous_mouse_genes if symbol[j] == symbol_mouse[j].upper()))
        )
        self._mouse2human = dict((v, k) for k, v in self._human2mouse.items())

    def __getitem__(self, key:str) -> tuple[str, str]:
        if key in self._human2mouse:
            return self._human2mouse[key]
        elif key in self._mouse2human:
            return self._mouse2human[key]
        else:
            raise KeyError(f"{key} is not a recognized gene")
    
    def __contains__(self, item):
        return item in self._human2mouse or item in self._mouse2human
    
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
            try:
                yield self._human2mouse[symbol]
            except KeyError:
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
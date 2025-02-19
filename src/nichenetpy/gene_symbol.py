import os
from anndata import AnnData
from collections.abc import Iterable


root = os.path.dirname(__file__)

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
        return self._mapping[key]
    
    def __iter__(self):
        return self._mapping.__iter__()
    
    def __contains__(self, item):
        return item in self._mapping
    
    def alias_to_symbol(self, obj:Iterable[str]|AnnData) -> list[str]|None:
        '''
        Converts gene aliases to gene symbols. 

        Parameters
        ----------
        obj : Iterable or AnnData
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
            doubles = [i for i, ngene in enumerate(output) if counts[ngene] > 1]
            for i in doubles:
                output[i] = obj[i]
            return output
        elif type(obj) is AnnData:
            obj.var["gene"] = self.alias_to_symbol(obj.var["gene"])
        elif isinstance(obj, Iterable):
            return self.alias_to_symbol(list(obj))
        else:
            raise TypeError(f"expected type of obj argument to be Iterable[str] or AnnData, got {type(obj)}")

mouse_alias_info = GeneAliasInfo(os.path.join(root, "../../data/gene_alias/geneinfo_alias_mouse.csv"))
'''
gene alias info for mice (3845 Mb)
'''

human_alias_info = GeneAliasInfo(os.path.join(root, "../../data/gene_alias/geneinfo_alias_human.csv"))
'''
gene alias info for humans (3845 Mb)
'''
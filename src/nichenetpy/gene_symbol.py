import os
from anndata import AnnData
from collections.abc import Iterable

root = os.path.dirname(__file__)

class GeneAliasInfo:
    def __init__(self, filename:str) -> None:
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
            raise ValueError(f"expected type of obj argument to be Iterable[str] or AnnData, got {type(obj)}")

mouse_alias_info = GeneAliasInfo(os.path.join(root, "../../data/gene_alias/geneinfo_alias_mouse.csv"))
human_alias_info = GeneAliasInfo(os.path.join(root, "../../data/gene_alias/geneinfo_alias_human.csv"))
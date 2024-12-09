import os

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

mouse_alias_info = GeneAliasInfo(os.path.join(root, "../../data/gene_alias/geneinfo_alias_mouse.csv"))
human_alias_info = GeneAliasInfo(os.path.join(root, "../../data/gene_alias/geneinfo_alias_human.csv"))
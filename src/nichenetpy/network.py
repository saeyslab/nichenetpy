
from collections.abc import Collection
class Network:
    def __init__(self, mapping:list=None, filename:str=None) -> None:
        if mapping is not None:
            self._mapping = mapping
        elif filename is not None:
            with open(filename) as file:
                lines = file.readlines()
            self._mapping = sorted(
                (tuple(word.strip("\"\'") for word in line.rstrip().split(",")) for line in lines[1:]),
                key=lambda x : x[0]
            )
        else:
            raise ValueError("either mapping or filename must be provided as arguments")
        self._index = dict()
        self._build_index()
    
    def __str__(self) -> str:
        return self._mapping.__str__()

    def __getitem__(self, key:str) -> list[str]:
        start, count = self._index[key]
        return set(item[1] for item in self._mapping[start:start+count])
    
    def __iter__(self):
        return self._mapping.__iter__()
    
    def __contains__(self, item):
        return item in self._mapping
    
    def _build_index(self):
        self._index.clear()
        for i, item in enumerate(self._mapping):
            if item[0] in self._index:
                self._index[item[0]][1] += 1
            else:
                self._index[item[0]] = [i, 1]

    def key_iter(self):
        return (self._mapping[start][0] for start, _ in self._index.values())

    def item_iter(self):
        return ((key, self[key]) for key in self.key_iter())

class LigandReceptorNetwork(Network):
    def get_ligands(self) -> set[str]:
        return set(self.key_iter())
    
    def get_receptors(self) -> set[str]:
        return set(receptor for _, receptor in self._mapping)

class WeightedNetwork(Network):
    def subset(self, from_to:Collection[tuple[str, str]]):
        return WeightedNetwork(mapping=[(f, t, w) for f, t, w in self._mapping if (f, t) in from_to])

    def subset_sep(self, fr:Collection[str], to:Collection[str]):
        return WeightedNetwork(mapping=[(f, t, w) for f, t, w in self._mapping if f in fr and t in to])
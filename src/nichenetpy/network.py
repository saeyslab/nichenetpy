
from collections.abc import Collection


class Network:
    '''
    A network with links. This is in essence a one to many mapping. 

    Parameters
    ----------
    mapping : list
        sorted list of tuples (from, to)
    filename : str
        name of the file to read the network from
    
    Notes
    -----
    You must pass a list SORTED by "from" as mapping or the name of a file to read from. 
    '''
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
        start, count = self._index[item[0]]
        return item in self._mapping[start:start+count]
    
    def _build_index(self):
        self._index.clear()
        for i, item in enumerate(self._mapping):
            if item[0] in self._index:
                self._index[item[0]][1] += 1
            else:
                self._index[item[0]] = [i, 1]

    def key_iter(self):
        '''
        Iterates over the "from" values in the mapping. 
        
        Yields
        ------
        str
            the "from" values in the mapping
        '''
        return (self._mapping[start][0] for start, _ in self._index.values())

    def item_iter(self):
        '''
        Iterates over the "from" values in the mapping along with the corresponding "to" values. 
        
        Yields
        ------
        tuple
            the "from" values in the mapping and a list of corresponding "to" values
        '''
        return ((key, self[key]) for key in self.key_iter())

class LigandReceptorNetwork(Network):
    '''
    A ligand-receptor network without weights. This is in essence a one to many mapping. 

    Parameters
    ----------
    mapping : list
        sorted list of tuples (from, to)
    filename : str
        name of the file to read the network from
    
    Notes
    -----
    You must pass a list SORTED by "from" as mapping or the name of a file to read from. 
    '''
    def get_ligands(self) -> set[str]:
        '''
        Get all the ligands present in the network. 
        
        Returns
        -------
        set
            set of ligands present in the network
        '''
        return set(self.key_iter())
    
    def get_receptors(self) -> set[str]:
        '''
        Get all the receptors present in the network. 
        
        Returns
        -------
        set
            set of receptors present in the network
        '''
        return set(receptor for _, receptor in self._mapping)

class WeightedNetwork(Network):
    '''
    A ligand-receptor network with weighted links. This is in essence a one to many mapping. 

    Parameters
    ----------
    mapping : list
        sorted list of tuples (from, to, weight)
    filename : str
        name of the file to read the network from
    '''
    def __init__(self, mapping = None, filename = None):
        super().__init__(mapping, filename)
        if filename is not None:
            # the weigths are read as strings and must be converted to floats
            self._mapping = [(l, r, float(w)) for l, r, w in self._mapping]
    
    def __getitem__(self, key:str) -> dict[str, float]:
        start, count = self._index[key]
        return dict(item[1:3] for item in self._mapping[start:start+count])

    def subset(self, from_to:Collection[tuple[str, str]]):
        '''
        Subset the network by the provided links. 

        Parameters
        ----------
        from_to : Collection
            collection of tuples (from, to) to subset by

        Returns
        -------
        WeightedNetwork
            the subsetted network
        '''
        return WeightedNetwork(mapping=[(f, t, w) for f, t, w in self._mapping if (f, t) in from_to])

    def subset_sep(self, fr:Collection[str], to:Collection[str]):
        '''
        Subset the network by the provided "from" and "to" values. 

        Parameters
        ----------
        from : Collection
            collection of "from" values to subset by
        to : Collection
            collection of "to" values to subset by

        Returns
        -------
        WeightedNetwork
            the subsetted network
        '''
        return WeightedNetwork(mapping=[(f, t, w) for f, t, w in self._mapping if f in fr and t in to])
    
    def get_ligands(self) -> set[str]:
        '''
        Get all the ligands present in the network. 
        
        Returns
        -------
        set
            set of ligands present in the network
        '''
        return set(self.key_iter())
    
    def get_receptors(self) -> set[str]:
        '''
        Get all the receptors present in the network. 
        
        Returns
        -------
        set
            set of receptors present in the network
        '''
        return set(receptor for _, receptor, _ in self._mapping)
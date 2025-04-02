from nichenetpy.io import read_network, read_weighted_network

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

    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if neither mapping nor filename is provided
    
    Attributes
    ----------
    mapping : list
        sorted list of tuples (from, to)
    _index : dict
        index for the mapping
    
    Notes
    -----
    You must pass a list SORTED by "from" as mapping or the name of a file to read from. 
    '''
    def __init__(self, mapping:list=None, filename:str=None) -> None:
        if mapping is not None:
            if type(mapping) is not list:
                raise TypeError(f"mapping should have type list, was {type(mapping)}")
            self._mapping = mapping
        elif filename is not None:
            if type(filename) is not str:
                raise TypeError(f"filename should have type str, was {type(filename)}")
            self._mapping = sorted(
                read_network(filename),
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
    
    def __len__(self):
        return len(self._mapping)
    
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
    
    def subset(self, from_to:Collection[tuple[str, str]]):
        '''
        Subset the network by the provided links. 

        Parameters
        ----------
        from_to : Collection
            collection of tuples (from, to) to subset by

        Returns
        -------
        Network
            the subsetted network
        
        Raises
        ------
        TypeError
            if the arguments have the wrong type
        '''
        if not isinstance(from_to, Collection):
            raise TypeError(f"from_to should be a Collection of tuple[str, str], was {type(from_to)}")
        return type(self)(mapping=[tup for tup in self._mapping if (tup[0], tup[1]) in from_to])

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
        Network
            the subsetted network
        
        Raises
        ------
        TypeError
            if the arguments have the wrong type
        '''
        if not isinstance(fr, Collection):
            raise TypeError(f"fr should be a Collection of str, was {type(fr)}")
        if not isinstance(to, Collection):
            raise TypeError(f"to should be a Collection of str, was {type(to)}")
        return type(self)(mapping=[tup for tup in self._mapping if tup[0] in fr and tup[1] in to])

class LigandReceptorNetwork(Network):
    '''
    A ligand-receptor network without weights. This is in essence a one to many mapping. 

    Parameters
    ----------
    mapping : list
        sorted list of tuples (from, to)
    filename : str
        name of the file to read the network from
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    ValueError
        if neither mapping nor filename is provided
    
    Attributes
    ----------
    mapping : list
        sorted list of tuples (from, to)
    _index : dict
        index for the mapping
    
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
    
    Raises
    ------
    TypeError
        if the arguments have the wrong type
    
    Attributes
    ----------
    mapping : list
        sorted list of tuples (from, to, weight)
    _index : dict
        index for the mapping
    '''
    def __init__(self, mapping = None, filename = None):
        if filename is not None:
            if type(filename) is not str:
                raise TypeError(f"filename should have type str, was {type(filename)}")
            mapping = sorted(
                read_weighted_network(filename),
                key=lambda x : x[0]
            )
        super().__init__(mapping=mapping)
    
    def __getitem__(self, key:str) -> dict[str, float]:
        start, count = self._index[key]
        return dict(item[1:3] for item in self._mapping[start:start+count])
    
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
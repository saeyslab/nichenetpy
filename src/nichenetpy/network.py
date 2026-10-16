from collections.abc import Collection, Iterator
from itertools import chain
from typing import Generic, overload

import pandas as pd

from nichenetpy.io import read_network, read_weighted_network
from nichenetpy.typing import gene_t, GeneSymbol


class Network(Generic[GeneSymbol]):
    """
    A network with links. This is in essence a one to many mapping.

    Parameters
    ----------
    mapping : list or pandas.DataFrame or None
        sorted list of tuples (from, to) or pandas dataframe with keys ("from", "to")
    filename : str or None
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
    You must pass a list SORTED by "from" as mapping, a pandas dataframe with keys ("from", "to") or the name of a file to read from.
    """

    @overload
    def __init__(self, mapping: list[tuple[GeneSymbol, GeneSymbol]] | pd.DataFrame, filename: None=None): ...

    @overload
    def __init__(self, mapping: None, filename: str): ...
    
    def __init__(self, mapping: list[tuple[GeneSymbol, GeneSymbol]] | pd.DataFrame | None = None, filename: str | None = None):
        self._mapping : list[tuple[GeneSymbol, GeneSymbol]]
        if mapping is not None:
            if type(mapping) is pd.DataFrame:
                mapping = sorted(zip(mapping["from"].tolist(), mapping["to"].tolist(), strict=True), key=lambda x: x[0])
            elif type(mapping) is not list:
                raise TypeError(f"mapping should have type list or pandas.DataFrame, was {type(mapping)}")
            self._mapping = mapping
        elif filename is not None:
            if type(filename) is not str:
                raise TypeError(f"filename should have type str, was {type(filename)}")
            self._mapping = sorted(read_network(filename), key=lambda x: x[0]) # type: ignore (if filename is None then self._mapping has type list[tuple[str, str]])
        else:
            raise ValueError("either mapping or filename must be provided as arguments")
        self._index : dict[GeneSymbol, list[int]] = {}
        self._build_index()

    def __str__(self) -> str:
        return self._mapping.__str__()

    def __getitem__(self, key) -> set[GeneSymbol]:
        return set(self.mapping_iter(key))

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self) -> Iterator[tuple[GeneSymbol, GeneSymbol]]:
        return self._mapping.__iter__()

    def __contains__(self, item: tuple):
        start, count = self._index[item[0]]
        return item in self._mapping[start : start + count]

    def _build_index(self):
        self._index.clear()
        for i, item in enumerate(self._mapping):
            if item[0] in self._index:
                self._index[item[0]][1] += 1
            else:
                self._index[item[0]] = [i, 1]

    def key_iter(self) -> Iterator[GeneSymbol]:
        """
        Iterates over the "from" values in the mapping.

        Yields
        ------
        gene_t
            the "from" values in the mapping
        """
        return (self._mapping[start][0] for start, _ in self._index.values())

    def item_iter(self) -> Iterator[tuple[GeneSymbol, set[GeneSymbol]]]:
        """
        Iterates over the "from" values in the mapping along with the corresponding "to" values.

        Yields
        ------
        tuple
            the "from" values in the mapping and a list of corresponding "to" values
        """
        return ((key, self[key]) for key in self.key_iter())

    def mapping_iter(self, key) -> Iterator[GeneSymbol]:
        """
        Iterates over the "to" values that correspond with the specified "from" value.

        Yields
        ------
        tuple
            the "to" values in the mapping
        """
        try:
            start, count = self._index[key]
        except KeyError as error:
            raise KeyError(f"there is no edge with '{key}' as source") from error
        for item in self._mapping[start : start + count]:
            yield item[1]

    def get_all(self) -> set[GeneSymbol]:
        """
        Return all values that are present in the network.

        Yields
        ------
        set
            all values present in the network
        """
        return set(chain(*zip(*self._mapping, strict=True)))

    def subset(self, from_to: Collection[tuple[GeneSymbol, GeneSymbol]]):
        """
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
        """
        if not isinstance(from_to, Collection):
            raise TypeError(f"from_to should be a Collection of tuple[GeneSymbol, GeneSymbol], was {type(from_to)}")
        return type(self)(mapping=[tup for tup in self._mapping if (tup[0], tup[1]) in from_to])

    def subset_sep(self, fr: Collection[GeneSymbol] | None = None, to: Collection[GeneSymbol] | None = None):
        """
        Subset the network by the provided "from" and "to" values.

        Parameters
        ----------
        from : Collection or None
            collection of "from" values to subset by
        to : Collection or None
            collection of "to" values to subset by

        Returns
        -------
        Network
            the subsetted network

        Raises
        ------
        TypeError
            if the arguments have the wrong type
        """
        if fr is None:
            fr = set(self.key_iter())
        if to is None:
            to = {e for _, e in self._mapping}
        if not isinstance(fr, Collection):
            raise TypeError(f"fr should be a Collection of gene_t, was {type(fr)}")
        if not isinstance(to, Collection):
            raise TypeError(f"to should be a Collection of gene_t, was {type(to)}")
        return type(self)(mapping=[tup for tup in self._mapping if tup[0] in fr and tup[1] in to])

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the network to a dataframe

        Returns
        -------
        pandas.DataFrame
            the network as a dataframe
        """
        return pd.concat([pd.DataFrame({"from": fr, "to": list(tos)}) for fr, tos in self.item_iter()])


class LigandReceptorNetwork(Network):
    """
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
    """

    def get_ligands(self) -> set[gene_t]:
        """
        Get all the ligands present in the network.

        Returns
        -------
        set
            set of ligands present in the network
        """
        return set(self.key_iter())

    def get_receptors(self) -> set[gene_t]:
        """
        Get all the receptors present in the network.

        Returns
        -------
        set
            set of receptors present in the network
        """
        return {receptor for _, receptor in self._mapping}


class WeightedNetwork(Network, Generic[GeneSymbol]):
    """
    A ligand-receptor network with weighted links. This is in essence a one to many mapping.

    Parameters
    ----------
    mapping : list or pandas.DataFrame or None
        sorted list of tuples (from, to, weight) or pandas dataframe with keys ("from", "to", "weight")
    filename : str or None
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
    """

    @overload
    def __init__(self, mapping: list[tuple[GeneSymbol, GeneSymbol, float]] | pd.DataFrame | None = None, filename: None = None): ...

    @overload
    def __init__(self, mapping: None, filename: str): ...

    def __init__(self, mapping: list[tuple[GeneSymbol, GeneSymbol, float]] | pd.DataFrame | None = None, filename: str | None = None):
        self._mapping : list[tuple[GeneSymbol, GeneSymbol, float]]
        if mapping is not None:
            if type(mapping) is pd.DataFrame:
                mapping = sorted(zip(mapping["from"], mapping["to"], mapping["weight"], strict=True), key=lambda x: x[0])
            elif type(mapping) is not list:
                raise TypeError(f"mapping should have type list or pandas.DataFrame, was {type(mapping)}")
            self._mapping = mapping
        elif filename is not None:
            if type(filename) is not str:
                raise TypeError(f"filename should have type str, was {type(filename)}")
            self._mapping = sorted(read_weighted_network(filename), key=lambda x: x[0]) # type: ignore (if filename is None then self._mapping has type list[tuple[str, str, float]])
        else:
            raise ValueError("either mapping or filename must be provided as arguments")
        self._index : dict[gene_t, list[int]] = {}
        self._build_index()

    def __getitem__(self, key: gene_t) -> dict[gene_t, float]:
        start, count = self._index[key]
        return dict(item[1:3] for item in self._mapping[start : start + count])

    def item_iter(self) -> Iterator[tuple[gene_t, dict[gene_t, float]]]:
        """
        Iterates over the "from" values in the mapping along with the corresponding "to" values and link weights.

        Yields
        ------
        tuple
            the "from" values in the mapping and a dictionary which maps corresponding "to" values to the weight of the link
        """
        return ((key, self[key]) for key in self.key_iter())

    def get_ligands(self) -> set[gene_t]:
        """
        Get all the ligands present in the network.

        Returns
        -------
        set
            set of ligands present in the network
        """
        return set(self.key_iter())

    def get_receptors(self) -> set[gene_t]:
        """
        Get all the receptors present in the network.

        Returns
        -------
        set
            set of receptors present in the network
        """
        return {receptor for _, receptor, _ in self._mapping}

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the network to a dataframe

        Returns
        -------
        pandas.DataFrame
            the network as a dataframe
        """
        temp = []
        for fr, mapping in self.item_iter():
            tos, ws = zip(*mapping.items(), strict=True)
            temp.append(pd.DataFrame({"from": fr, "to": tos, "weight": ws}))
        return pd.concat(temp)

# copied from https://github.com/vmedaert/genetic_decision_rules/blob/main/genetic_decision_rules.py

from math import sqrt


class Pareto:
    def __init__(self, *values) -> None:
        if len(values) == 1:
            self._values = values[0]
        else:
            self._values = values
    
    def __str__(self):
        return str(self._values)
    
    def __repr__(self):
        return str(self)
    
    def __hash__(self):
        return self._values.__hash__()

    def __len__(self):
        return len(self._values)
    
    def __getitem__(self, i):
        return self._values[i]

    def __add__(self, other):
        if len(self) != len(other):
            raise ValueError("vectors should have same size")
        return Pareto(x + y for x, y in zip(self, other))
    
    def __iadd__(self, other):
        if len(self) != len(other):
            raise ValueError("vectors should have same size")
        for i in range(len(self)):
            self._values[i] += other._values[i]
        return self
    
    def __sub__(self, other):
        if len(self) != len(other):
            raise ValueError("vectors should have same size")
        return Pareto(x - y for x, y in zip(self, other))
    
    def __isub__(self, other):
        if len(self) != len(other):
            raise ValueError("vectors should have same size")
        for i in range(len(self)):
            self._values[i] -= other._values[i]
        return self

    def __mul__(self, other):
        return Pareto(x * other for x in self)
    
    def __imul__(self, other):
        for i in range(len(self)):
            self._values[i] *= other
        return self
    
    def __truediv__(self, other):
        return Pareto(x / other for x in self)
    
    def __itruediv__(self, other):
        for i in range(len(self)):
            self._values[i] /= other
        return self

    def __iter__(self):
        return self._values.__iter__()
    
    '''
    pareto dominance:
    a vector x dominates a vector y if there is no i such that x[i] < y[i]
    and there is at least one i such that x[i] > y[i]
    '''
    def __lt__(self, other):
        if len(self) != len(other):
            raise ValueError("vectors must be of same length")
        i = 0
        less = False
        while i < len(other) and self[i] <= other[i]:
            if self[i] < other[i]:
                less = True
            i += 1
        return i == len(self) and less
    
    def __gt__(self, other):
        return other < self
    
    def __eq__(self, other):
        return self._values == other._values
    
    def __neq__(self, other):
        return self._values != other._values
    
    def __le__(self, other):
        return self == other or self < other
    
    def __ge__(self, other):
        return self == other or self > other
    
    def norm(self):
        return sqrt(sum(x**2 for x in self._values))

class ParetoSet:
    def __init__(self):
        self._content = dict()
    
    def __str__(self):
        return str(self._content.items())
    
    def __iter__(self):
        return self._content.items().__iter__()
    
    def __len__(self):
        return len(self._content)
    
    def add(self, item:tuple) -> bool:
        dominated = False
        marked = set()
        # pareto dominance is transitive
        for e in self:
            if item[0] <= e[0]:
                # if dominated or equal, do not add to pareto set
                dominated = True
                break
            elif item[0] > e[0]:
                # when added to pareto set, remove dominated items
                marked.add(e[0])
        if not dominated:
            for key in marked:
                self._content.pop(key)
            self._content[item[0]] = item[1]
        return not dominated

    def clear(self):
        self._content.clear()
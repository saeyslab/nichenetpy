class AnnError(Exception):
    """raised when an AnnData object is not suitable for a NicheNet analysis"""

    pass


class NicheNetError(RuntimeError):
    """raised when something goes wrong during a NicheNet analysis"""

    pass


class NoTrueSamplesError(ValueError):
    """raised when an evaluation has no true samples (e.g. no active ligand), so the evaluation metrics are undefined"""

    pass


class NotInModelError(KeyError, ValueError):
    """raised when a gene or ligand is not in the ligand-target matrix (a KeyError, and a ValueError for code that catches ValueError)"""

    def __str__(self) -> str:
        # a KeyError shows its argument as a repr (in quotes), the message is shown as is
        return str(self.args[0]) if len(self.args) == 1 else super().__str__()

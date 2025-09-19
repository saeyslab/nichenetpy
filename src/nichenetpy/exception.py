class AnnError(Exception):
    '''
    raised when an AnnData object is not suitable for a NicheNet analysis
    '''
    pass

class NicheNetError(RuntimeError):
    '''
    raised when something goes wrong during a NicheNet analysis
    '''
    pass
try:
    from rpy2 import robjects
    r_installed = True
except ImportError:
    r_installed = False

if r_installed:
    import pandas as pd

    del r_installed

    _readRDS = robjects.r['readRDS']
    _rownames = robjects.r["rownames"]


    def convert_rpy2(
        obj,
        pandas:bool=True
    ):
        '''
        Converts an rpy2 object to a standard python object:
            - Named vectors are converted to dictionaries
            - Unnamed vectors with multiple elements are converted to lists
            - Vectors with a single element have their element returned
            - DataFrames are converted to pandas.DataFrame or a dictionary of columns

        Parameters
        ----------
        obj : Any
            the object to convert
        pandas : bool
            if True data frames are converted to pandas.DataFrame

        Returns
        -------
        Any
            a python object which corresponds with the R object
        '''
        if type(obj) is robjects.vectors.DataFrame:
            if pandas:
                return pd.DataFrame(dict(zip(obj.names, obj)), index=_rownames(obj))
            else:
                dct = dict(zip(obj.names, (convert_rpy2(e, pandas) for e in obj)))
                if "index" not in dct:
                    dct["index"] = convert_rpy2(_rownames(obj))
                return dct
        elif isinstance(obj, robjects.Vector):
            if obj.names:
                return dict(zip(obj.names, (convert_rpy2(e, pandas) for e in obj)))
            elif len(obj) > 1:
                return list(obj)
            else:
                return obj[0]
        else:
            return obj

    def read_RDS(
        file:str,
        pandas:bool=True
    ):
        '''
        Read the contents of an RDS file (native R format). 

        Parameters
        ----------
        file : str
            the name of the RDS file

        Returns
        -------
        Any
            a python object which corresponds with the R object
        
        Notes
        -----
        Uses rpy2 internally. 
        '''
        return convert_rpy2(_readRDS(file), pandas)

    def read_settings_from_RDS(file:str):
        '''
        Read settings (evaluation data from NicheNetR) from an RDS file

        Parameters
        ----------
        file : str
            the name of the RDS file

        Returns
        -------
        Any
            the settings
        '''
        settings = read_RDS(file, pandas=False)
        for setting in settings.values():
            setting["diffexp"]["gene"] = setting["diffexp"].pop("index")
        return settings
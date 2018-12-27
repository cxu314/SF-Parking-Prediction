import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from model import *

def assertPandasContains(columns, df):
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Pandas dataframe does not contain '{col}', it contains: " + str(list(df.columns)))

def assertPandasExclude(columns, df):
    for col in columns:
        if col in df.columns:
            raise KeyError(f"Pandas dataframe cannot contain '{col}', it contains: " + str(list(df.columns)))

def makeLabelEncoding(df, col):
    ids = df[col].unique()
    return pd.Series(data = np.arange(ids.shape[0]), index = ids)

def labelEncoder(columns, drop = True, suffix = ''):
    if not drop and suffix == '':
        raise Exception('Must have suffix if drop is True')

    def colInfo(df):
        assertPandasContains(columns, df)
        if not drop:
            assertPandasExclude([col + suffix for col in columns])

        return {col : makeLabelEncoding(df, col) for col in columns}

    def colMap(info, df):
        assertPandasContains(columns, df)
        if not drop:
            assertPandasExclude([col + suffix for col in columns])

        return df.drop(columns if drop else [], axis = 1)\
                .assign(**{col + suffix : df[col].map(m) for col, m in info.items()})

    return SimpleModelE(colInfo, colMap)

def hashEncoder(columns):
    pass

def joinFun(fun, on = None, how = 'inner', rsuffix = None):
    """fun : DataFrame -> DataFrame"""
    def joinMap(info, df):
        if rsuffix is not None:
            info = info.rename({col : col + rsuffix for col in info.columns}, axis = 1)

        return df.join(info, on = on, how = how)

    return SimpleModelE(fun, joinMap)

def columnMapper(fun, input_col, output_col):
    def mapCol(info, df):
        assertPandasContains([input_col], df)
        assertPandasExclude([output_col], df)
        return df.assign(**{output_col : fun(df[input_col])})

    return SimpleModelE(lambda df: None, mapCol)

def dropCols(cols):
    def dropper(df):
        assertPandasContains(cols, df)
        return df.drop(cols, axis = 1)

    return MapE(dropper)

def oneHotEncoder(columns, **params):
    def colInfo(df):
        assertPandasContains(columns, df)
        return {col : OneHotEncoder(**params).fit(df[col]) for col in columns}

    def colMap(info, df):
        assertPandasContains(columns, df)
        return df.drop(columns).assign(**{col : m.transform(df[col]) for col, m in info.items()})

    return SimpleModelE(colInfo, colMap)

class TrainOnly(Estimator):
    def __init__(self, fun):
        self.fun = fun

    def fit_transform(self, df):
        return IdentityTransformer(), self.fun(df)

def trainOnly(fun):
    return TrainOnly(fun)

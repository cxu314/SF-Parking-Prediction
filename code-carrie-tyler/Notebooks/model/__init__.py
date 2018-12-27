# Base Classes
class Estimator():
    def __mul__(f, g):
        stages = []
        if type(f) == Pipeline and type(g) == Pipeline:
            stages = f.stages + g.stages
        elif type(f) == Pipeline:
            stages = f.stages + [g]
        elif type(g) == Pipeline:
            stages = [f] + g.stages
        else:
            stages = [f, g]

        return Pipeline(stages)

    def fit_transform(self, df):
        raise NotImplementedError("Must implement this")

class Transformer():
    def __mul__(f, g):
        stages = []
        if type(f) == ThenTransfrom and type(g) == ThenTransfrom:
            stages = f.stages + g.stages
        elif type(f) == ThenTransfrom:
            stages = f.stages + [g]
        elif type(g) == ThenTransfrom:
            stages = [f] + g.stages
        else:
            stages = [f, g]

        return ThenTransfrom(stages)

    def transform(self, df):
        raise NotImplementedError("Must implement this")

class Predictor():
    def predict(self, df):
        raise NotImplementedError("Must implement this")

    def predict_proba(self, df):
        raise NotImplementedError("Must implement this")

    def predict_log_proba(self, df):
        raise NotImplementedError("Must implement this")

# Simple Model that does nothing with *
class IdentityTransformer(Transformer):
    def transform(self, df):
        return df

class IdentityE(Estimator):
    def fit_transform(df):
        return IdentityTransformer(), df

# Performs models one after another
class ThenTransfrom(Transformer, Predictor):
    def __init__(self, stages):
        self.stages = stages

    def transform(self, df):
        for stage in self.stages:
            df = stage.transform(df)

        return df

    def predict(self, df):
        for stage in self.stages[:-1]:
            df = stage.transform(df)

        return self.stages[-1].predict(df)

class Pipeline(Estimator):
    def __init__(self, stages):
        self.stages = stages

    def fit_transform(self, df):
        ts = []
        for stage in self.stages:
            t, df = stage.fit_transform(df)
            ts.append(t)

        return ThenTransfrom(ts), df

def pipeline(stages):
    return Pipeline(stages)

class ConstE(Estimator):
    def __init__(self, t):
        self.t = t

    def fit_transform(self, df):
        return self.t, self.t.transform(df)

class FuncPredcit(Predictor):
    def __init__(self, func):
        self.func = func

    def predict(self, df):
        return self.func(df)

class MapT(Transformer):
    def __init__(self, func):
        self.func = func

    def transform(self, df):
        return self.func(df)

class MapE(Estimator):
    def __init__(self, func):
        self.func = func

    def fit_transform(self, df):
        t = MapT(self.func)
        return t, t.transform(df)

class SimpleModelE(Estimator):
    def __init__(self, fit_func, func):
        self.fit_func = fit_func
        self.func = func

    def fit_transform(self, df):
        info = self.fit_func(df)
        t = MapT(lambda df2: self.func(info, df2))
        return t, t.transform(df)

class SimplePredE(Estimator):
    def __init__(self, fit_func, func):
        self.fit_func = fit_func
        self.func = func

    def fit_transform(self, df):
        info = self.fit_func(df)
        t = FuncPredcit(lambda df2: func(info, df2))
        return t, t.transform(df)

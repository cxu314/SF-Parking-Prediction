class TypeError(Exception):
    def __init__(self, message):
        self.message = message

class Type:
    def assertInstanceOf(self, x):
        raise NotImplemented()

class PandasContainsExcept(Type):
    def __init__(self, contains, excludes = []):
        self.contains = contains
        self.excludes = excludes

    def assertInstanceOf(self, x):
        if type(x) == pd.DataFrame:
            for col in self.contains:
                if col not in x.columns:
                    raise TypeError(f'Pandas Dataframe did not contain "{col}" when it is required to.')

            for col in self.excludes:
                if col in x.columns:
                    raise TypeError(f'Pandas Dataframe contains "{col}" when it is required not to.')

            return True

        raise TypeError(f'Expected a pandas dataframe, recived a {type(x)}')

class SimpleType(Type):
    def __init__(self, t):
        self.t = t

    def assertInstanceOf(self, x):
        if type(x) == self.t:
            return True

        raise TypeError(f'Expected a {self.t}, recived a {type(x)}')

class AnyType(Type):
    def assertInstanceOf(self, x):
        return True

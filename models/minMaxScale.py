class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        self.min = X.min(axis = 0)
        self.max = X.max(axis = 0)
        return self

    def transform(self, X):  
        X = (X - self.min) / (self.max - self.min)    
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
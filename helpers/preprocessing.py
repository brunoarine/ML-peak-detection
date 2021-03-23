from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn import ensemble
import numpy as np
from scipy import signal

class Cwt(BaseEstimator,TransformerMixin):
    def __init__(self, maxwidth=5):
        self.maxwidth = maxwidth
    def transform(self, X, *_):
        #widths = np.linspace(1,8,15)/2.355
        widths = np.arange(1,self.maxwidth)
        new_X = []
        for i in range(len(X)):            
            new_X.append(signal.cwt(X[i], signal.ricker, widths).ravel())
        return np.array(new_X)
    def fit(self, X, *_):
        return self


class ForestSelect(BaseEstimator,TransformerMixin):
    def __init__(self, k=21, trees=500, max_depth=4):        
        self.k = k
        self.trees = trees
        self.max_depth = max_depth
    def transform(self, X, *_):                          
        return X[:,self.positions_]
    
    def fit(self, X, y, *_):
        select = ensemble.RandomForestClassifier(self.trees, max_depth=self.max_depth)
        select.fit(X,y)
        self.importances_ = select.feature_importances_
        self.positions_ = np.argsort(self.importances_)[::-1][:self.k]
        return self
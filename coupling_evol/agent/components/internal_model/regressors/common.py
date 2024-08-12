import numpy as np
import os

class ModelInterface:
    """
    Base interface for all models. Should be loadable/saveable aswell.
    """
    def fit(self, X, y, *args):
        pass

    def predict(self, X, *args):
        pass

    def derivative(self, X, *args):
        """
        @param X: the derivative can depend on the input
        @return: @return: W (y_dim, u_dim * granularity)
        """
        pass

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass

class JustMean(ModelInterface):
    def __init__(self, prior_mean):
        self.mean = prior_mean

    def fit(self, X, y, *args):
        if len(y) > 0:
            self.mean = np.mean(y, axis=0)

    def predict(self, X, *args):
        return np.asarray([self.mean]*len(X))

    def derivative(self, X, *args):
        np.zeros((self.mean.shape[0], X.shape[1], self.mean.shape[1]))
        return np.asarray([np.zeros(self.mean.shape)]*len(X))

    def save(self, path):
        np.savetxt(os.path.join(path, "mean"), self.mean)

    @classmethod
    def load(cls, path):
        mean = np.loadtxt(os.path.join(path, "mean"))
        return cls(mean)
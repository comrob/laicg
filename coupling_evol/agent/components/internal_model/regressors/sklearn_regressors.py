from coupling_evol.agent.components.internal_model.regressors.common import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
import pickle

class StdNormedLinearRegressor(ModelInterface):
    NUMERICAL_EPS = 0.000001

    def __init__(self):

        self.model = LinearRegression()
        self.y_mean = np.asarray([])
        self.X_mean = np.asarray([])
        self.y_std = np.asarray([])
        self.X_std = np.asarray([])

    def fit(self, X, y, *args):
        self.X_mean = np.mean(X, axis=0)
        self.y_mean = np.mean(y, axis=0)
        self.X_std = np.maximum(np.std(X, axis=0), self.NUMERICAL_EPS)
        self.y_std = np.maximum(np.std(y, axis=0), self.NUMERICAL_EPS)
        _X = (X - self.X_mean)/self.X_std
        _y = (y - self.y_mean)/self.y_std
        self.model.fit(_X, _y)

    def predict(self, X, *args):
        _X = (X - self.X_mean) / self.X_std
        std_ret = self.model.predict(_X)
        return (std_ret * self.y_std) + self.y_mean

    def derivative(self, X, *args):
        """
        d(b+Wx)/dx = W
        @param X:
        @param args:
        @return: W (y_dim, u_dim * granularity)
        """
        return self.model.coef_

    def save(self, path):
        with open(os.path.join(path, "params.sav"), "wb") as file:
            pickle.dump(self.model, file)
        np.savetxt(os.path.join(path, "X_mean"), self.X_mean)
        np.savetxt(os.path.join(path, "y_mean"), self.y_mean)
        np.savetxt(os.path.join(path, "y_std"), self.y_std)
        np.savetxt(os.path.join(path, "X_std"), self.X_std)

    @classmethod
    def load(cls, path):
        model = cls()
        with open(os.path.join(path, "params.sav"), "rb") as file:
            model.model = pickle.load(file)
        model.X_mean = np.loadtxt(os.path.join(path, "X_mean"))
        model.y_mean = np.loadtxt(os.path.join(path, "y_mean"))
        model.y_std = np.loadtxt(os.path.join(path, "y_std"))
        model.X_std = np.loadtxt(os.path.join(path, "X_std"))
        return model


class StdNormedRidgeRegressor(ModelInterface):
    NUMERICAL_EPS = 0.000001

    def __init__(self):
        self.model = RidgeCV()
        self.y_mean = np.asarray([])
        self.X_mean = np.asarray([])
        self.y_std = np.asarray([])
        self.X_std = np.asarray([])

    def fit(self, X, y, *args):
        self.X_mean = np.mean(X, axis=0)
        self.y_mean = np.mean(y, axis=0)
        self.X_std = np.maximum(np.std(X, axis=0), self.NUMERICAL_EPS)
        self.y_std = np.maximum(np.std(y, axis=0), self.NUMERICAL_EPS)
        _X = (X - self.X_mean)/self.X_std
        _y = (y - self.y_mean)/self.y_std
        self.model.fit(_X, _y)

    def predict(self, X, *args):
        _X = (X - self.X_mean) / self.X_std
        std_ret = self.model.predict(_X)
        return (std_ret * self.y_std) + self.y_mean

    def save(self, path):
        with open(os.path.join(path, "params.sav"), "wb") as file:
            pickle.dump(self.model, file)
        np.savetxt(os.path.join(path, "X_mean"), self.X_mean)
        np.savetxt(os.path.join(path, "y_mean"), self.y_mean)
        np.savetxt(os.path.join(path, "y_std"), self.y_std)
        np.savetxt(os.path.join(path, "X_std"), self.X_std)

    @classmethod
    def load(cls, path):
        model = cls()
        with open(os.path.join(path, "params.sav"), "rb") as file:
            model.model = pickle.load(file)
        model.X_mean = np.loadtxt(os.path.join(path, "X_mean"))
        model.y_mean = np.loadtxt(os.path.join(path, "y_mean"))
        model.y_std = np.loadtxt(os.path.join(path, "y_std"))
        model.X_std = np.loadtxt(os.path.join(path, "X_std"))
        return model


class GaussianProcessRbf(ModelInterface):
    def __init__(self):
        self.model = GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(),
        )
        # self.model = GaussianProcessRegressor(kernel=DotProduct(sigma_0_bounds=(1e-3, 1e3)) + WhiteKernel()*10)

    def fit(self, X, y, *args):
        self.model.fit(X, y)

    def predict(self, X, *args):
        return self.model.predict(X)

    def save(self, path):
        with open(os.path.join(path, "params.sav"), "wb") as file:
            pickle.dump(self.model, file)

    @classmethod
    def load(cls, path):
        model = cls()
        with open(os.path.join(path, "params.sav"), "rb") as file:
            model.model = pickle.load(file)
        return model


class GaussianProcessDot(ModelInterface):
    def __init__(self):
        self.model = GaussianProcessRegressor(kernel=DotProduct(sigma_0_bounds=(1e-3, 1e3)) + WhiteKernel()*10)

    def fit(self, X, y, *args):
        self.model.fit(X, y)

    def predict(self, X, *args):
        return self.model.predict(X)

    def save(self, path):
        with open(os.path.join(path, "params.sav"), "wb") as file:
            pickle.dump(self.model, file)

    @classmethod
    def load(cls, path):
        model = cls()
        with open(os.path.join(path, "params.sav"), "rb") as file:
            model.model = pickle.load(file)
        return model


class StdNormedPerceptronRegressor(ModelInterface):
    NUMERICAL_EPS = 0.000001
    def __init__(self):
        self.model = MLPRegressor(random_state=1, max_iter=1000, activation='tanh')
        # self.model = MLPRegressor(random_state=1, max_iter=1000, activation='logistic')
        self.y_mean = np.asarray([])
        self.X_mean = np.asarray([])
        self.y_std = np.asarray([])
        self.X_std = np.asarray([])

    def fit(self, X, y, *args):
        self.X_mean = np.mean(X, axis=0)
        self.y_mean = np.mean(y, axis=0)
        self.X_std = np.maximum(np.std(X, axis=0), self.NUMERICAL_EPS)
        self.y_std = np.maximum(np.std(y, axis=0), self.NUMERICAL_EPS)
        _X = (X - self.X_mean)/self.X_std
        _y = (y - self.y_mean)/self.y_std
        self.model.fit(_X, _y)

    def predict(self, X, *args):
        _X = (X - self.X_mean) / self.X_std
        std_ret = self.model.predict(_X)
        return (std_ret * self.y_std) + self.y_mean

    def save(self, path):
        with open(os.path.join(path, "params.sav"), "wb") as file:
            pickle.dump(self.model, file)
        np.savetxt(os.path.join(path, "X_mean"), self.X_mean)
        np.savetxt(os.path.join(path, "y_mean"), self.y_mean)
        np.savetxt(os.path.join(path, "y_std"), self.y_std)
        np.savetxt(os.path.join(path, "X_std"), self.X_std)

    @classmethod
    def load(cls, path):
        model = cls()
        with open(os.path.join(path, "params.sav"), "rb") as file:
            model.model = pickle.load(file)
        model.X_mean = np.loadtxt(os.path.join(path, "X_mean"))
        model.y_mean = np.loadtxt(os.path.join(path, "y_mean"))
        model.y_std = np.loadtxt(os.path.join(path, "y_std"))
        model.X_std = np.loadtxt(os.path.join(path, "X_std"))
        return model



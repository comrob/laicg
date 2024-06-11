import numpy as np


class Session(object):
    def __init__(self):
        pass

    def step(self, u: np.ndarray) -> np.ndarray:
        pass


class Environment(object):
    def position_zero(self):
        pass

    def end_session(self):
        pass

    def create_session(self) -> Session:
        pass


class DummySession(Session):
    def __init__(self, transform, translate):
        super().__init__()
        self.transform = transform
        self.translate = translate

    def step(self, u: np.ndarray) -> np.ndarray:
        ret = u.dot(self.transform) + self.translate
        return ret


class DummyEnvironment(Environment):
    def __init__(self, u_dim: int, y_dim: int):
        self.u_dim = u_dim
        self.y_dim = y_dim
        self.transform = np.linspace(-1, 1, u_dim * y_dim).reshape((u_dim, y_dim))
        self.translate = np.arange(y_dim)

    def position_zero(self):
        print("Dummy to position zero.")

    def end_session(self):
        print("Dummy ending session")

    def create_session(self) -> Session:
        return DummySession(self.transform, self.translate)


if __name__ == '__main__':
    env = DummyEnvironment(3, 4)
    sess = env.create_session()
    print(sess.step(np.random.randn(3)))
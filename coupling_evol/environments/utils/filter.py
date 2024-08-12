import numpy as np
_ROUND_ANGLE = np.pi * 2


class DifferentialFilter:
    def __init__(self, dimension: int, delta_t=0.01, clip=0.1):
        self._last_value = np.zeros((dimension, ))
        self.delta_t=delta_t
        self.first = True
        self.clip = clip

    def __call__(self, value: np.ndarray):
        if self.first:
            self._last_value = value
            self.first = False
        ret = np.clip(value - self._last_value, a_max=self.clip, a_min=-self.clip)/self.delta_t
        self._last_value = value
        return ret


class AngleDifferentialFilter:
    def __init__(self, dimension: int, delta_t=0.01, clip=0.1, cheap_diff=True):
        self._last_value = np.zeros((dimension, ))
        self.delta_t = delta_t
        self.first = True
        self.clip = clip
        if cheap_diff:
            self._diff = self._cheap_diff
        else:
            self._diff = self._expensive_diff

    @staticmethod
    def _cheap_diff(val1, val2):
        return np.mod(val2 - val1 + np.pi, _ROUND_ANGLE) - np.pi

    @staticmethod
    def _expensive_diff(val1, val2):
        return np.arctan2(np.sin(val2-val1), np.cos(val2-val1))

    def __call__(self, value: np.ndarray):
        if self.first:
            self._last_value = value
            self.first = False
        ret = self._diff(self._last_value, value)/self.delta_t
        self._last_value = value
        return np.clip(ret, a_min=-self.clip, a_max=self.clip)


class MeanFilter:
    def __init__(self, sample_length: int, dimension: int):
        self.buffer = np.zeros((sample_length, dimension))
        self.idx = 0
        self.length = sample_length

    def __call__(self, value: np.ndarray):
        self.buffer[self.idx, :] = value
        self.idx = (self.idx + 1) % self.length
        return np.mean(self.buffer, axis=0)
from coupling_evol.engine.experiment_executor import ScenarioController
import numpy as np
from typing import Union


class SensoryLinearTransform(ScenarioController):
    def __init__(self, shift: Union[np.ndarray, float], scale: Union[np.ndarray, float]):
        super().__init__()
        self.shift = shift
        self.scale = scale

    def observation_transform(self, observation: np.ndarray) -> np.ndarray:
        return observation * self.scale + self.shift


class MotorLinearTransform(ScenarioController):
    def __init__(self, shift: Union[np.ndarray, float], scale: Union[np.ndarray, float]):
        super().__init__()
        self.shift = shift
        self.scale = scale

    def command_transform(self, command: np.ndarray) -> np.ndarray:
        return command * self.scale + self.shift

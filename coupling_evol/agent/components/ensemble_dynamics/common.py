from typing import List, Callable
from coupling_evol.agent.components.internal_model import forward_model as FM
from coupling_evol.engine import common as C
import numpy as np

from coupling_evol.engine.embedded_control import EmbeddedTargetParameter


class EnsembleDynamics(object):
    ZERO_MODEL_ID = -1

    def __init__(self,
                 sensory_dim, motor_dim, phase_dim,
                 models: List[FM.MultiPhaseModel]):
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        self.phase_dim = phase_dim
        self._models = models

    def __call__(self, sensory_embedding: np.ndarray, motor_embedding: np.ndarray,
                 target_parameter: EmbeddedTargetParameter, motion_phase: np.ndarray) -> int:
        pass


ENSEMBLE_DYNAMICS_FACTORY = Callable[[int, int, int, List[FM.MultiPhaseModel]], EnsembleDynamics]

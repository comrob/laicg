from coupling_evol.engine.dynamic_lifecycle import DynamicLifecycle, WorldModel
import numpy as np
from coupling_evol.engine.embedded_control import EmbeddedTargetParameter

import coupling_evol.data_process.inprocess.recorder as rlog
from coupling_evol.engine import common as C

REC = rlog.get_recorder(C.RecordNamespace.LIFE_CYCLE.key)


class Dummy(DynamicLifecycle):
    def __init__(self, world_model: WorldModel, sensor_dim: int, motor_dim: int, granularity: int):
        super().__init__(world_model, sensor_dim, motor_dim, granularity)

    def __call__(self, target: EmbeddedTargetParameter, observation: np.ndarray) -> C.COMMAND_T:
        if len(observation) == 0:
            observation = np.zeros((self.sensor_dim,))
        u = np.zeros((self.motor_dim,)) + 1
        REC("u", u)
        REC("y", observation)
        return u, C.CommandType.DIRECT

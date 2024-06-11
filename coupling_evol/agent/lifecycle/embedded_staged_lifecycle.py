from coupling_evol.engine.dynamic_lifecycle import DynamicLifecycle, WorldModel
from coupling_evol.agent.components.controllers import motor_babbling as MB
from coupling_evol.agent.lifecycle.control_manager.common import EmbeddingControlManager
from coupling_evol.agent.components.embedding import cpg_rbf_embedding as CRE
import numpy as np
import logging

from coupling_evol.engine.embedded_control import EmbeddingController

LOG = logging.getLogger(__name__)
from enum import Enum


class EmbeddedStagedLC(DynamicLifecycle):

    R_STAGE = "stg"

    def __init__(self,
                 world_model: WorldModel, sensor_dim: int, motor_dim: int, granularity: int,
                 motor_babbler: MB.MotorBabbler, embedding_controller_manager: EmbeddingControlManager,
                 rbf_epsilon=1., integration_step_size=0.01, natural_cpg_frequency=1.,
                 babbling_rate=2.
                 ):
        super().__init__(world_model, sensor_dim, motor_dim, granularity)
        if embedding_controller_manager.has_variables():
            self._base_gait = embedding_controller_manager.get_base_gait()
        else:
            self._base_gait = np.zeros((motor_dim, granularity))
        self._rbf_epsilon = rbf_epsilon
        self._integration_step_size = integration_step_size
        self._natural_cpg_frequency = natural_cpg_frequency
        ## Can be build before process
        self.cpg_rbf = CRE.CpgRbfDiscrete(
            natural_frequency=natural_cpg_frequency,
            rbf_epsilon=rbf_epsilon,
            granularity=granularity,
            step_size=integration_step_size
        )

        self.observation_embedder = CRE.Embedder(
            dimension=sensor_dim,
            granularity=granularity,
            combiner=CRE.mean_combiner()
        )
        self.babbling_rate = babbling_rate
        self.motor_babbler = motor_babbler
        self.natural_frequency = natural_cpg_frequency
        self.integration_step_size = integration_step_size
        self.embedding_controller_manager = embedding_controller_manager


class BabblePerformanceAlternation(EmbeddedStagedLC):

    OBSERVATION_EMBEDDING_PSFX = CRE.UEMB_B
    CPG_PSFX = CRE.CPG_PSFX
    BABBLE_PSFX = MB.BABBLE_PSFX
    CONTROL_PSFX = EmbeddingController.CONTROL_PSFX

    class StageStates(Enum):
        BABBLING_PREPARATION = 0,
        BABBLING_STAGE = 1,
        AFTER_BABBLING_LEARNING = 2,
        PERFORMANCE_STAGE = 3,
        BABBLING_INIT = 4,
        PERFORMANCE_INIT = 5,

        def get_intervals(self, stage_signal: np.ndarray, min_length=1000):
            intervals = []
            in_interval = False
            start_id = 0
            for i in range(len(stage_signal)):
                if stage_signal[i] == self.value[0] and not in_interval:
                    start_id = i
                    in_interval = True
                elif stage_signal[i] != self.value[0] and in_interval:
                    _end = i
                    if i - start_id < min_length:
                        _end = min((start_id+min_length), len(stage_signal))
                    intervals.append((start_id, _end))
                    in_interval = False
            if in_interval:
                intervals.append((start_id, len(stage_signal)))
            return intervals

    def __init__(self, world_model: WorldModel, sensor_dim: int, motor_dim: int, granularity: int,
                 motor_babbler: MB.MotorBabbler, embedding_controller_manager: EmbeddingControlManager,
                 rbf_epsilon=1., integration_step_size=0.01, natural_cpg_frequency=1.,
                 babbling_rate=2.
                 ):
        super().__init__(world_model, sensor_dim, motor_dim, granularity, motor_babbler, embedding_controller_manager,
                         rbf_epsilon, integration_step_size, natural_cpg_frequency, babbling_rate)



from coupling_evol.engine.dynamic_lifecycle import WorldModel
import numpy as np

from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import EmbeddingControlManager, \
    BabblePerformanceAlternation
from coupling_evol.engine.embedded_control import ConstantEmbeddingController, EmbeddingController, EmbeddedTargetParameter
import coupling_evol.engine.common as C

from coupling_evol.agent.components.controllers import motor_babbling as MB
from coupling_evol.agent.components.embedding import cpg_rbf_embedding as CRE
import logging
from typing import Tuple
LOG = logging.getLogger(__name__)

import coupling_evol.data_process.inprocess.recorder as rlog
from coupling_evol.engine import common as C
REC = rlog.get_recorder(C.RecordNamespace.LIFE_CYCLE.key)


class RepeatedScheduleLC(BabblePerformanceAlternation):
    _PH_B = "ph"
    _U_B = "u"
    _UEMB_B = "u_emb"
    _Y_B = "y"
    _BASE_GAIT_MEAN_N = 20
    _TRAIN_START_CROP = 200

    def __init__(self, world_model: WorldModel, sensor_dim: int, motor_dim: int, granularity: int,
                 motor_babbler: MB.MotorBabbler, embedding_controller_manager: EmbeddingControlManager,
                 babble_stage_iterations: int, performance_stage_iterations: int, start_with_babble=True,
                 rbf_epsilon=1.,
                 integration_step_size=0.01, natural_cpg_frequency=1., babbling_rate=2.
                 ):

        super().__init__(world_model, sensor_dim, motor_dim, granularity, motor_babbler, embedding_controller_manager,
                         rbf_epsilon, integration_step_size, natural_cpg_frequency, babbling_rate)
        self._stage_counter = 0
        self._babbling_controller = ConstantEmbeddingController(self._base_gait)
        self._performing_controller = ConstantEmbeddingController(self._base_gait)
        self.babble_stage_iterations = babble_stage_iterations
        self.performance_stage_iterations = performance_stage_iterations
        self._control_routine_buffer = {}

        if start_with_babble:
            self.stage = self.StageStates.BABBLING_INIT
        else:
            assert len(world_model) > 0, "Starting with performance stage but there is no model."
            self.stage = self.StageStates.PERFORMANCE_INIT


    def _update_control_routine_buffer(self, ph_act, observation, u, u_embedded):
        self._control_routine_buffer[self._PH_B].append(ph_act)
        self._control_routine_buffer[self._U_B].append(u)
        self._control_routine_buffer[self._Y_B].append(observation)
        self._control_routine_buffer[self._UEMB_B].append(u_embedded)
        pass

    def _clear_control_routine_buffer(self):
        self._control_routine_buffer = {
            self._PH_B: [],
            self._U_B: [],
            self._UEMB_B: [],
            self._Y_B: [],
        }

    def _prepare_control_routine(self):
        self._clear_control_routine_buffer()

    def _get_collected_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts the learning data. There is one catch: the temporal relation is Y(t-1), U(t), however
        we want to predict future sensor from command (not guess previous sensor from command).
        @return:
        @rtype:
        """
        return np.asarray(self._control_routine_buffer[self._PH_B])[self._TRAIN_START_CROP:-1],\
               np.asarray(self._control_routine_buffer[self._U_B])[self._TRAIN_START_CROP:-1],\
               np.asarray(self._control_routine_buffer[self._Y_B][self._TRAIN_START_CROP+1:])

    def _update_base_gait(self):
        self._base_gait = np.mean(np.asarray(self._control_routine_buffer[self._UEMB_B])[-self._BASE_GAIT_MEAN_N:, :, :], axis=0)

    def _get_base_gait(self) -> np.ndarray:
        return self._base_gait

    def _control_routine(self, embedding_controller: EmbeddingController,
                         target: EmbeddedTargetParameter, observation: np.ndarray, babbling_rate: float):
        """
        The direct control routine. Returns the control command and also records the step into buffer.
        @param embedding_controller:
        @param target:
        @param observation:
        @param babbling_rate:
        @return:
        """

        # CPG EMBEDDING
        self.cpg_rbf.step(
            perturbation=0.,
            natural_frequency=self.cpg_rbf.natural_frequency
        )
        ph_act = self.cpg_rbf.current_activation()
        ph_act_soft = self.cpg_rbf.current_activation_soft()
        # Observation embedding
        self.observation_embedder.step(ph_act, observation)
        y_obs_emb = self.observation_embedder.current_embedding()
        # Babbling update
        self.motor_babbler.step(y_obs_emb, target, phase_activation=ph_act)
        u_babble = self.motor_babbler.current_babble()

        # MOTOR COMMAND
        # inverse model
        u_embedded = embedding_controller(y_obs_emb, target, ph_act_soft)
        # joint embedded command
        u_joint = u_embedded + u_babble * babbling_rate
        # de-embedding
        u = CRE.soft_de_embedding(phase_activation=ph_act_soft, embedding=u_joint)
        ##
        self._update_control_routine_buffer(
            ph_act=ph_act,
            observation=observation,
            u=u,
            u_embedded=u_joint
        )
        return u

    def _babbling_preparation(self) -> C.COMMAND_T:
        """
        After performing, the new base_gait must be uploaded and the storage for new training data prepared.
        @return:
        @rtype:
        """
        self.embedding_controller_manager.save_controller_variables()
        self._update_base_gait()
        self._babbling_controller = ConstantEmbeddingController(self._get_base_gait())
        self._prepare_control_routine()
        return np.zeros((self.motor_dim, )), C.CommandType.POSITION_ZERO

    def _babbling_init(self) -> C.COMMAND_T:
        """
        Init just prepares the environment to position zero.
        @return:
        @rtype:
        """
        self._babbling_controller = ConstantEmbeddingController(self._get_base_gait())
        self._prepare_control_routine()
        return np.zeros((self.motor_dim, )), C.CommandType.POSITION_ZERO

    def _babbling_stage(self, target: EmbeddedTargetParameter, observation: np.ndarray) -> C.COMMAND_T:
        """
        During the babbling we collect training data.
        @param target:
        @type target:
        @param observation:
        @type observation:
        @return:
        @rtype:
        """
        u= self._control_routine(
            self._babbling_controller, target=target, observation=observation, babbling_rate=self.babbling_rate)
        return u, C.CommandType.DIRECT

    def _after_babbling_learning(self) -> C.COMMAND_T:
        """
        After the babbling the collected data are used for creating a new model for the world model.
        The performing controller is updated with the new world model.
        @return:
        @rtype:
        """
        phase_signal, u_signal, y_signal = self._get_collected_training_data()
        base_gait = self._get_base_gait()
        self.wm.learn_and_append(phase_signal, u_signal, y_signal)
        self.embedding_controller_manager.rebuild_controller(base_gait=base_gait, models=self.wm.models)
        self._prepare_control_routine()
        return np.zeros((self.motor_dim, )), C.CommandType.POSITION_ZERO

    def _performing_init(self) -> C.COMMAND_T:
        """
        The performing controller is built with the world model.
        @return:
        @rtype:
        """
        base_gait = self._get_base_gait()
        self.embedding_controller_manager.rebuild_controller(base_gait=base_gait, models=self.wm.models)
        self._prepare_control_routine()
        return np.zeros((self.motor_dim, )), C.CommandType.POSITION_ZERO

    def _performing_stage(self, target: EmbeddedTargetParameter, observation: np.ndarray) -> C.COMMAND_T:
        """
        During the performance we just perform.
        @param target:
        @type target:
        @param observation:
        @type observation:
        @return:
        @rtype:
        """
        u = self._control_routine(
            self.embedding_controller_manager.get_controller(), target=target, observation=observation, babbling_rate=0.)
        return u, C.CommandType.DIRECT

    def _resolve_stage(self):
        if self.stage == self.StageStates.BABBLING_PREPARATION and self._stage_counter > 0:
            self.stage = self.StageStates.BABBLING_STAGE
            LOG.info(f"Switching to {self.stage.name}")
        elif self.stage == self.StageStates.BABBLING_STAGE and self._stage_counter > self.babble_stage_iterations:
            self.stage = self.StageStates.AFTER_BABBLING_LEARNING
            self._stage_counter = 0
            LOG.info(f"Switching to {self.stage.name}")
        elif self.stage == self.StageStates.AFTER_BABBLING_LEARNING and self._stage_counter > 0:
            self.stage = self.StageStates.PERFORMANCE_STAGE
            LOG.info(f"Switching to {self.stage.name}")
        elif self.stage == self.StageStates.PERFORMANCE_STAGE and self._stage_counter > self.performance_stage_iterations:
            self.stage = self.StageStates.BABBLING_PREPARATION
            self._stage_counter = 0
            LOG.info(f"Switching to {self.stage.name}")
        elif self.stage == self.StageStates.BABBLING_INIT and self._stage_counter > 0:
            self.stage = self.StageStates.BABBLING_STAGE
            LOG.info(f"Switching to {self.stage.name}")
        elif self.stage == self.StageStates.PERFORMANCE_INIT and self._stage_counter > 0:
            self.stage = self.StageStates.PERFORMANCE_STAGE
            LOG.info(f"Switching to {self.stage.name}")
        ##
        self._stage_counter += 1

    def _execute_stage(self, target: EmbeddedTargetParameter, observation: np.ndarray) -> C.COMMAND_T:
        self._resolve_stage()
        ##
        if self.stage == self.StageStates.BABBLING_PREPARATION:
            return self._babbling_preparation()
        if self.stage == self.StageStates.BABBLING_STAGE:
            return self._babbling_stage(target, observation)
        if self.stage == self.StageStates.AFTER_BABBLING_LEARNING:
            return self._after_babbling_learning()
        if self.stage == self.StageStates.PERFORMANCE_STAGE:
            return self._performing_stage(target, observation)
        if self.stage == self.StageStates.BABBLING_INIT:
            return self._babbling_init()
        if self.stage == self.StageStates.PERFORMANCE_INIT:
            return self._performing_init()

    def __call__(self, target: EmbeddedTargetParameter, observation: np.ndarray) -> C.COMMAND_T:
        if C.is_empty_observation(observation):
            observation = np.zeros((self.sensor_dim, ))
        ###
        cmd, cmd_t = self._execute_stage(target, observation)
        ###

        REC(self.R_CMD, cmd)
        REC(self.R_CMD_T, cmd_t.value[0])
        REC(self.R_OBSERVATION, observation)
        REC(self.R_STAGE, self.stage.value[0])
        target.write_into_logger(REC, self.R_TARGET)
        return cmd, cmd_t

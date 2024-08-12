import numpy as np
from coupling_evol.engine.dynamic_lifecycle import WorldModel
from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import EmbeddingControlManager, \
    BabblePerformanceAlternation
import coupling_evol.engine.common as C
from coupling_evol.agent.components.ensemble_dynamics.common import ENSEMBLE_DYNAMICS_FACTORY, EnsembleDynamics
from coupling_evol.engine.embedded_control import ConstantEmbeddingController, EmbeddingController, \
    EmbeddedTargetParameter

from coupling_evol.agent.components.controllers import motor_babbling as MB
from coupling_evol.agent.components.embedding import cpg_rbf_embedding as CRE
import logging
from typing import Tuple

LOG = logging.getLogger(__name__)

import coupling_evol.data_process.inprocess.recorder as rlog
from coupling_evol.engine import common as C

REC = rlog.get_recorder(C.RecordNamespace.LIFE_CYCLE.key)

from abc import ABC


class CompetitionHandlingStrategy(ABC):
    def __init__(self):
        self.prev_model_sel = -1
        self.is_model_sel_switch = False

    def evaluate(self, ed: EnsembleDynamics, sensory_embedding: np.ndarray, motor_embedding: np.ndarray,
                 target_parameter: EmbeddedTargetParameter, motion_phase: np.ndarray):
        model_sel = ed(sensory_embedding=sensory_embedding, motor_embedding=motor_embedding,
                       target_parameter=target_parameter, motion_phase=motion_phase)
        self.is_model_sel_switch = False
        if self.prev_model_sel != model_sel:
            self.is_model_sel_switch = True
        self.prev_model_sel = model_sel

    def implant_model(self, emc: EmbeddingControlManager, wm: WorldModel, ed: EnsembleDynamics):
        pass

    def is_switched_to_learning(self):
        return self.is_model_sel_switch and self.prev_model_sel == EnsembleDynamics.ZERO_MODEL_ID


class SimpleCompetitionHandler(CompetitionHandlingStrategy):

    def implant_model(self, emc: EmbeddingControlManager, wm: WorldModel, ed: EnsembleDynamics):
        if self.is_model_sel_switch and self.prev_model_sel is not EnsembleDynamics.ZERO_MODEL_ID:
            emc.force_model(wm.models[self.prev_model_sel])


class WeightTransferHandler(CompetitionHandlingStrategy):

    def implant_model(self, emc: EmbeddingControlManager, wm: WorldModel, ed: EnsembleDynamics):
        emc.force_model(ed.current_compound_model)


class ModelCompetitionDrivenLC(BabblePerformanceAlternation):
    ##

    _PH_B = "ph"
    _U_B = "u"
    _UEMB_B = "u_emb"
    _Y_B = "y"
    _BASE_GAIT_MEAN_N = 20
    _TRAIN_START_CROP = 200

    def __init__(self, world_model: WorldModel, sensor_dim: int, motor_dim: int, granularity: int,
                 motor_babbler: MB.MotorBabbler, embedding_controller_manager: EmbeddingControlManager,
                 ensemble_dynamics_factory: ENSEMBLE_DYNAMICS_FACTORY, babble_stage_iterations: int,
                 competition_handling_strategy: CompetitionHandlingStrategy,
                 start_with_babble=True, rbf_epsilon=1., integration_step_size=0.01, natural_cpg_frequency=1.,
                 babbling_rate=2., performing_babble_rate=0., force_keep_same_model=False,
                 ):

        super().__init__(world_model, sensor_dim, motor_dim, granularity, motor_babbler, embedding_controller_manager,
                         rbf_epsilon, integration_step_size, natural_cpg_frequency, babbling_rate)
        self._stage_counter = 0
        self._ensemble_dynamics_factory = ensemble_dynamics_factory
        self.performing_babble_rate = performing_babble_rate
        self.motor_copy_embedder = CRE.Embedder(
            dimension=motor_dim,
            granularity=granularity,
            combiner=CRE.mean_combiner()
        )

        self._ensemble_dynamics = self._ensemble_dynamics_factory(
            self.sensor_dim, self.motor_dim, self.granularity, self.wm.models)
        self._babbling_controller = ConstantEmbeddingController(self._base_gait)
        self._performing_controller = ConstantEmbeddingController(self._base_gait)
        self.babble_stage_iterations = babble_stage_iterations

        self._control_routine_buffer = {}

        self.prev_u_output = np.zeros((motor_dim,))

        self.competition_handling_strategy = competition_handling_strategy

        if start_with_babble:
            self.stage = self.StageStates.BABBLING_INIT
        else:
            assert len(world_model) > 0, "Starting with performance stage but there is no model."
            self.stage = self.StageStates.PERFORMANCE_INIT

        #
        self._force_keep_same_model = force_keep_same_model
        if self._force_keep_same_model:
            LOG.info("Will select last model.")
        else:
            LOG.info("Will select models according to dynamics.")

    def _update_control_routine_buffer(self, ph_act, observation, u, u_embedded):
        self._control_routine_buffer[self._PH_B].append(ph_act)
        self._control_routine_buffer[self._U_B].append(u)
        self._control_routine_buffer[self._Y_B].append(observation)
        self._control_routine_buffer[self._UEMB_B].append(u_embedded)

    def _clear_control_routine_buffer(self):
        self._control_routine_buffer = {
            self._PH_B: [],
            self._U_B: [],
            self._UEMB_B: [],
            self._Y_B: [],
        }

    def _rebuild_modules(self):
        self.motor_copy_embedder = CRE.Embedder(
            dimension=self.motor_dim,
            granularity=self.granularity,
            combiner=CRE.mean_combiner()
        )
        self.cpg_rbf = CRE.CpgRbfDiscrete(
            natural_frequency=self._natural_cpg_frequency,
            rbf_epsilon=self._rbf_epsilon,
            granularity=self.granularity,
            step_size=self._integration_step_size
        )

        self.observation_embedder = CRE.Embedder(
            dimension=self.sensor_dim,
            granularity=self.granularity,
            combiner=CRE.mean_combiner()
        )

    def _prepare_control_routine(self):
        self._clear_control_routine_buffer()
        self.motor_babbler.reset()
        ##
        self._rebuild_modules()
        ##
        # Commented on purpose: if this prev_u_output is defaulted, there is an annoying prediction error
        # which occurs at the start. Might be due to data_pipe having the old value aswell?
        # self.prev_u_output = np.zeros((self.motor_dim, ))

        self._ensemble_dynamics = self._ensemble_dynamics_factory(
            self.sensor_dim, self.motor_dim, self.granularity, self.wm.models)

    def _get_collected_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts the learning data. There is one catch: the temporal relation is Y(t-1), U(t), however
        we want to predict future sensor from command (not guess previous sensor from command).
        @return:
        @rtype:
        """
        return np.asarray(self._control_routine_buffer[self._PH_B])[self._TRAIN_START_CROP:-1], \
            np.asarray(self._control_routine_buffer[self._U_B])[self._TRAIN_START_CROP:-1], \
            np.asarray(self._control_routine_buffer[self._Y_B][self._TRAIN_START_CROP + 1:])

    def _update_base_gait(self):
        self._base_gait = np.mean(
            np.asarray(self._control_routine_buffer[self._UEMB_B])[-self._BASE_GAIT_MEAN_N:, :, :], axis=0)

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
        # Copy embedding
        self.motor_copy_embedder.step(phase_activation=ph_act, signal=self.prev_u_output)

        # Babbling update
        self.motor_babbler.step(y_obs_emb, target, phase_activation=ph_act)
        u_babble = self.motor_babbler.current_babble()

        # MOTOR COMMAND
        # inverse model
        u_embedded = embedding_controller(y_obs_emb, target, ph_act_soft)
        # joint embedded command
        u_joint = u_embedded + u_babble * babbling_rate
        ##
        self.competition_handling_strategy.evaluate(
            ed=self._ensemble_dynamics,
            sensory_embedding=y_obs_emb, motor_embedding=self.motor_copy_embedder.current_embedding(),
            target_parameter=target, motion_phase=ph_act
        )
        # de-embedding
        u = CRE.soft_de_embedding(phase_activation=ph_act_soft, embedding=u_joint)
        ##
        self._update_control_routine_buffer(
            ph_act=ph_act,
            observation=observation,
            u=u,
            u_embedded=u_joint
        )
        ##
        self.prev_u_output = u

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
        return np.zeros((self.motor_dim,)), C.CommandType.POSITION_ZERO

    def _babbling_init(self) -> C.COMMAND_T:
        """
        Init just prepares the environment to position zero.
        @return:
        @rtype:
        """
        self._babbling_controller = ConstantEmbeddingController(self._get_base_gait())
        self._prepare_control_routine()
        return np.zeros((self.motor_dim,)), C.CommandType.POSITION_ZERO

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
        u = self._control_routine(
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
        return np.zeros((self.motor_dim,)), C.CommandType.POSITION_ZERO

    def _performing_init(self) -> C.COMMAND_T:
        """
        The performing controller is built with the world model.
        @return:
        @rtype:
        """
        # The new controller is always rebuilt using the last model - the switching can change it later
        if self.embedding_controller_manager.has_variables():
            self._base_gait = self.embedding_controller_manager.get_base_gait()
            base_gait = self._base_gait
        else:
            self._base_gait = self.wm.models[-1].u_mean
            base_gait = self.wm.models[-1].u_mean
        self.embedding_controller_manager.rebuild_controller(base_gait=base_gait, models=self.wm.models)
        ##
        self._prepare_control_routine()
        return np.zeros((self.motor_dim,)), C.CommandType.POSITION_ZERO

    def _switching_model_in_controller(self):
        self.competition_handling_strategy.implant_model(
            self.embedding_controller_manager, self.wm, self._ensemble_dynamics)

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
        if not self._force_keep_same_model:
            self._switching_model_in_controller()
        u = self._control_routine(
            self.embedding_controller_manager.get_controller(), target=target, observation=observation,
            babbling_rate=self.performing_babble_rate)
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
        elif (self.stage == self.StageStates.PERFORMANCE_STAGE and
              self.competition_handling_strategy.is_switched_to_learning()):
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
            observation = np.zeros((self.sensor_dim,))
        ###
        cmd, cmd_t = self._execute_stage(target, observation)
        ###

        REC(self.R_CMD, cmd)
        REC(self.R_CMD_T, cmd_t.value[0])
        REC(self.R_OBSERVATION, observation)
        REC(self.R_STAGE, self.stage.value[0])
        target.write_into_logger(REC, self.R_TARGET)
        return cmd, cmd_t

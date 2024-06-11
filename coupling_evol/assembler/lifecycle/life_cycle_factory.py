from coupling_evol.agent.lifecycle.control_manager.common import EmbeddingControlManager
from coupling_evol.agent.lifecycle.scheduled import RepeatedScheduleLC
from coupling_evol.agent.lifecycle.model_driven import ModelCompetitionDrivenLC, SimpleCompetitionHandler, WeightTransferHandler
from coupling_evol.engine import dynamic_lifecycle as LC
from coupling_evol.agent.components import MotorBabbler
from coupling_evol.assembler.lifecycle.ensemble_dynamics_factory import EnsembleDynamicsFactory, EnsembleDynamicsType
from abc import ABC, abstractmethod
from coupling_evol.assembler.common import *
from typing import Tuple


class LifeCycleType(Enum):
    REPEATED_SCHEDULE = 1
    MODEL_COMPETITION_DRIVEN = 2


class ModelCompositionType(Enum):
    SOFTMAX = "softmax"
    ONEHOT = "one_hot"


class LifeCycleConfiguration(FactoryConfiguration[LifeCycleType]):
    def __init__(self, natural_frequency: float, rbf_epsilon: float, babble_iters: int, delta_search_iters: int,
                 babbling_rate: float):
        super().__init__()
        self.created_type: LifeCycleType = LifeCycleType.MODEL_COMPETITION_DRIVEN
        self.rbf_epsilon = rbf_epsilon
        self.natural_frequency = natural_frequency
        self.babble_iters = babble_iters
        self.delta_search_iters = delta_search_iters
        self.start_with_babbling = True
        self.performance_babble_rate = 0.
        self.babbling_rate = babbling_rate


_FULL_CONFIG = Tuple[LifeCycleConfiguration, EssentialParameters, LifeCycleParameters]


class LifeCycleFactory(ABC):
    def __init__(self, configuration: _FULL_CONFIG):
        self.configuration = configuration[0]
        self.essentials = configuration[1]
        self.lifecycle = configuration[2]

    @abstractmethod
    def __call__(self,
                 world_model: LC.WorldModel, controller_manager: EmbeddingControlManager, motor_babbler: MotorBabbler,
                 ensemble_dynamics_factory: EnsembleDynamicsFactory
                 ):
        pass


class RepeatedScheduleLCFactory(LifeCycleFactory):
    def __call__(self,
                 world_model: LC.WorldModel, controller_manager: EmbeddingControlManager, motor_babbler: MotorBabbler,
                 ensemble_dynamics_factory: EnsembleDynamicsFactory):
        cnf = self.configuration
        return RepeatedScheduleLC(
            world_model=world_model, sensor_dim=self.essentials.sensory_dimension,
            motor_dim=self.essentials.motor_dimension,
            granularity=self.lifecycle.granularity,
            motor_babbler=motor_babbler, embedding_controller_manager=controller_manager,
            babble_stage_iterations=cnf.babble_iters,
            performance_stage_iterations=cnf.delta_search_iters,
            start_with_babble=cnf.start_with_babbling,
            rbf_epsilon=cnf.rbf_epsilon,
            integration_step_size=self.essentials.integration_step_size,
            natural_cpg_frequency=cnf.natural_frequency,
            babbling_rate=cnf.babbling_rate
        )


class ModelCompetitionDrivenLCFactory(LifeCycleFactory):
    def __call__(self,
                 world_model: LC.WorldModel, controller_manager: EmbeddingControlManager, motor_babbler: MotorBabbler,
                 ensemble_dynamics_factory: EnsembleDynamicsFactory):
        cnf = self.configuration

        competition_handler = SimpleCompetitionHandler()
        if any([
            ensemble_dynamics_factory.configuration.created_type is EnsembleDynamicsType.TWO_STAGE_AGGREGATE_COMPOSITION,
            ensemble_dynamics_factory.configuration.created_type is EnsembleDynamicsType.TWO_STAGE_AGGREGATE_ODOMETRYSUM_COMPOSITION
        ]):
            competition_handler = WeightTransferHandler()

        return ModelCompetitionDrivenLC(
            world_model=world_model, sensor_dim=self.essentials.sensory_dimension,
            motor_dim=self.essentials.motor_dimension,
            granularity=self.lifecycle.granularity,
            motor_babbler=motor_babbler, embedding_controller_manager=controller_manager,
            babble_stage_iterations=cnf.babble_iters,
            ensemble_dynamics_factory=ensemble_dynamics_factory,
            start_with_babble=cnf.start_with_babbling,
            rbf_epsilon=cnf.rbf_epsilon,
            integration_step_size=self.essentials.integration_step_size,
            natural_cpg_frequency=cnf.natural_frequency,
            babbling_rate=cnf.babbling_rate,
            performing_babble_rate=cnf.performance_babble_rate,
            force_keep_same_model=self.lifecycle.force_keep_same_model,
            competition_handling_strategy=competition_handler
        )


_ENUM_FACTORY_MAP = {
    LifeCycleType.REPEATED_SCHEDULE: RepeatedScheduleLCFactory,
    LifeCycleType.MODEL_COMPETITION_DRIVEN: ModelCompetitionDrivenLCFactory
}


def dynamic_lifecycle_factory(lifecycle_config: LifeCycleConfiguration, essentials: EssentialParameters,
                              lifecycle: LifeCycleParameters):
    return _ENUM_FACTORY_MAP[lifecycle_config.created_type]((lifecycle_config, essentials, lifecycle))

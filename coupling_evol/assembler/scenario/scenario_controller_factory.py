from coupling_evol.engine.experiment_executor import ScenarioController
from coupling_evol.engine import scenarios
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from coupling_evol.assembler.common import *

from coupling_evol.engine.scenarios.combiners import TimedSwitch
from coupling_evol.engine.scenarios.stateless import SensoryLinearTransform, MotorLinearTransform


class ScenarioControllerType(Enum):
    NONE = 1
    TIMED_INVERT_SENSORS = 2
    TIMED_INVERT_FIRST_LEG = 3
    TIMED_PARALYZE_FIRST_LEG = 4
    DISTANCE_PARALYZE_FIRST_LEG = 5
    ##
    TIMED_PARALYZE_FIRST_FOURTH_LEGS = 6
    TIMED_PARALYZE_FIRST_SIXTH_LEGS = 7
    TIMED_PARALYZE_THIRD_SIXTH_LEGS = 8
    ##
    TIMED_PARALYZE_THIRD_LEG = 9
    TIMED_PARALYZE_FIFTH_LEG = 10


class ScenarioControllerConfiguration(FactoryConfiguration[ScenarioControllerType]):
    def __init__(self):
        super().__init__()
        self.created_type = ScenarioControllerType.NONE
        self.switch_fractions = (-1,)
        self.switch_distances = (-1,)
        self.center = (0, 0)


SCENARIO_CONTROLLER_FULL = Tuple[ScenarioControllerConfiguration, EssentialParameters, ExperimentSetupParameters]


class ScenarioControlFactory(ABC):
    def __init__(self, full_configuration: SCENARIO_CONTROLLER_FULL):
        self.configuration = full_configuration[0]
        self.essential_param = full_configuration[1]
        self.experiment_param = full_configuration[2]

    @abstractmethod
    def __call__(self) -> ScenarioController:
        pass


class UnchangingScenarioFactory(ScenarioControlFactory):
    @abstractmethod
    def __call__(self) -> ScenarioController:
        return ScenarioController()


class TimedSwitchFactory(ScenarioControlFactory, ABC):
    def __init__(self, full_configuration: SCENARIO_CONTROLLER_FULL):
        super().__init__(full_configuration)
        self.switch_times = [int(frac * self.experiment_param.max_iters) for frac in
                             self.configuration.switch_fractions]


class TimedInvertSensorsFactory(TimedSwitchFactory):
    def __call__(self):
        first = ScenarioController()
        second = SensoryLinearTransform(shift=0., scale=-1.)
        return TimedSwitch(
            switch_iteration=self.switch_times[0],
            switch_back_iteration=self.switch_times[1],
            first_scenario_controller=first,
            second_scenario_controller=second
        )


def _get_joint_command_alternation(from_tos: List[Tuple[int, int]], scaling: float, motor_dimension: int):
    alternation = np.ones((motor_dimension,))
    for fr, to in from_tos:
        alternation[fr:to] = scaling
    return alternation


class TimedCommandAlternationFactory(TimedSwitchFactory):
    def __init__(self, full_configuration: SCENARIO_CONTROLLER_FULL,
                 from_tos: List[Tuple[int, int]], scaling: float):
        super().__init__(full_configuration)
        self.alternation = _get_joint_command_alternation(from_tos, scaling, self.essential_param.motor_dimension)

    def __call__(self):
        first = ScenarioController()
        second = MotorLinearTransform(shift=0., scale=self.alternation)
        return scenarios.combiners.TimedSwitch(
            switch_iteration=self.switch_times[0],
            switch_back_iteration=self.switch_times[1],
            first_scenario_controller=first,
            second_scenario_controller=second
        )

    @classmethod
    def inversion(cls, from_tos: List[Tuple[int, int]]):
        return lambda configuration: cls(full_configuration=configuration, from_tos=from_tos, scaling=-1)

    @classmethod
    def paralysis(cls, from_tos: List[Tuple[int, int]]):
        return lambda configuration: cls(full_configuration=configuration, from_tos=from_tos, scaling=0)


class DistancedCommandAlternationFactory(ScenarioControlFactory):

    def __init__(self, full_configuration: SCENARIO_CONTROLLER_FULL,
                 from_tos: List[Tuple[int, int]], scaling: float):
        super().__init__(full_configuration)
        self.alternation = _get_joint_command_alternation(from_tos, scaling, self.essential_param.motor_dimension)
        self.center = self.configuration.center
        self.switch_distances = self.configuration.switch_distances

    def __call__(self) -> ScenarioController:
        first = ScenarioController()
        second = MotorLinearTransform(shift=0., scale=self.alternation)
        return scenarios.combiners.NeighbourhoodDistanceSwitch(
            neighbourhood_center=self.center,
            switch_distance=self.switch_distances[0],
            switch_back_distance=self.switch_distances[1],
            first_scenario_controller=first,
            second_scenario_controller=second
        )

    @classmethod
    def paralysis(cls, from_tos: List[Tuple[int, int]]):
        return lambda configuration: cls(full_configuration=configuration, from_tos=from_tos, scaling=0)


_ENUM_FACTORY_MAP = {
    ScenarioControllerType.NONE: UnchangingScenarioFactory,
    ScenarioControllerType.TIMED_INVERT_SENSORS: TimedInvertSensorsFactory,
    ScenarioControllerType.TIMED_INVERT_FIRST_LEG: TimedCommandAlternationFactory.inversion([(0, 3)]),
    ScenarioControllerType.TIMED_PARALYZE_FIRST_LEG: TimedCommandAlternationFactory.paralysis([(0, 3)]),
    ScenarioControllerType.TIMED_PARALYZE_THIRD_LEG: TimedCommandAlternationFactory.paralysis([(6, 9)]),
    ScenarioControllerType.TIMED_PARALYZE_FIFTH_LEG: TimedCommandAlternationFactory.paralysis([(12, 15)]),
    ScenarioControllerType.TIMED_PARALYZE_FIRST_FOURTH_LEGS: TimedCommandAlternationFactory.paralysis(
        [(0, 3), (9, 12)]),
    ScenarioControllerType.TIMED_PARALYZE_FIRST_SIXTH_LEGS: TimedCommandAlternationFactory.paralysis(
        [(0, 3), (15, 18)]),
    ScenarioControllerType.TIMED_PARALYZE_THIRD_SIXTH_LEGS: TimedCommandAlternationFactory.paralysis(
        [(6, 9), (15, 18)]),
    ScenarioControllerType.DISTANCE_PARALYZE_FIRST_LEG: DistancedCommandAlternationFactory.paralysis([(0, 3)])
}


def factory(config: ScenarioControllerConfiguration,
            essential: EssentialParameters, experiment_setup: ExperimentSetupParameters) -> ScenarioController:
    return _ENUM_FACTORY_MAP[config.created_type]((config, essential, experiment_setup))()

from coupling_evol.engine.experiment_executor import ScenarioController
import numpy as np
from coupling_evol.engine import common as C
from typing import Tuple

import coupling_evol.data_process.inprocess.recorder as rlog
from coupling_evol.engine import common as C
REC = rlog.get_recorder(C.RecordNamespace.SCENARIO_CONTROLLER.key)

R_SWITCH = "switch"


class TimedSwitch(ScenarioController):
    def __init__(self, switch_iteration: int, switch_back_iteration: int,
                 first_scenario_controller: ScenarioController,
                 second_scenario_controller: ScenarioController):
        super().__init__()
        assert switch_iteration < switch_back_iteration
        self.switch_iteration = switch_iteration
        self.switch_back_iteration = switch_back_iteration
        self._iter_count = 0
        self._observation_q = 1
        self._first = first_scenario_controller
        self._second = second_scenario_controller
        self._current = self._first


    def update(self, environment_data: np.ndarray):
        if self.switch_iteration < self._iter_count < self.switch_back_iteration:
            self._current = self._second
            REC(R_SWITCH, 1.)
        else:
            self._current = self._first
            REC(R_SWITCH, 0.)

        self._iter_count += 1
        self._current.update(environment_data)

    def command_transform(self, cmd_val: np.ndarray) -> np.ndarray:
        return self._current.command_transform(cmd_val)

    def observation_transform(self, observation: np.ndarray) -> np.ndarray:
        return self._current.observation_transform(observation)

    def run_end(self) -> bool:
        return self._current.run_end()


class NeighbourhoodDistanceSwitch(ScenarioController):
    def __init__(self, neighbourhood_center: Tuple[float, float],
                 switch_distance: float, switch_back_distance: float,
                 first_scenario_controller: ScenarioController,
                 second_scenario_controller: ScenarioController,
                 xy_location_data_ids=(0, 1)
                 ):
        super().__init__()
        assert switch_distance > switch_back_distance
        assert len(neighbourhood_center) == 2

        self.switch_distance = switch_distance
        self.switch_back_distance = switch_back_distance
        self.center = np.asarray(neighbourhood_center)
        self.xy_ids = list(xy_location_data_ids)

        self._first = first_scenario_controller
        self._second = second_scenario_controller
        self._current = self._first

        self._rank = 0

    def update(self, environment_data: np.ndarray):
        distance = np.linalg.norm(environment_data[self.xy_ids] - self.center)
        if self._rank == 0 and distance < self.switch_distance:
            self._rank = 1
        elif self._rank == 1 and distance < self.switch_back_distance:
            self._rank = 2

        if self._rank == 1:
            self._current = self._second
            REC(R_SWITCH, 1.)
        else:
            self._current = self._first
            REC(R_SWITCH, 0.)

        self._current.update(environment_data)

    def command_transform(self, cmd_val: np.ndarray) -> np.ndarray:
        return self._current.command_transform(cmd_val)

    def observation_transform(self, observation: np.ndarray) -> np.ndarray:
        return self._current.observation_transform(observation)

    def run_end(self) -> bool:
        return self._current.run_end()

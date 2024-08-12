import numpy as np

from coupling_evol.engine.dynamic_lifecycle import DynamicLifecycle
from coupling_evol.engine.environment import Environment
import time
from coupling_evol.engine import common as C
from coupling_evol.engine.embedded_control import EmbeddedTargetParameter
from typing import Dict, Callable, Union
import logging

LOG = logging.getLogger(__name__)

import coupling_evol.data_process.inprocess.recorder as rlog
from coupling_evol.engine import common as C
REC = rlog.get_recorder(C.RecordNamespace.EXECUTOR.key)

R_T = "t"
R_TIMESTAMP = "timestamp"
R_DURAVG = "duravg"


class DataPipe(object):
    """
    Loads data from environment and stores them for the lifecycle
    and target provider
    """

    def __init__(self):
        pass

    def get_lifecycle_data(self) -> np.ndarray:
        pass

    def get_target_provider_data(self) -> np.ndarray:
        pass

    def get_scenario_controller_data(self) -> np.ndarray:
        pass

    def __call__(self, environment_data_raw: np.ndarray):
        pass


class TargetProvider(object):
    """
    Provides the target
    """

    def __init__(self):
        pass

    def __call__(self, data: np.ndarray) -> EmbeddedTargetParameter:
        pass


class ScenarioController(object):
    """
    Represents meta control of the experiment run.
    """

    def __init__(self):
        pass

    def update(self, environment_data: np.ndarray):
        pass

    def command_transform(self, cmd_val: np.ndarray) -> np.ndarray:
        return cmd_val

    def observation_transform(self, observation: np.ndarray) -> np.ndarray:
        return observation

    def run_end(self) -> bool:
        return False


def empty_callback(d: dict):
    pass


class ExperimentExecutor(object):
    """
    Runnable session. Similar to CollectionSetup
    """
    _DUR_AVG_WINDOW = 20

    def __init__(self,
                 lifecycle: DynamicLifecycle,
                 environment: Environment,
                 target_provider: TargetProvider,
                 data_pipe: DataPipe,
                 scenario_controller: ScenarioController
                 ):
        self.lifecycle = lifecycle
        self.environment = environment
        self.target_provider = target_provider
        self.data_pipe = data_pipe
        self.scenario_controller = scenario_controller
        self._was_run = False

    def _init_run(self):
        self._was_run = True
        self._environment_session = self.environment.create_session()

    def _close_run(self):
        rlog.save_and_flush(wait_for_it=True)
        self.environment.end_session()

    def _control_step(self):
        self.scenario_controller.update(self.data_pipe.get_scenario_controller_data())
        y_target = self.target_provider(self.data_pipe.get_target_provider_data())
        cmd_val, cmd_type = self.lifecycle(
            y_target,
            self.scenario_controller.observation_transform(self.data_pipe.get_lifecycle_data())
        )
        if cmd_type == C.CommandType.DIRECT:
            y_last_raw = self._environment_session.step(
                self.scenario_controller.command_transform(cmd_val)
            )
            self.data_pipe(y_last_raw)
        elif cmd_type == C.CommandType.POSITION_ZERO:
            self.environment.position_zero()
            rlog.save_and_flush()
        else:
            raise NotImplemented(f"Unknown CommandType {cmd_type}.")

    def run(self, max_iters: int, d_t: float, step_sleep: float):
        assert not self._was_run, "This executor was already executed"
        self._init_run()
        LOG.info(
            f"Running experiment with {max_iters} which will take ~{max_iters * step_sleep / 60} mins, integration set on {d_t}.")
        #############
        t = 0
        begin_time = float(time.time())
        end_time = float(time.time())
        dur_data = [step_sleep / 2] * self._DUR_AVG_WINDOW
        for i in range(max_iters):
            start_time = float(time.time())
            self._control_step()
            end_time = float(time.time())
            t += d_t
            # Management
            dur_data[i % self._DUR_AVG_WINDOW] = (end_time - start_time)
            dur_avg = np.average(dur_data)
            time.sleep(np.maximum(step_sleep - dur_avg, 0))

            REC(R_TIMESTAMP, float(end_time))
            REC(R_T, float(t))
            REC(R_DURAVG, dur_avg)
            rlog.increment()
            if self.scenario_controller.run_end():
                break

        ##########
        LOG.info(f"The experiment is finished with ~{int(t / d_t)} taking {(end_time - begin_time) / 60} mins")
        self._close_run()

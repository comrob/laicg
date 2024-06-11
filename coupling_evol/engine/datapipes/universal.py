from coupling_evol.engine.experiment_executor import DataPipe
import numpy as np


class Forwarder(DataPipe):

    def __init__(self):
        super().__init__()
        self._data = np.empty((0,))

    def get_lifecycle_data(self) -> np.ndarray:
        return self._data

    def get_target_provider_data(self) -> np.ndarray:
        return self._data

    def get_scenario_controller_data(self) -> np.ndarray:
        return self._data

    def __call__(self, environment_data_raw: np.ndarray):
        self._data = environment_data_raw

from coupling_evol.engine.experiment_executor import DataPipe
import numpy as np
from coupling_evol.environments.hebi.hebi_environment import HRPYS_DIMENSION, EFF_DIMENSION, XYZ_DIMENSION, Q_DIMENSION

Y_SIZE = HRPYS_DIMENSION + EFF_DIMENSION + XYZ_DIMENSION + Q_DIMENSION
class HebiPipeWithEffort(DataPipe):
    def __init__(self):
        super().__init__()
        self._data = np.zeros((Y_SIZE, ))

    def get_lifecycle_data(self) -> np.ndarray:
        return self._data[:HRPYS_DIMENSION + EFF_DIMENSION]

    def get_target_provider_data(self) -> np.ndarray:
        return self._data[HRPYS_DIMENSION + EFF_DIMENSION:]

    def get_scenario_controller_data(self) -> np.ndarray:
        return self._data

    def __call__(self, environment_data_raw: np.ndarray):
        self._data = environment_data_raw

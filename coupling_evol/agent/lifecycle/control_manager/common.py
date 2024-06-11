import numpy as np
from coupling_evol.engine.embedded_control import EmbeddingController
from typing import Dict, Union
from coupling_evol.data_process.inprocess import records as R
import os
from coupling_evol.agent.components.internal_model.forward_model import MultiPhaseModel
from typing import List
from abc import ABC, abstractmethod



SNAPSHOT = Dict[str, Union[np.ndarray, float]]


class EmbeddingControlManager(ABC):
    VARIABLE_STORAGE_FILENAME = "vars.hdf5"
    VARIABLE_MEAN_SIZE = 20

    def __init__(self, directory_path):
        self.directory_path = directory_path
        self._controller: EmbeddingController = EmbeddingController()

    @abstractmethod
    def rebuild_controller(self, base_gait: np.ndarray, models: List[MultiPhaseModel], **kwargs):
        """
        Here the stored variables are loaded and the controller is rebuild.
        @param base_gait: new base gait
        @param models: new list of models for the controller
        @param kwargs:
        @return:
        """
        pass

    @abstractmethod
    def get_base_gait(self) -> np.ndarray:
        pass

    @abstractmethod
    def force_model(self, model: MultiPhaseModel):
        """
        Online switches the model in the current controller.
        @param model:
        """
        pass

    def get_controller(self) -> EmbeddingController:
        return self._controller

    def has_variables(self):
        return self.is_variables_stored(self.directory_path)

    def load_controller_variables(self) -> SNAPSHOT:
        return self.load_variables(self.directory_path)

    def save_controller_variables(self):
        """
        Looks inside the self._controller and takes what is needed.
        @return:
        @rtype:
        """
        hist = self._controller.get_history()
        rec = R.numpyfy_record(hist)
        vars = {}
        for k in rec:
            vars[k] = np.mean(rec[k][-self.VARIABLE_MEAN_SIZE:], axis=0)
        self.save_variables(self.directory_path, variables=vars)

    @classmethod
    def save_variables(cls, directory_path: str, variables: Dict[str, Union[np.ndarray, float]]):
        R.save_records(os.path.join(directory_path, cls.VARIABLE_STORAGE_FILENAME), [variables])

    @classmethod
    def load_variables(cls, directory_path: str) -> Dict[str, Union[np.ndarray, float]]:
        return R.load_records(os.path.join(directory_path, cls.VARIABLE_STORAGE_FILENAME))[0]

    @classmethod
    def is_variables_stored(cls, directory_path: str):
        return os.path.exists(os.path.join(directory_path, cls.VARIABLE_STORAGE_FILENAME))



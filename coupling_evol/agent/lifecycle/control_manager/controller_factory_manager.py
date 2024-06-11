from coupling_evol.agent.lifecycle.compound_model import CompoundModel
from coupling_evol.agent.lifecycle.control_manager.common import EmbeddingControlManager, SNAPSHOT
from coupling_evol.agent.components.internal_model.forward_model import MultiPhaseModel
import numpy as np
from typing import List
import coupling_evol.agent.components.controllers.fep_controllers as FEP
from abc import ABC, abstractmethod

from coupling_evol.engine.embedded_control import EmbeddingController

MODELS = List[MultiPhaseModel]


class EmbeddingControllerFactory(ABC):

    @abstractmethod
    def create(self, models: MODELS, variables: SNAPSHOT, **kwargs) -> EmbeddingController:
        """
        Creates new EmbeddingController
        """
        pass


class ControllerFactoryManager(EmbeddingControlManager):

    def __init__(self, directory_path, factory: EmbeddingControllerFactory):
        super().__init__(directory_path)
        self._factory = factory

    def rebuild_controller(self, base_gait: np.ndarray, models: MODELS, **kwargs):
        # TODO How to deal with compound models?, Shouldn't be the world model given?
        assert len(models) > 0, "Creating a controller without any model is not possible."
        if self.has_variables():
            variables = self.load_controller_variables()
        else:
            variables = {}
        self._controller = self._factory.create(models=models, variables=variables)

    def get_base_gait(self) -> np.ndarray:
        return self.load_controller_variables()[FEP.BASE_GAIT]

    def force_model(self, model: MultiPhaseModel):
        if hasattr(self._controller, "model"):
            self._controller.model = model  # TODO this shouldnt be done by direct access
        else:
            print(f"WARNING: {type(self._controller)} doesn't have model attribute, nothing happens then.")


class MultiControllerFactoryManager(ControllerFactoryManager):
    """Wrapper spawning multiple FEP controllers for different models and putting them into MultiFepControllerContainer"""

    def __init__(self, directory_path, factory: EmbeddingControllerFactory):
        super().__init__(directory_path, factory)

    def rebuild_controller(self, base_gait: np.ndarray, models: List[MultiPhaseModel], **kwargs):
        assert len(models) > 0, "Creating a controller without any model is not possible."
        controllers = []
        if self.has_variables():
            variables = self.load_controller_variables()
        else:
            variables = {}
        for model in models:
            # variables = {}
            controllers.append(self._factory.create(models=[model], variables=variables))
        self._controller = FEP.MultiFepControllerContainer(controllers)
        self._controller.set_model_motor_phase_weights(
            np.ones((len(models), models[0].u_dim, models[0].phase_n)) / len(models))

    def get_base_gait(self) -> np.ndarray:
        return self.load_controller_variables()[FEP.BASE_GAIT]

    def force_model(self, model: CompoundModel):
        model_motor_phase_weights = np.ones((len(model._models), model.u_dim, model.phase_n)) * np.average(
            model.get_model_modality_phase_weights(), axis=(1, 2))[:, None, None]
        self._controller.set_model_motor_phase_weights(model_motor_phase_weights)

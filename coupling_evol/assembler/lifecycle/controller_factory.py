from abc import ABC

from coupling_evol.agent.lifecycle.control_manager.controller_factory_manager import (EmbeddingControllerFactory, SNAPSHOT, MODELS, EmbeddingController)
from enum import Enum
from coupling_evol.assembler.common import *
from coupling_evol.agent.components.controllers.fep_controllers import WaveFepFusionController, TargetErrorProcessing


class ControllerType(Enum):
    WAVE_FEP_FUSION = 1


class ControllerConfiguration(FactoryConfiguration[ControllerType]):
    def __init__(self):
        super().__init__()
        self.arguments = {}
        self.created_type: ControllerType = ControllerType.WAVE_FEP_FUSION


class ConfiguredControllerFactory(EmbeddingControllerFactory, ABC):
    def __init__(self, configuration: ControllerConfiguration):
        self.configuration: ControllerConfiguration = configuration


class WaveFepFusionFactory(ConfiguredControllerFactory):

    def __init__(self, configuration: ControllerConfiguration):
        super().__init__(configuration)

    def create(self, models: MODELS, variables: SNAPSHOT, **kwargs) -> EmbeddingController:
        cnf = self.configuration.arguments
        perc = WaveFepFusionController(
            model=models[-1], prev_control_snap=variables,
            prior_strength=cnf["log_prior_strength"],
            gait_learning_rate=cnf["gait_learning_rate"],
            likelihood_variance_learning_rate=cnf["likelihood_variance_learning_rate"],
            symmetry_strength=cnf["amplitude_symmetry_strength"],
            observed_variance=cnf["observed_sensory_variance"],
            estimation_prior_variance=cnf["estimation_prior_variance"],
            estimation_learning_rate=cnf["estimation_learning_rate"],
            prediction_variance_lower_bound=cnf["prediction_variance_lower_bound"],
            target_error_variance_learning_rate=cnf["target_error_variance_learning_rate"],
            prediction_precision_scale=cnf["prediction_precision_scale"],
            target_precision_scale=cnf["target_precision_scale"],
            target_error_processing=TargetErrorProcessing.from_int(cnf["target_error_processing"]),
            sum_amplitude_energy=cnf["sum_amplitude_energy"],
            sum_amplitude_energy_strength=cnf["sum_amplitude_energy_strength"],
            simplified_gradient_switch=cnf["simplified_gradient_switch"]
        )
        ## hist
        return perc


_ENUM_FACTORY_MAP = {
    ControllerType.WAVE_FEP_FUSION: WaveFepFusionFactory
}


def get_controller_factory(configuration: ControllerConfiguration) -> ConfiguredControllerFactory:
    return _ENUM_FACTORY_MAP[configuration.created_type](configuration)

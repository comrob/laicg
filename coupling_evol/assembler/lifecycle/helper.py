from coupling_evol.agent.lifecycle.control_manager.common import EmbeddingControlManager
from coupling_evol.agent.lifecycle.control_manager.controller_factory_manager import MultiControllerFactoryManager, \
    ControllerFactoryManager
from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import EmbeddedStagedLC
from coupling_evol.agent.components.internal_model.regressors.sklearn_regressors import StdNormedLinearRegressor

import coupling_evol.engine.dynamic_lifecycle as LC
import numpy as np
import coupling_evol.assembler.lifecycle.motor_babbling_factory as MBH
from typing import List
from coupling_evol.agent.components.internal_model.forward_model import MultiPhaseModel
##
from coupling_evol.assembler.lifecycle import controller_factory as CF

from coupling_evol.assembler.lifecycle.life_cycle_factory import dynamic_lifecycle_factory
from coupling_evol.data_process.postprocess.lifecycle_data import ExperimentExecutorPaths
from coupling_evol.assembler import *
from coupling_evol.assembler.lifecycle import ensemble_dynamics_factory as EDF


def create_controller_manager(controller_path: str,
                              configuration: ControllerConfiguration) -> EmbeddingControlManager:
    if ("use_fep_controller_container" in configuration.arguments and
            configuration.arguments["use_fep_controller_container"]):
        return MultiControllerFactoryManager(controller_path, CF.get_controller_factory(configuration))
    else:
        return ControllerFactoryManager(controller_path, CF.get_controller_factory(configuration))


def create(
        paths: ExperimentExecutorPaths,
        lifecycle_config: LifeCycleConfiguration,
        controller_configuration: ControllerConfiguration,
        ensemble_dynamics_configuration: EnsembleDynamicsConfiguration,
        babbler_param: BabblerParameterization,
        essential_param: EssentialParameters,
        lifecycle_param: LifeCycleParameters,
        transferred_models: List[MultiPhaseModel],
        transferred_controller_variables: Dict[str, Union[np.ndarray, float]]
) -> EmbeddedStagedLC:

    wm = LC.WorldModel.create_with_models(
        models=transferred_models,
        directory_path=paths.ensemble_path(), regressor_builder=lambda: StdNormedLinearRegressor(),
        transgait_window_size=1, force_overwrite=False
    )

    ctr_m = create_controller_manager(paths.controller_path(), controller_configuration)

    if len(transferred_controller_variables) > 0:
        ctr_m.save_variables(paths.controller_path(), transferred_controller_variables)

    bbl = MBH.get_parametrized_babbler(
        bp=babbler_param,
        motor_dimension=essential_param.motor_dimension, granularity=lifecycle_param.granularity)

    ensemble_dynamics_factory = EDF.factory(ensemble_dynamics_configuration, essentials=essential_param,
                                            lifecycle=lifecycle_param)

    return dynamic_lifecycle_factory(lifecycle_config=lifecycle_config, essentials=essential_param,
                                     lifecycle=lifecycle_param)(
        world_model=wm, controller_manager=ctr_m, motor_babbler=bbl, ensemble_dynamics_factory=ensemble_dynamics_factory)

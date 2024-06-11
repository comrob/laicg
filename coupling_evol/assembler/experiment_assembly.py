from coupling_evol.agent.lifecycle.control_manager.common import EmbeddingControlManager
from coupling_evol.data_process.inprocess.record_logger import RecordLogger
import coupling_evol.assembler.scenario.helper as S_H
import coupling_evol.assembler.lifecycle.helper as LC_H
import coupling_evol.engine.experiment_executor as EE
from coupling_evol.data_process.postprocess.lifecycle_data import ExperimentExecutorPaths
from coupling_evol.engine.environment import Environment
import os
from coupling_evol.agent.components.internal_model.regressors.sklearn_regressors import StdNormedLinearRegressor
import coupling_evol.engine.dynamic_lifecycle as LC
import logging
from coupling_evol.assembler import *
import coupling_evol.data_process.inprocess.recorder as rlog


LOG = logging.getLogger(__name__)


class AssemblyConfiguration:
    def __init__(self,
                 essential_parameters: EssentialParameters,
                 lifecycle_parameters: LifeCycleParameters,
                 experiment_setup_parameters: ExperimentSetupParameters,
                 babbler_parametrization: BabblerParameterization,
                 controller_configuration: ControllerConfiguration,
                 ensemble_dynamics_configuration: EnsembleDynamicsConfiguration,
                 life_cycle_configuration: LifeCycleConfiguration,
                 scenario_controller_configuration: ScenarioControllerConfiguration,
                 provider_configuration: ProviderConfiguration
                 ):
        self.essential_parameters: EssentialParameters = essential_parameters
        self.lifecycle_parameters: LifeCycleParameters = lifecycle_parameters
        self.experiment_setup_parameters: ExperimentSetupParameters = experiment_setup_parameters
        self.babbler_parametrization: BabblerParameterization = babbler_parametrization
        self.controller_configuration: ControllerConfiguration = controller_configuration
        self.ensemble_dynamics_configuration: EnsembleDynamicsConfiguration = ensemble_dynamics_configuration
        self.life_cycle_configuration: LifeCycleConfiguration = life_cycle_configuration
        self.scenario_controller_configuration: ScenarioControllerConfiguration = scenario_controller_configuration
        self.provider_configuration: ProviderConfiguration = provider_configuration
        ##


def _prepare_directory_structure(results_path: str, experiment_name: str,
                                 overwrite_files=False, safe_files=True):
    assert not (overwrite_files and safe_files), "overwriting and being safe cannot go together ..."
    root_dir = os.path.join(results_path, experiment_name)
    ##
    if os.path.exists(root_dir) and safe_files:
        raise ValueError(f"Experiment directory {root_dir} already exists!")
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    ##
    paths = ExperimentExecutorPaths(results_path=results_path, collection_name=experiment_name)
    if paths.any_exists():
        print(f"WARNING: There are existing files in {root_dir}, they can be modified or deleted.")
    if overwrite_files:
        paths.purge_dirs()
    paths.mkdirs()
    return paths


def assemble_experiment_executor(results_path: str, experiment_name: str,
                                 assembly_configuration: AssemblyConfiguration,
                                 environment: Environment,
                                 overwrite_files=False, clean_files=True,
                                 transferred_ensemble_path: str = None,
                                 transferred_controller_path: str = None
                                 ):
    """
    This factory uses factories from gait_bootstrap package.
    """
    paths = _prepare_directory_structure(results_path, experiment_name,
                                         overwrite_files=overwrite_files, safe_files=clean_files)
    ##
    transferred_models = []
    if transferred_ensemble_path is not None:
        LOG.info(f"Transferring models from {transferred_ensemble_path} into {paths.ensemble_path()}.")
        wm = LC.WorldModel(transferred_ensemble_path, regressor_builder=lambda: StdNormedLinearRegressor())
        transferred_models = wm.models

    if transferred_controller_path is not None:
        transferred_controller_variables = EmbeddingControlManager.load_variables(transferred_controller_path)
        LOG.info(f"Transferring controller variables from {transferred_controller_path} into {paths.controller_path()}.")
    else:
        transferred_controller_variables = {}
    ##
    life_cycle = LC_H.create(
        paths=paths, lifecycle_config=assembly_configuration.life_cycle_configuration,
        controller_configuration=assembly_configuration.controller_configuration,
        ensemble_dynamics_configuration=assembly_configuration.ensemble_dynamics_configuration,
        lifecycle_param=assembly_configuration.lifecycle_parameters,
        babbler_param=assembly_configuration.babbler_parametrization,
        essential_param=assembly_configuration.essential_parameters,
        transferred_models=transferred_models,
        transferred_controller_variables=transferred_controller_variables
    )

    scenario = S_H.create(
        target_config=assembly_configuration.provider_configuration,
        scenario_controller_config=assembly_configuration.scenario_controller_configuration,
        essential=assembly_configuration.essential_parameters,
        experiment_setup=assembly_configuration.experiment_setup_parameters,
        lifecycle=assembly_configuration.lifecycle_parameters,
        environment=environment
    )

    rlog.reset(paths.snaps_path(), buffer_max_size=1000)

    ee = EE.ExperimentExecutor(
        lifecycle=life_cycle,
        data_pipe=scenario.data_pipe,
        scenario_controller=scenario.scenario_controller,
        target_provider=scenario.target_provider,
        environment=scenario.environment
    )

    return ee

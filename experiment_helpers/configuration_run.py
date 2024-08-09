# from vrep_experiments_confs import *  # Windows + multiprocessing fix: https://stackoverflow.com/questions/63206414/multiprocessing-process-calls-main-function-from-process-creation-line-to-end-of
import coupling_evol.assembler.experiment_assembly as EE_H
from coupling_evol.assembler import ProviderConfiguration, ScenarioControllerConfiguration
from coupling_evol.data_process.inprocess import callbacks
import coupling_evol.engine.environment as ENV
from typing import Tuple, Callable
import logging
from experiment_helpers import configuration_build as C
from experiment_helpers.configuration_build import build_assembly_configuration
import os
import numpy as np
import coupling_evol.data_process.inprocess.recorder as rlog
import copy

###
LOG = logging.getLogger(__name__)


def prepare_and_run_experiment_executor(ac: EE_H.AssemblyConfiguration,
                                        env: ENV.Environment, collection_name, collection_results_path,
                                        overwrite=False,
                                        transferred_ensemble_path: str = None,
                                        transferred_controller_path: str = None
                                        ):
    assert transferred_ensemble_path is not None or transferred_controller_path is None, "Transfering the controller without the ensemble is not supported."

    ee = EE_H.assemble_experiment_executor(
        results_path=collection_results_path, experiment_name=collection_name,
        assembly_configuration=ac, environment=env, overwrite_files=overwrite,
        clean_files=not overwrite, transferred_ensemble_path=transferred_ensemble_path,
        transferred_controller_path=transferred_controller_path
    )
    rlog.set_callback(callbacks.competing_fep_callback, callback_period=500)

    ee.run(max_iters=ac.experiment_setup_parameters.max_iters,
           d_t=ac.essential_parameters.integration_step_size,
           step_sleep=ac.experiment_setup_parameters.step_sleep)


def standard_run(clc: C.ComplexLifecycleConfiguration,
                 unique_collection_name, environment: ENV.Environment,
                 provider_cfg: ProviderConfiguration,
                 scenario_controller_cfg: ScenarioControllerConfiguration,
                 results_path, context_steps, overwrite=False,
                 transferred_ensemble_path: str = None,
                 transferred_controller_path: str = None,
                 ):
    clc = copy.deepcopy(clc)

    if transferred_ensemble_path is not None and not os.path.exists(transferred_ensemble_path):
        LOG.error(f"Transfering ensemble from {transferred_ensemble_path} but it does not exist.")
        raise ValueError(f"Transfering ensemble from {transferred_ensemble_path} but it does not exist.")

    if transferred_controller_path is not None and not os.path.exists(transferred_controller_path):
        LOG.error(f"Transfering controller from {transferred_controller_path} but it does not exist.")
        raise ValueError(f"Transfering controller from {transferred_controller_path} but it does not exist.")


    ##
    if transferred_ensemble_path is not None:
        LOG.info("Trial has transferred ensemble, thus it starts with performing.")
        clc.dynamic_lifecycle.set_start_with_performing()

    ac = build_assembly_configuration(
        clc=clc,
        scc=scenario_controller_cfg,
        pc=provider_cfg,
        context_num=context_steps)

    LOG.info(f"Experiment {results_path}/{unique_collection_name}")
    LOG.info(f"SEARCH ARGS:{ac.controller_configuration.arguments}")
    LOG.info(f"DYNAMIC LIFECYCLE ARGS:{ac.life_cycle_configuration.arguments}")
    LOG.info(f"TARGET PROVIDER ARGS:({ac.provider_configuration.created_type}){ac.provider_configuration.arguments}")

    """ Complete learning pipeline """
    prepare_and_run_experiment_executor(
        ac=ac,
        env=environment,
        collection_name=unique_collection_name,
        collection_results_path=results_path,
        overwrite=overwrite,
        transferred_ensemble_path=transferred_ensemble_path,
        transferred_controller_path=transferred_controller_path
    )

from coupling_evol.data_process.postprocess.lifecycle_data import LifeCycleRawData
from coupling_evol.data_process.postprocess.permanent_post_processing import data_portfolios
from vrep_experiments_confs import *  # Windows + multiprocessing fix: https://stackoverflow.com/questions/63206414/multiprocessing-process-calls-main-function-from-process-creation-line-to-end-of
import coupling_evol.assembler.experiment_assembly as EE_H
from coupling_evol.data_process.inprocess import callbacks
import coupling_evol.engine.environment as ENV
from typing import Tuple
import logging
import configuration_build as C
from configuration_build import GoalType, EnvironmentType, build_assembly_configuration
import os
import numpy as np
import shutil
import coupling_evol.data_process.inprocess.recorder as rlog



###
LOG = logging.getLogger(__name__)

DUMMY = False
STEP_SLEEP = 0.01


def _compute_max_iters_from_context_steps(clc: C.ComplexLifecycleConfiguration, ctx_num):
    return (clc.rp.babble_iters + clc.rp.delta_search_iters + 2) * ctx_num + 1


def prepare_and_run_experiment_executor(clc: EE_H.AssemblyConfiguration,
                                        env: ENV.Environment, collection_name, collection_results_path,
                                        overwrite=False,
                                        transferred_ensemble_path: str = None,
                                        transferred_controller_path: str = None
                                        ):
    assert transferred_ensemble_path is not None or transferred_controller_path is None, "Transfering the controller without the ensemble is not supported."

    ee = EE_H.assemble_experiment_executor(
        results_path=collection_results_path, experiment_name=collection_name,
        assembly_configuration=clc, environment=env, overwrite_files=overwrite,
        clean_files=not overwrite, transferred_ensemble_path=transferred_ensemble_path,
        transferred_controller_path=transferred_controller_path
    )
    rlog.set_callback(callbacks.competing_fep_callback, callback_period=500)
    # ee.set_callback_printer(callbacks.competing_fep_callback, periodicity=500)

    ee.run(max_iters=clc.experiment_setup_parameters.max_iters,
           d_t=clc.essential_parameters.integration_step_size,
           step_sleep=clc.experiment_setup_parameters.step_sleep)





def standard_run(clc: C.ComplexLifecycleConfiguration,
                 unique_collection_name, environment_cfg: Tuple[EnvironmentType, dict],
                 target_cfg: Tuple[GoalType, dict, C.ScenarioBuild],
                 results_path, context_steps, overwrite=False,
                 transferred_ensemble_path: str = None,
                 transferred_controller_path: str = None
                 ):
    clc = copy.deepcopy(clc)

    if transferred_ensemble_path is not None and not os.path.exists(transferred_ensemble_path):
        LOG.error(f"Transfering ensemble from {transferred_ensemble_path} but it does not exist.")
        raise ValueError(f"Transfering ensemble from {transferred_ensemble_path} but it does not exist.")

    if transferred_controller_path is not None and not os.path.exists(transferred_controller_path):
        LOG.error(f"Transfering controller from {transferred_controller_path} but it does not exist.")
        raise ValueError(f"Transfering controller from {transferred_controller_path} but it does not exist.")

    """ Configuration modifications """
    clc.ep.natural_frequency = 6.
    clc.set_target_error_processing(typ=2)


    """ Environment preparation """
    ##
    if environment_cfg[0] is EnvironmentType.DUMMY:
        environment = ENV.DummyEnvironment(u_dim=MOTOR_DIMENSION, y_dim=SENSORY_DIMENSION)
        clc = clc.set_step_sleep(0.)

    elif environment_cfg[0] is EnvironmentType.SIMULATION:
        from coupling_evol.environments.coppeliasim import coppeliasim_environment as SIM_ENV
        clc = clc.set_step_sleep(STEP_SLEEP)
        environment = SIM_ENV.CoppeliaSimEnvironment(
            ampl_min=np.array([-0.32, 0.2, -0.2]), ampl_max=np.array([0.32, 0.7, 0.3]))
    else:
        raise NotImplemented(f"Environment {environment_cfg[0]} option is not implemented.")
    ##

    if transferred_ensemble_path is not None:
        LOG.info("Trial has transferred ensemble, thus it starts with performing.")
        clc.dynamic_lifecycle.set_start_with_performing()

    ac = build_assembly_configuration(
        clc=clc,
        scc=target_cfg[2].build(),
        pc=create_provider_configuration(goal_type=target_cfg[0], target_args=target_cfg[1]),
        context_num=context_steps)

    LOG.info(f"Experiment {results_path}/{unique_collection_name}")
    LOG.info(f"SEARCH ARGS:{ac.controller_configuration.arguments}")
    LOG.info(f"DYNAMIC LIFECYCLE ARGS:{ac.life_cycle_configuration.arguments}")
    LOG.info(f"TARGET PROVIDER ARGS:({ac.provider_configuration.created_type}){ac.provider_configuration.arguments}")

    """ Complete learning pipeline """
    prepare_and_run_experiment_executor(
        clc=ac,
        env=environment,
        collection_name=unique_collection_name,
        collection_results_path=results_path,
        overwrite=overwrite,
        transferred_ensemble_path=transferred_ensemble_path,
        transferred_controller_path=transferred_controller_path
    )


def extract_data_portfolio(results_path, coll_name, data_output_path, context_steps):
    _output_data_path = os.path.join(data_output_path, coll_name)
    if not os.path.exists(_output_data_path):
        os.mkdir(_output_data_path)
    dlcdp = LifeCycleRawData(results_path, coll_name, context_steps + 1)
    data_portfolios.LongevEvaluation(_output_data_path).process_and_save(dlcdp)


def repeated_main():
    overwrite = True
    # overwrite = False
    extract_n_delete = False
    banner_name = "040724_vrep"
    # exp_name = "hyp3"
    # exp_name = "pidcmp"
    exp_name = "test"

    context_steps = 5
    trials = [10]

    ##
    # transfer_run = "resources/transfer/0906_test"
    transfer_run = "resources/transfer/_walk_only"
    # transfer_run = "resources/transfer/walk_n_paralysis"
    # transfer_run = "resources/transfer/181023_walking"
    # transfer_run = "resources/transfer/080224_vrep" #bootstrapped three models
    # transfer_run = "resources/transfer/130224_vrep" #bootstrapped three models
    # transfer_run = None
    transfer_controller = True
    # transfer_controller = False

    ##
    data_output_path = os.path.join(DATA_OUTPUT_DICT, banner_name)
    if not os.path.exists(data_output_path) and extract_n_delete:
        os.mkdir(data_output_path)

    if transfer_run is not None and transfer_controller:
        transferred_ensemble_path, transferred_controller_path = (
            os.path.join(transfer_run, d) for d in ["ensemble", "controller"])
    elif transfer_run is not None and not transfer_controller:
        transferred_ensemble_path = os.path.join(transfer_run, "ensemble")
        transferred_controller_path = None
    else:
        transferred_ensemble_path = None
        transferred_controller_path = None

    results_path = os.path.join(EXP_ROOT_PATH, banner_name)
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    for trial in trials:
        for trg_cnf_name in TARGET_CONFIGS:
            for lc_cnf_name in LIFECYCLE_CONFIGS:
                for env_cnf_name in ENVIRONMENT_CONFIGS:
                    coll_name = get_collection_name(exp_name, lc_cnf_name, env_cnf_name, trg_cnf_name)
                    coll_name += f"_r{trial}"
                    standard_run(
                        clc=LIFECYCLE_CONFIGS[lc_cnf_name],
                        environment_cfg=ENVIRONMENT_CONFIGS[env_cnf_name],
                        target_cfg=TARGET_CONFIGS[trg_cnf_name],
                        unique_collection_name=coll_name,
                        results_path=results_path,
                        context_steps=context_steps,
                        overwrite=overwrite,
                        transferred_ensemble_path=transferred_ensemble_path,
                        transferred_controller_path=transferred_controller_path
                    )
                    if extract_n_delete:
                        extract_data_portfolio(results_path, coll_name, data_output_path, context_steps)
                        shutil.rmtree(os.path.join(results_path, coll_name))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.DEBUG)
    repeated_main()

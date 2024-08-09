from coupling_evol.data_process.postprocess.lifecycle_data import LifeCycleRawData
from coupling_evol.data_process.postprocess.permanent_post_processing import data_portfolios
from vrep_experiments_confs import *  # Windows + multiprocessing fix: https://stackoverflow.com/questions/63206414/multiprocessing-process-calls-main-function-from-process-creation-line-to-end-of
import logging
import os
import shutil
from experiment_helpers.configuration_run import standard_run
from experiment_helpers.configuration_build import get_collection_name


###
LOG = logging.getLogger(__name__)


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
                        environment=get_environment(*ENVIRONMENT_CONFIGS[env_cnf_name]),
                        provider_cfg=create_provider_configuration(
                            goal_type=TARGET_CONFIGS[trg_cnf_name][0], target_args=TARGET_CONFIGS[trg_cnf_name][1]),
                        scenario_controller_cfg=TARGET_CONFIGS[trg_cnf_name][2](),
                        unique_collection_name=coll_name,
                        results_path=results_path,
                        context_steps=context_steps,
                        overwrite=overwrite,
                        transferred_ensemble_path=transferred_ensemble_path,
                        transferred_controller_path=transferred_controller_path,
                    )
                    if extract_n_delete:
                        extract_data_portfolio(results_path, coll_name, data_output_path, context_steps)
                        shutil.rmtree(os.path.join(results_path, coll_name))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.INFO)
    repeated_main()

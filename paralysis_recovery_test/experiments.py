from coupling_evol.data_process.postprocess.lifecycle_data import LifeCycleRawData
from coupling_evol.data_process.postprocess.permanent_post_processing import data_portfolios
import shutil
from paralysis_recovery_test.configuration import *
from experiment_helpers.configuration_build import *
from experiment_helpers.configuration_run import *
import re
from visuals.dataportfolio_view.decim_journal_evaluation.portfolio import generate_detail as generate_portfolio_decim
import enum
###
LOG = logging.getLogger(__name__)
FWD_REG = re.compile('(\d+)_fwm')


class Variant(enum.Enum):
    ORIGINAL = 0
    SINGLE_MODEL = 1
    REACTIVE = 2


def _walk_transfer_dir_path(banner_name):
    return os.path.join(TRANSFER_PATH, banner_name, WALK_TRANSFER_COLLECTION_NAME)


def extract_data_portfolio(banner_name, collection_name):
    extract_path = os.path.join(EXTRACT_PATH, banner_name)
    results_path = os.path.join(RECORD_PATH, banner_name)
    collection_extract_path = os.path.join(extract_path, collection_name)
    if not os.path.exists(collection_extract_path):
        os.makedirs(collection_extract_path)
    lcrd = LifeCycleRawData(results_path, collection_name)
    data_portfolios.LongevEvaluation(collection_extract_path).process_and_save(lcrd)


def render_visual_portfolio(banner_name, collection_name):
    extract_dir = os.path.join(EXTRACT_PATH, banner_name)
    collection_extract_path = os.path.join(extract_dir, collection_name)
    visuals_dir = os.path.join(VISUALS_PATH, banner_name)
    visuals_path = os.path.join(visuals_dir, collection_name)
    if not os.path.exists(visuals_path):
        os.makedirs(visuals_path)
    dp = data_portfolios.LongevEvaluation(collection_extract_path)
    generate_portfolio_decim(output_path=visuals_path, dp=dp, name="")


def _get_lowest_confidence(banner_name, collection_name, cut_from=400):
    extract_dir = os.path.join(EXTRACT_PATH, banner_name)
    collection_extract_path = os.path.join(extract_dir, collection_name)
    dp = data_portfolios.LongevEvaluation(collection_extract_path)
    return np.min(dp.segmented_model_selection.data.second_score[cut_from:])


def threshold_from_undisturbed_walk_experiment(banner_name):
    lowest = _get_lowest_confidence(banner_name, UNDISTURBED_COLLECTION_NAME, cut_from=400)
    threshold = lowest - lowest * 0.1
    LOG.info(f"From banner {banner_name} the lowest world-model:zero-model log-odds is {lowest}."
             f" The threshold value {threshold} will be log-odds lower bound.")
    return threshold


def prepare_transfer(banner_name, collection_name):
    transfer_path = os.path.join(TRANSFER_PATH, banner_name, WALK_TRANSFER_COLLECTION_NAME)
    result_path = os.path.join(RECORD_PATH, banner_name, collection_name)
    controller_path = os.path.join(result_path, "controller")
    ensemble_path = os.path.join(result_path, "ensemble")
    ##
    if not os.path.exists(transfer_path):
        os.makedirs(transfer_path)
    ##
    controller_transfer_path = os.path.join(transfer_path, "controller")
    ensemble_transfer_path = os.path.join(transfer_path, "ensemble")
    if not os.path.exists(controller_transfer_path):
        os.makedirs(controller_transfer_path)
    if not os.path.exists(ensemble_transfer_path):
        os.makedirs(ensemble_transfer_path)
    ##
    for f in os.listdir(controller_path):
        shutil.copy(os.path.join(controller_path, f), os.path.join(controller_transfer_path, f))

    furthest_model = 1
    for f in os.listdir(ensemble_path):
        model_id = int(FWD_REG.findall(f)[0])
        if model_id > furthest_model:
            furthest_model = model_id
    model_name = f"{furthest_model}_fwm"
    new_model_name = "1_fwm"
    shutil.copytree(os.path.join(ensemble_path, model_name), os.path.join(ensemble_transfer_path, new_model_name))


def transfer_ensemble_controller_paths(transfer_banner_name):
    if transfer_banner_name is None:
        return None, None
    else:
        transfer_path = _walk_transfer_dir_path(transfer_banner_name)
        return str(os.path.join(transfer_path, "ensemble")), str(os.path.join(transfer_path, "controller"))


def change_variant(clc: ComplexLifecycleConfiguration, variant: Variant, new_threshold=None) -> ComplexLifecycleConfiguration:
    config = copy.deepcopy(clc)
    if new_threshold is not None:
        config.dynamic_lifecycle = config.dynamic_lifecycle.min_confident_elements_rate(new_threshold)
        LOG.info(f"The threshold value is set to {config.dynamic_lifecycle.args['min_confident_elements_rate']}")
    if variant is Variant.ORIGINAL:
        pass
    elif variant is Variant.REACTIVE:
        config.dynamic_lifecycle = switch_dynamic_life_cycle_variant(config.dynamic_lifecycle,
                                                                     DLCSelectionVariant.REACTIVE,
                                                                     DLCLearningDecisionVariant.PURE_REACTIVE,
                                                                     DLCCompositionVariant.LEAVE_ORIGINAL)
    elif variant is Variant.SINGLE_MODEL:
        config.dynamic_lifecycle = switch_dynamic_life_cycle_variant(config.dynamic_lifecycle,
                                                                     DLCSelectionVariant.ONE_MODEL,
                                                                     DLCLearningDecisionVariant.PURE_SCHEDULED,
                                                                     DLCCompositionVariant.LEAVE_ORIGINAL)
    return config


def parametrized_run(banner_name, collection_name,
                     target_config_name: str,
                     variant: Variant, transfer_banner_name=None, context_steps=5, overwrite=False,
                     undisturbed_measurement_banner_name=None
                     ):
    transferred_ensemble_path, transferred_controller_path = transfer_ensemble_controller_paths(transfer_banner_name)

    results_path = os.path.join(RECORD_PATH, banner_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    threshold = 0
    if undisturbed_measurement_banner_name is not None:
        threshold = threshold_from_undisturbed_walk_experiment(undisturbed_measurement_banner_name)

    lc_cnf = change_variant(DYNAMIC_LIFECYCLE_CONFIGS["model_competition"], variant=variant, new_threshold=threshold)
    env_cnf = ENVIRONMENT_CONFIGS["Vrep"]
    trg_cnf = NAVIGATING_TARGET_FARGOAL[target_config_name]

    standard_run(
        clc=lc_cnf,
        environment=get_environment(*env_cnf),
        provider_cfg=create_provider_configuration(
            goal_type=trg_cnf[0], target_args=trg_cnf[1]),
        scenario_controller_cfg=trg_cnf[2].build(),
        unique_collection_name=collection_name,
        results_path=results_path,
        context_steps=context_steps,
        overwrite=overwrite,
        transferred_ensemble_path=transferred_ensemble_path,
        transferred_controller_path=transferred_controller_path,
    )


def train_walking_model_experiment(banner_name, context_steps=5, overwrite=False):
    collection_name = WALK_LEARNING_COLLECTION_NAME
    parametrized_run(banner_name, collection_name=collection_name, target_config_name="just_walk",
                     variant=Variant.ORIGINAL, transfer_banner_name=None, context_steps=context_steps,
                     overwrite=overwrite, undisturbed_measurement_banner_name=None)
    prepare_transfer(banner_name=banner_name, collection_name=collection_name)
    extract_data_portfolio(banner_name, collection_name)
    # render_visual_portfolio(banner_name, collection_name)


def undisturbed_walking_experiment(banner_name, transfer_banner_name, context_steps=3, overwrite=False):
    collection_name = UNDISTURBED_COLLECTION_NAME
    parametrized_run(banner_name, collection_name=collection_name, target_config_name="just_walk",
                     variant=Variant.SINGLE_MODEL, transfer_banner_name=transfer_banner_name, context_steps=context_steps,
                     overwrite=overwrite, undisturbed_measurement_banner_name=None)
    extract_data_portfolio(banner_name, collection_name)
    # render_visual_portfolio(banner_name, collection_name)


def paralysis_and_recovery_experiment(banner_name, collection_name, transfer_banner_name,
                                      variant: Variant,
                                      undisturbed_measurement_banner_name=None, context_steps=5, overwrite=False):
    parametrized_run(banner_name, collection_name=collection_name, target_config_name="paralysis_recovery",
                     variant=variant, transfer_banner_name=transfer_banner_name, context_steps=context_steps,
                     overwrite=overwrite, undisturbed_measurement_banner_name=undisturbed_measurement_banner_name)
    extract_data_portfolio(banner_name, collection_name)
    render_visual_portfolio(banner_name, collection_name)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.INFO)
    # train_walking_model_experiment("c")
    # single_model_walk("a", "a")
    paralysis_and_recovery_experiment("c", "usual2",
                                      transfer_banner_name="c", undisturbed_measurement_banner_name="c",
                                      variant=Variant.REACTIVE)
    # undisturbed_walking_experiment("c", transfer_banner_name="c")
    # print(_get_lowest_confidence("c", UNDISTURBED_COLLECTION_NAME))

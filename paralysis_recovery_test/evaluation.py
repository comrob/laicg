import os

import numpy as np

from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation
from paralysis_recovery_test.configuration import *
from coupling_evol.data_process.postprocess.permanent_post_processing import data_portfolios
from visuals.dataportfolio_view.decim_journal_evaluation.portfolio import generate_detail as generate_portfolio_decim
import json
import logging
LOG = logging.getLogger(__name__)

def learning_experiment_test(banner_name):
    pth = os.path.join(EXTRACT_PATH, banner_name, WALK_LEARNING_COLLECTION_NAME)
    dp = data_portfolios.LongevEvaluation(pth)
    model_n = len(dp.models.data)
    return [(f"Expected 3 models learned during walk bootstrapping. {model_n} were trained.", bool(model_n == 3))]


def undisturbed_experiment_test(banner_name):
    pth = os.path.join(EXTRACT_PATH, banner_name, UNDISTURBED_COLLECTION_NAME)
    dp = data_portfolios.LongevEvaluation(pth)
    goal = dp.navigation.data.goal[-1, :]
    phase_n = dp.uyt_mem.data.segments.shape[1]
    itr = (dp.segmented_model_selection.data.iter).astype(int)
    location = dp.navigation.data.location[itr]
    dists = np.sqrt(np.sum(np.square(goal - location), axis=1))
    easy_approach = (f"Robot approaches the goal {goal}. The reached location is {location[-1]}."
                     f" Thus start distance {dists[1]:1.2f}m > final distance {dists[-1]:1.2f}m", bool(dists[1] > dists[-1]))

    ddists = dists[:-1] - dists[1:]  # dist is distance from the goal, we want speed towards the goal
    arr = np.zeros((ddists.shape[0] - phase_n,))
    for i in range(dp.uyt_mem.data.segments.shape[1]):
        arr += ddists[i:-phase_n + i]
    minspeed = np.min(arr[3 * len(arr) // 4:])
    gait_speed = (f"During last 1/4 of the experiment, the robot approaches the goal each gait period."
                  f" Minimal measured gait speed is {minspeed:1.2f} > 0.", bool(minspeed > 0))

    zero_model_score = dp.segmented_model_selection.data.second_score
    d_zms = zero_model_score[1:] - zero_model_score[:-1]
    avg_d_zms = np.average(d_zms[len(d_zms) // 2:])
    score_not_falling = (f"During last 1/2 of the experiment, the world-model:zero-model log-odds do not rapidly decrease."
                         f" Average log-odds change is {avg_d_zms:1.2f} > -0.01", bool(avg_d_zms > -0.01))

    return [easy_approach, gait_speed, score_not_falling]


def paralysis_recovery_experiment_test(banner_name, collection):
    pth = os.path.join(EXTRACT_PATH, banner_name, collection)
    dp = data_portfolios.LongevEvaluation(pth)
    selected_models = dp.segmented_model_selection.data.selected_model
    itr = (dp.segmented_model_selection.data.iter).astype(int)
    seconds = itr * STEP_SLEEP
    stages = dp.navigation.data.stages[itr]
    bbl_intervals = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)
    first_bbl = bbl_intervals[0]
    # timsetamp = dp.time_axis.data.timestamp - dp.time_axis.data.timestamp[0]
    is_paralysis = (dp.scenario_switch.data.phase[itr]).astype(int) == 1
    switches_time = np.arange(len(is_paralysis) - 1)[is_paralysis[1:] ^ is_paralysis[:-1]] + 1
    paralysis_start, paralysis_end = switches_time[0], switches_time[1]
    ##
    is_pim = selected_models == 1.
    paralysis_model_switches = np.arange(len(selected_models)-1)[is_pim[1:] ^ is_pim[:-1]] + 1

    ##
    paralysis_model_usage_rate = np.sum(selected_models[first_bbl[1]:paralysis_end] == 1.) / (
                paralysis_end - first_bbl[1])
    recovery_model_usage_rate = np.sum(selected_models[paralysis_end:] == 1.) / (len(selected_models) - paralysis_end)
    ##
    paralysis_recognition = (
        f"After paralysis starts at {seconds[paralysis_start]:1.0f}s, the robot recognizes new dynamics"
        f" and trained new model at {seconds[first_bbl[0]]:1.0f}s,"
        f" before the recovery at {seconds[paralysis_end]:1.0f}s.",
        bool(paralysis_start < first_bbl[0] < paralysis_end))
    paralysis_selection = (
        f"After learning, the paralysis model was utilized more than half of the time:"
        f" {paralysis_model_usage_rate:0.2f}>0.5.",
        bool(paralysis_model_usage_rate > 0.5))
    recovery_recognition = (
        f"After the recovery, the paralysis model was utilized less than half of the time:"
        f" {recovery_model_usage_rate:0.2f}<0.5.",
        bool(recovery_model_usage_rate < 0.5))
    return [paralysis_recognition, paralysis_selection, recovery_recognition]


def test_banner(banner_name):
    banner_pth = os.path.join(EXTRACT_PATH, banner_name)
    ret = {}
    for f in os.listdir(banner_pth):
        if f == WALK_LEARNING_COLLECTION_NAME:
            ret[WALK_LEARNING_COLLECTION_NAME] = learning_experiment_test(banner_name)
        elif f == UNDISTURBED_COLLECTION_NAME:
            ret[UNDISTURBED_COLLECTION_NAME] = undisturbed_experiment_test(banner_name)
        else:
            ret[f] = paralysis_recovery_experiment_test(banner_name, f)
    return ret


def render_visual_portfolio(banner_name, collection_name):
    extract_dir = os.path.join(EXTRACT_PATH, banner_name)
    collection_extract_path = os.path.join(extract_dir, collection_name)
    visuals_dir = os.path.join(REPORT_PATH, banner_name)
    visuals_path = os.path.join(visuals_dir, collection_name)
    if not os.path.exists(visuals_path):
        os.makedirs(visuals_path)
    dp = data_portfolios.LongevEvaluation(collection_extract_path)
    generate_portfolio_decim(output_path=visuals_path, dp=dp, name="")


def test_success_rate(eval_result):
    cou = 0
    suc = 0
    for exp in eval_result:
        for tst in eval_result[exp]:
            cou += 1
            suc += 1 if tst[1] else 0
    return cou, suc


def create_report(banner_name):
    banner_pth = os.path.join(EXTRACT_PATH, banner_name)
    report_pth = os.path.join(REPORT_PATH, banner_name)
    if not os.path.exists(report_pth):
        os.makedirs(report_pth)

    ##
    eval_result = test_banner(banner_name)
    with open(os.path.join(report_pth, 'report.json'), 'w') as file:
        file.writelines(json.dumps(eval_result, sort_keys=True, indent=2))
    tst_n, suc_n = test_success_rate(eval_result)
    ##

    for f in os.listdir(banner_pth):
        if f == WALK_LEARNING_COLLECTION_NAME:
            pass
        elif f == UNDISTURBED_COLLECTION_NAME:
            pass
        else:
            render_visual_portfolio(banner_name, f)

    print(f"{suc_n}/{tst_n} of tests are successful. The report is in directory {os.path.abspath(report_pth)} .")

    return tst_n == suc_n


if __name__ == '__main__':
    create_report("test")
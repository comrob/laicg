from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation
from coupling_evol.data_process.postprocess.permanent_post_processing.data_portfolios import DecimJournalEvaluation
from coupling_evol.data_process.postprocess.permanent_post_processing.helpers import parameter_parser, \
    PortfolioAggregate, Tag
from coupling_evol.engine.embedded_control import EmbeddedTargetParameter
from visuals.image_merger import vertical_merge_images
import visuals.dataportfolio_view.decim_journal_evaluation.pretty_visuals as PV
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List
from visuals import common as C
from PIL import Image


NEG_POS_CMAP = "coolwarm"
# NEG_POS_CMAP = "Spectral"
POS_CMAP = "YlOrBr"

class Style:
    def __init__(self, label, color, marker='.'):
        self.label: str = label
        self.color: str = color
        self.marker: str = marker


PERFORMING_COLOR = 'k'
BABBLING_COLOR = 'g'
PARALYSIS_COLOR = 'y'
LOGODDS_COLOR = 'k'

LIFECYCLE_COLORS = ["g", "c", "m", "r", "y","violet","orange"]

BABBLING_STAGE_STYLE = Style('babble', color=BABBLING_COLOR, marker='s')

MODEL_COLORS = ['darkorange', 'fuchsia', 'g', 'r', 'y', 'darkviolet', 'orangered', 'peru', 'springgreen']
MODEL_MARKERS = ["^", "v", "o", "s", "x"]
MODEL_STYLES = [
    Style(str(i), color=MODEL_COLORS[i%len(MODEL_COLORS)], marker=MODEL_MARKERS[i%len(MODEL_MARKERS)])
    for i in range(8)
]

# MODEL_STYLES = [
#     Style("0", color='darkorange', marker="^"),
#     Style("1", color='fuchsia', marker='v'),
#     Style("2", color='g', marker='o'),
#     Style("3", color='r', marker='s'),
#     Style("4", color='y', marker='x'),
#     Style("5", color='y', marker='x'),
#     Style("6", color='y', marker='x'),
#     Style("7", color='y', marker='x'),
#     Style("8", color='y', marker='x'),
# ]

D_T = 0.01
TIME_LABEL = "Time (s)"

SENSORY_LABELS = ["head", "roll", "pitch", "yaw", "side"]
LEG_LABELS = ["L1 coxa", "R1 coxa", "L2 coxa", "R2 coxa", "L3 coxa", "R3 coxa"]

LOCATION_LABELS = ("X location (m)", "Y location (m)")

PRIMITIVE_LABELS = ["left", "forward", "right"]


def argmedian(data):
    return np.argsort(data)[len(data) // 2]


def _sqr(diff):
    return np.square(diff)


def _abs(diff):
    return np.absolute(diff)


def get_int(nrm=_sqr):
    def _int(err):
        return np.cumsum(np.mean(nrm(err), axis=1))

    return _int


def get_smh(nrm=_sqr):
    def _smh(err):
        convolution_window = 100
        v = np.ones((convolution_window * 2,)) / convolution_window
        v[convolution_window:] = 0
        return np.convolve(np.mean(nrm(err), axis=1), v=v)[convolution_window * 2:-convolution_window * 2]

    return _smh


# PERFORMANCE_METRICS = [(get_smh(_sqr), "Performance MSE"), (get_int(_sqr), "Cumulative performance MSE")]
PERFORMANCE_METRICS = [(get_smh(_abs), "Performance MAE"), (get_int(_abs), "Cumulative performance MAE")]
GOAL_METRICS = [(lambda diff: np.linalg.norm(diff, axis=1), "Goal distance")]


def select_trial(ps: List[DecimJournalEvaluation]):
    metric, _ = PERFORMANCE_METRICS[1]
    diffs = []
    is_prfm = []
    min_length = min([len(p.uyt_mem.data.target) for p in ps])

    for trial in ps:
        # t = Tag(trial=trial, scenario=sc_id, lifecycle=lifecycle, environment=env_id)
        d = trial.uyt_mem.data

        # sns = _filter_nonperforming(, d.performing_stage, gait_n=4)
        diff_mem = EmbeddedTargetParameter.difference_mem(
            target_mem=d.target,
            metric_mem=d.metrics,
            weight_mem=d.weights,
            sensory_mem=d.observation
        )
        diffs.append(diff_mem[:min_length - 1, :])
        is_prfm.append(d.performing_stage[:min_length - 1])

    errs = np.asarray([metric(trial) for k, trial in enumerate(diffs)])[:, 0]
    return argmedian(errs)


###########

###########


def _get_lc_style_dictionary(lifecycles: List[str]):
    ret = {}
    for i, lc in enumerate(lifecycles):
        ret[lc] = Style(label=lc, color=LIFECYCLE_COLORS[i])
    return ret


def _get_ordered_lc_list(lifecycles: List[str]):
    parsed = [(lcn, [v for v in parameter_parser(lcn).values()]) for lcn in lifecycles]
    return [s[0] for s in sorted(parsed, key=lambda x: x[1])]


def get_performance_diffs(uyt_mem_data, min_length):
    diffs = []
    is_prfm = []
    for d in uyt_mem_data:
        diff_mem = EmbeddedTargetParameter.difference_mem(
            target_mem=d.target,
            metric_mem=d.metrics,
            weight_mem=d.weights,
            sensory_mem=d.observation
        )
        diffs.append(diff_mem[:min_length - 1, :])
        is_prfm.append(d.performing_stage[:min_length - 1])
    return diffs


def error_figs(fig: C.FigProvider, pa: PortfolioAggregate[DecimJournalEvaluation],
               resutls_path, name, convolution_window=100
               ):
    lifecycles = _get_ordered_lc_list(pa.lifecycle_range())
    trials = pa.trial_range()
    environments = pa.environment_range()
    scenarios = pa.scenario_range()
    env_id = environments[0]
    sc_id = scenarios[0]
    min_length = min([len(p.uyt_mem.data.target) for p, _ in pa.agg])
    lc_diffs = []
    lc_is_performing = []
    lc_labels = []

    styles = _get_lc_style_dictionary(lifecycles)
    first_tag = Tag(trial=trials[0], scenario=scenarios[0], lifecycle=lifecycles[0],
                    environment=environments[0])
    scenario_phases = pa.get(first_tag).scenario_switch.data.phase
    iter_len = len(pa.get(first_tag).uyt_mem.data.performing_stage)
    iter_seg_ratio = len(scenario_phases) // iter_len
    scenario_phases_seg = scenario_phases[::iter_seg_ratio]

    for i, lifecycle in enumerate(lifecycles):
        diffs = []
        is_prfm = []
        for trial in trials:
            t = Tag(trial=trial, scenario=sc_id, lifecycle=lifecycle, environment=env_id)
            d = pa.get(t).uyt_mem.data

            # sns = _filter_nonperforming(, d.performing_stage, gait_n=4)
            diff_mem = EmbeddedTargetParameter.difference_mem(
                target_mem=d.target,
                metric_mem=d.metrics,
                weight_mem=d.weights,
                sensory_mem=d.observation
            )
            diffs.append(diff_mem[:min_length - 1, :])
            is_prfm.append(d.performing_stage[:min_length - 1])
        lc_diffs.append(diffs)
        lc_labels.append(lifecycle)
        lc_is_performing.append(is_prfm)

    plt.rcParams["figure.figsize"] = (10, 10)
    _fig = fig()
    figs = C.subplots(_fig, 3, 2)

    PV.comparison(
        [(figs[0][0], figs[0][1]), (figs[1][0], figs[1][1])], diffs=lc_diffs, scenario_phases=scenario_phases_seg,
        method_labels=[styles[lc].label for lc in lc_labels], colors=[styles[lc].color for lc in lc_labels],
        metrics=PERFORMANCE_METRICS,
        mean_window=convolution_window,
        ts=np.arange(len(lc_diffs[0][0])) * iter_seg_ratio * D_T, time_label=TIME_LABEL, legend_on=True,
        has_first=True, has_last=False
    )
    """
    EVOL
    """
    plt.rcParams["figure.figsize"] = (12, 3)

    def tag(_lc, _tr):
        return Tag(trial=_tr, scenario=sc_id, lifecycle=_lc, environment=env_id)

    scenario_phase = pa.get(tag(lifecycles[0], trials[0])).scenario_switch.data.phase
    diffs = []
    ts = np.arange(len(pa.get(tag(lifecycles[0], trials[0])).navigation.data.location)) * D_T
    for i, lifecycle in enumerate(lifecycles):
        nav_data = [pa.get(tag(lifecycle, trial)).navigation.data for trial in trials]
        diffs.append([nav.location - nav.goal for nav in nav_data])

    PV.comparison([(figs[2][0], figs[2][1])], diffs=diffs,
                  method_labels=[styles[lc].label for lc in lifecycles],
                  colors=[styles[lc].color for lc in lifecycles], scenario_phases=scenario_phase, metrics=GOAL_METRICS,
                  ts=ts, mean_window=100, time_label=TIME_LABEL, legend_on=False, has_first=False, has_last=True
                  )

    plt.savefig(os.path.join(resutls_path, name + "metric_comparison.png"), bbox_inches='tight')
    # C.standalone_legend(C.plt,
    #                     [{"linestyle": '-', "color": styles[ll].color, "label": styles[ll].label} for ll in lc_labels])
    # plt.savefig(os.path.join(resutls_path, name + "comparison_legend.png"), bbox_inches='tight')


def map_fig(fig: C.FigProvider, pa: PortfolioAggregate[DecimJournalEvaluation],
            resutls_path, name):
    lifecycles = pa.lifecycle_range()
    trials = pa.trial_range()
    environments = pa.environment_range()
    scenarios = pa.scenario_range()
    env_id = environments[0]
    sc_id = scenarios[0]
    styles = _get_lc_style_dictionary(lifecycles)

    def tag(_lc, _tr):
        return Tag(trial=_tr, scenario=sc_id, lifecycle=_lc, environment=env_id)

    """
    MAPS
    """
    plt.rcParams["figure.figsize"] = (4, 16)
    map_fig = fig()
    axs = C.subplots(map_fig, len(lifecycles), 1)
    for i, lifecycle in enumerate(lifecycles):
        ax = axs[i][0]
        trial_ps = [pa.get(tag(lifecycle, trial)) for trial in trials]
        nav_data = [trial_p.navigation.data for trial_p in trial_ps]
        best = select_trial(trial_ps)
        best_nav = nav_data[best]

        for j, nav in enumerate(nav_data):
            if j == best:
                continue
            ax.plot(nav.location[:, 0], nav.location[:, 1], color='k', alpha=0.3)

        PV.xy_heading_stages_map(ax, xy_translation=best_nav.location, xy_goal=best_nav.goal)
        PV.draw_staged_path(ax=ax,
                            xy_translation=best_nav.location,
                            yaw=best_nav.heading,
                            stages=best_nav.stages,
                            show_yaw=False,
                            vector_draw_granularity=2000,
                            babbling_style={"color": BABBLING_COLOR, "linestyle": '-', "alpha": 0.8},
                            performing_style={"color": PERFORMING_COLOR, "linestyle": '-', "alpha": 0.8},
                            )
        ax.set_ylim(-110, 10)
        ax.set_xlim(-110, 10)
        ax.plot(best_nav.location[-1, 0], best_nav.location[-1, 1], 'bo', label="reached")
        ax.set_title(styles[lifecycle].label)

    # axs[0][-1].legend(loc='upper left')
    for i in range(len(axs[0])):
        if i != 0:
            axs[0][i].set_yticklabels([])
        axs[0][i].set_xlabel(LOCATION_LABELS[0])
    axs[0][0].set_ylabel(LOCATION_LABELS[1])
    plt.savefig(os.path.join(resutls_path, name + "navigation.png"), bbox_inches='tight')


def get_first_model_activation_time(t, selected_model, scenario_phase):
    first_chosen = (selected_model == 0).astype(dtype=int)
    model_switch = first_chosen[1:] - first_chosen[:-1]
    first_model_on_times = t[1:][model_switch > 0]
    first_model_off_times = t[1:][model_switch < 0]
    paralysis_end = np.arange(len(scenario_phase))[scenario_phase == 1][-1]
    first_model_on_times = first_model_on_times[first_model_on_times > paralysis_end * D_T]
    first_model_on_time = None
    if len(first_model_on_times) > 0 and len(first_model_off_times) > 0:  # and first_model_on_times[-1] > \
        # first_model_off_times[-1] and first_model_on_times[-1] > 100:
        first_model_on_time = first_model_on_times[0]
    return first_model_on_time


def get_second_model_activation_after_learning(t, selected_model, bbls):
    second_model_times = t[selected_model == 1]
    bb1_en = 0
    if len(bbls) > 0:
        _, bb1_en = bbls[0]

    second_model_times = second_model_times[second_model_times > (bb1_en * D_T) + 100]
    sec_model_time = None
    if len(second_model_times) > 0:
        sec_model_time = second_model_times[2]
    return sec_model_time


def model_selection_comparison(fig: C.FigProvider, pa: PortfolioAggregate[DecimJournalEvaluation],
                               resutls_path, name):
    lifecycles = _get_ordered_lc_list(pa.lifecycle_range())
    trials = pa.trial_range()
    environments = pa.environment_range()
    scenarios = pa.scenario_range()
    env_id = environments[0]
    sc_id = scenarios[0]
    styles = _get_lc_style_dictionary(lifecycles)

    offset = 0.1
    shift = 0.3
    space = 1
    info_space = 0.4
    evol_markersize = 10
    map_markersize = 8

    # SPECIFIC PARAMS!
    decision_threshold = 0.035
    weighting = np.asarray([30, 2, 2, 30, 30] + [1] * 18)


    def tag(_lc, _tr):
        return Tag(trial=_tr, scenario=sc_id, lifecycle=_lc, environment=env_id)

    x_minmax = (0,0)
    y_minmax = (0,0)
    for i, lifecycle in enumerate(lifecycles):
        trial_data = [pa.get(tag(lifecycle, trial)) for trial in trials]
        for j, td in enumerate(trial_data):
            x_minmax = (
                np.minimum(np.min(td.navigation.data.location[:,0]), x_minmax[0]),
                np.maximum(np.max(td.navigation.data.location[:,0]), x_minmax[1])
            )
            y_minmax = (
                np.minimum(np.min(td.navigation.data.location[:,1]), y_minmax[0]),
                np.maximum(np.max(td.navigation.data.location[:,1]), y_minmax[1])
            )

    """
    Switch and Model Selection
    """
    img_paths = []
    for i, lifecycle in enumerate(lifecycles):

        plt.rcParams["figure.figsize"] = (11, 4)
        map_fig = fig()
        map_fig.suptitle(styles[lifecycle].label)
        axs = C.subplots(map_fig, 1, 2, gridspec_kw={'width_ratios': [1, 2]})
        ax_sc = axs[0][1]
        ax_map = axs[0][0]
        trial_data = [pa.get(tag(lifecycle, trial)) for trial in trials]
        best = select_trial(trial_data)

        ##
        model_num = max([len(td.models.data) for td in trial_data])
        model_bands = [(i * (space + shift), (i + 1) * (space + shift) - shift) for i in range(model_num)]
        # model_2_score_band = model_bands[0]
        # model_1_score_band = model_bands[1]
        second_score_band = (model_bands[-1][1] + shift, model_bands[-1][1] + shift + space)
        info_band = (second_score_band[1] + shift, second_score_band[1] + shift + info_space)
        ylim = (-offset, info_band[1] + offset)
        ##

        iters = trial_data[best].segmented_model_selection.data.iter.astype(int)
        it_p_sg = len(trial_data[best].navigation.data.stages) / len(iters)
        toseg = lambda _q: int(_q / it_p_sg)
        segt = iters * D_T
        t = segt

        scenario_phase = pa.get(tag(lifecycle, trials[0])).scenario_switch.data.phase

        """MAP"""
        for j, td in enumerate(trial_data):
            nav = td.navigation.data
            if j == best:
                continue
            ax_map.plot(nav.location[:, 0], nav.location[:, 1], color='k', alpha=0.1)

        best_nav = trial_data[best].navigation.data
        PV.xy_heading_stages_map(ax_map, xy_translation=best_nav.location, xy_goal=best_nav.goal)
        PV.draw_staged_path(ax=ax_map,
                            xy_translation=best_nav.location,
                            yaw=best_nav.heading,
                            stages=best_nav.stages,
                            show_yaw=False,
                            vector_draw_granularity=2000,
                            babbling_style={"color": BABBLING_COLOR, "linestyle": '-', "alpha": 0.8},
                            performing_style={"color": PERFORMING_COLOR, "linestyle": '-', "alpha": 0.8},
                            )
        ax_map.plot(best_nav.location[scenario_phase == 1, 0], best_nav.location[scenario_phase == 1, 1], '-',
                    color=PARALYSIS_COLOR, linewidth=10, label="paralysis", alpha=0.5)
        ax_map.set_ylim(1.1 * y_minmax[0], 1.1 * y_minmax[1])
        ax_map.set_xlim(1.1 * x_minmax[0], 1.1 * x_minmax[1])
        ax_map.plot(best_nav.location[-1, 0], best_nav.location[-1, 1], 'bo', label="reached")
        # ax_map.set_title(styles[lifecycle].label)

        ####
        processed_second_scores = []
        for td in trial_data:
            bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(td.navigation.data.stages)
            sec_score = td.segmented_model_selection.data.second_score
            sec_score[0:408] = np.min(sec_score)
            for st, en in bbls:
                sec_score[toseg(st):toseg(en) + 408] = np.min(sec_score)
            processed_second_scores.append(sec_score)

        pss_min = np.min(list(map(np.min, processed_second_scores)))
        pss_max = np.max(list(map(np.max, processed_second_scores)))
        nrm_sec_sc = [1 - (scor - pss_min) / (pss_max - pss_min) for scor in processed_second_scores]
        _decision_threshold = 1 - (decision_threshold - pss_min) / (pss_max - pss_min)
        """SCORES"""
        competition = []
        for k, td in enumerate(trial_data):
            stages = td.navigation.data.stages
            _iters = td.segmented_model_selection.data.iter.astype(int)
            t_trial = _iters * D_T
            """Competition score postprocess"""
            """Number of models does not equal number of babbles!!!"""
            bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)
            fs = td.segmented_model_selection.data.first_score
            min_sc = np.min(fs)
            _first_scores = np.zeros((fs.shape[0], np.maximum(fs.shape[1], model_num))) + min_sc
            _first_scores[:, 0] = fs[:, 0]
            if len(bbls) > 0 and fs.shape[1] == model_num:
                # assign signal to the ones that are already after babble
                # if there are some pre-learned models, they must have the signal from the start
                pre_trained_id = model_num - len(bbls)
                for j in range(pre_trained_id):
                    _first_scores[:, j] = fs[:, j]

                for j in range(pre_trained_id, np.maximum(fs.shape[1], pre_trained_id)):
                    _st, _en = bbls[j-pre_trained_id]
                    en = toseg(_en)
                    _first_scores[en:, j] = fs[en:, j]

            exp_sc = np.exp(_first_scores * 100)
            first_scores = exp_sc / (np.sum(exp_sc, axis=1)[:, None])

            for _st, _en in bbls:
                first_scores[toseg(_st):toseg(_en + 100), :] = 0.5  # _first_scores[toseg(_st):toseg(_en+100), 0:1]

            competition.append(first_scores)

            """Decision score postprocess"""


            nrm_w = weighting / np.sum(weighting)
            sec_score = td.segmented_model_selection.data.second_score
            for st, en in bbls:
                sec_score[toseg(st):toseg(en)] = 0

            """Plotting"""
            ##
            for mn in range(model_num):
                ax_sc.plot(t_trial, first_scores[:, model_num-1-mn] + model_bands[mn][0], color=MODEL_STYLES[model_num-1-mn].color, alpha=0.3)
            ##
            ax_sc.plot(t_trial, nrm_sec_sc[k] + second_score_band[0], color='k', alpha=0.3)

        ##
        for mn in range(model_num):
            ax_sc.plot(t, competition[best][:, model_num-1-mn] + model_bands[mn][0], color=MODEL_STYLES[model_num-1-mn].color, alpha=1.,
                       linewidth=3.)
        # ax_sc.plot(t, competition[best][:, 0] + model_bands[1][0], color=MODEL_STYLES[0].color, alpha=1.,
        #            linewidth=3.)
        ##

        ax_sc.plot(t, nrm_sec_sc[best] + second_score_band[0], color='k',
                   alpha=1.,
                   linewidth=2.)
        ax_sc.axhline(y=_decision_threshold + second_score_band[0], color='r', linestyle='--')

        """Stage events"""
        bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(
            trial_data[best].navigation.data.stages)
        for _st, _en in bbls:
            st = toseg(_st)
            ax_sc.plot([t[st]], [info_band[0] + info_space / 2], marker=BABBLING_STAGE_STYLE.marker,
                       color=BABBLING_STAGE_STYLE.color, markersize=evol_markersize,
                       linewidth=3, linestyle='')
            ax_sc.axvline(x=t[st], linestyle='-', color=BABBLING_STAGE_STYLE.color, linewidth=1)
            ax_map.plot([best_nav.location[_st, 0]], [best_nav.location[_st, 1]],
                        marker=BABBLING_STAGE_STYLE.marker, color=BABBLING_STAGE_STYLE.color,
                        markersize=map_markersize
                        )

        # First model
        # first_model_on_time = get_first_model_activation_time(t, trial_data[
        #     best].segmented_model_selection.data.selected_model, scenario_phase)
        first_model_post_paralysis_times = [
            get_first_model_activation_time(td.segmented_model_selection.data.iter.astype(int) * D_T,
                                            td.segmented_model_selection.data.selected_model, scenario_phase)
            for td in trial_data]
        first_model_on_time = first_model_post_paralysis_times[best]

        for i, first_model_on_time in enumerate(first_model_post_paralysis_times):
            if first_model_on_time is not None:
                if i == best:
                    ax_sc.plot([first_model_on_time], [info_band[0] + info_space / 2],
                               MODEL_STYLES[0].marker, color=MODEL_STYLES[0].color,
                               markersize=evol_markersize, linewidth=3)
                    ax_sc.axvline(x=float(first_model_on_time), linestyle='-', color=MODEL_STYLES[0].color, linewidth=1)
                    itr = int(first_model_on_time / D_T)
                    ax_map.plot([best_nav.location[itr, 0]], [best_nav.location[itr, 1]],
                                marker=MODEL_STYLES[0].marker, color=MODEL_STYLES[0].color, markersize=map_markersize)
                else:
                    ax_sc.plot([first_model_on_time], [info_band[0] + info_space / 2],
                               MODEL_STYLES[0].marker, color=MODEL_STYLES[0].color,
                               markersize=evol_markersize, linewidth=1, alpha=0.5)

        # Second model
        second_model_post_learning_times = [
            get_second_model_activation_after_learning(
                td.segmented_model_selection.data.iter.astype(int) * D_T,
                td.segmented_model_selection.data.selected_model,
                BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(
                    td.navigation.data.stages)
            )
            for td in trial_data]
        # sec_model_time = get_second_model_activation_after_learning(t, trial_data[
        #     best].segmented_model_selection.data.selected_model, bbls)
        for i, sec_model_time in enumerate(second_model_post_learning_times):
            if sec_model_time is not None:
                if i == best:
                    ax_sc.axvline(x=float(sec_model_time), linestyle='-', color=MODEL_STYLES[1].color, linewidth=1)
                    ax_sc.plot([sec_model_time], [info_band[0] + info_space / 2],
                               MODEL_STYLES[1].marker, color=MODEL_STYLES[1].color,
                               markersize=evol_markersize, linewidth=3)
                    itr = int(sec_model_time / D_T)
                    ax_map.plot([best_nav.location[itr, 0]], [best_nav.location[itr, 1]],
                                marker=MODEL_STYLES[1].marker, color=MODEL_STYLES[1].color, markersize=map_markersize)
                else:
                    ax_sc.plot([sec_model_time], [info_band[0] + info_space / 2],
                               MODEL_STYLES[1].marker, color=MODEL_STYLES[1].color,
                               markersize=evol_markersize, linewidth=1, alpha=0.5)

        """Styling"""
        ax_sc.yaxis.set_ticks_position('right')
        for mn in range(model_num):
            ax_sc.axhline(y=0.5 + model_bands[mn][0], color='r', linestyle='--', alpha=.5)

        ax_sc.axhline(y=info_space / 2 + info_band[0], color='k', linestyle='-')
        ##
        PV.fill_boolean(ax_sc, np.arange(len(scenario_phase)), dt=D_T, y1=info_band[0], y2=info_band[1],
                        color='k', alpha=0.2)
        PV.fill_boolean(ax_sc, np.arange(len(scenario_phase)), dt=D_T, y1=second_score_band[0], y2=second_score_band[1],
                        color='k', alpha=0.2)
        for mn in range(model_num):
            PV.fill_boolean(ax_sc, np.arange(len(scenario_phase)), dt=D_T, y1=model_bands[mn][0],
                            y2=model_bands[mn][1],
                            color='k', alpha=0.2)
        PV.fill_boolean(ax_sc, scenario_phase == 1., dt=D_T, y1=ylim[0], y2=ylim[1], color=PARALYSIS_COLOR, alpha=0.5)

        # ax_sc.set_title(styles[lifecycle].label)
        ax_sc.set_ylim(*ylim)
        ax_sc.set_xlim(left=np.min(t), right=np.max(t))
        # ax_sc.set_xticks([])
        ax_sc.set_ylim(*ylim)

        ax_sc.set_yticks(
            [model_band[0] + space / 2 for model_band in model_bands] +
            [second_score_band[0] + space / 2, info_band[0] + info_space / 2])
        ax_sc.set_yticklabels([MODEL_STYLES[model_num-1-mn].label + " IM" for mn in range(model_num)] + ['Confidence', 'stage'])
        ax_sc.set_xlabel(TIME_LABEL)
        ax_map.set_xlabel(LOCATION_LABELS[0])
        ax_map.set_ylabel(LOCATION_LABELS[1])
        ax_sc.spines['top'].set_visible(False)
        ax_sc.spines['right'].set_visible(False)
        # ax_sc.spines['bottom'].set_visible(False)
        ax_sc.spines['left'].set_visible(False)
        ax_map.spines['top'].set_visible(False)
        ax_map.spines['right'].set_visible(False)
        # ax_map.spines['bottom'].set_visible(False)
        # ax_map.spines['left'].set_visible(False)

        img_path = os.path.join(resutls_path, name + f"life_cycle_stages_{styles[lifecycle].label}.png")
        plt.savefig(img_path, bbox_inches='tight')
        img_paths.append(img_path)

    agg_img = vertical_merge_images([Image.open(pth) for pth in img_paths])
    agg_img.save(os.path.join(resutls_path, name + f"life_cycle_stages.png"))
    # """LEGEND"""
    # fig()
    # C.standalone_legend(C.plt, [
    #     {"color": PARALYSIS_COLOR, "linestyle": '-', "alpha": .5, "label": 'paralysis', "linewidth": 10},
    #     {"color": BABBLING_STAGE_STYLE.color, "linestyle": '', "marker": BABBLING_STAGE_STYLE.marker,
    #      "alpha": 0.8, "label": 'babbling'},
    #     {"color": MODEL_STYLES[1].color, "linestyle": '', "marker": MODEL_STYLES[1].marker,
    #      "alpha": 0.8, "label": "using " + MODEL_STYLES[1].label + " IM"},
    #     {"color": MODEL_STYLES[0].color, "linestyle": '', "marker": MODEL_STYLES[0].marker,
    #      "alpha": 0.8, "label": "return to " + MODEL_STYLES[0].label + " IM"},
    #     # {"color": PERFORMING_COLOR, "linestyle": '-', "alpha": 0.8, "label": 'performing'},
    #     # {"color": 'r', "linestyle": '', "marker": "o", "alpha": 1., "label": 'goal'},
    #     {"color": 'b', "linestyle": '', "marker": "o", "alpha": 1., "label": 'reached'},
    #     # {"color": 'b', "linestyle": '', "marker": "o", "alpha": 1., "label": 'reached'},
    # ])
    # plt.savefig(os.path.join(resutls_path, name + "mini_map_legend.png"), bbox_inches='tight')


def _is_any(references):
    ret = np.zeros((len(references),))
    for i in range(len(references)):
        ret[i] = np.sum(references[i, 0, :]) > 5
    return ret


def _is_forwardish(references):
    ret = np.zeros((len(references),))
    for i in range(len(references)):
        ret[i] = np.sum(references[i, 0, :]) > 5 and np.asarray(np.sum(references[i, 3, :])) < 0.1
    return ret


def _is_leftish(references):
    ret = np.zeros((len(references),))
    for i in range(len(references)):
        ret[i] = np.sum(references[i, 0, :]) > 0 and np.sum(references[i, 3, :]) > 0.5
    return ret


def _is_rightish(references):
    ret = np.zeros((len(references),))
    for i in range(len(references)):
        ret[i] = np.sum(references[i, 0, :]) > 0 and np.sum(references[i, 3, :]) < -0.5
    return ret


def detail(fig: C.FigProvider, p: DecimJournalEvaluation,
           resutls_path, name):
    from coupling_evol.agent.components.internal_model.forward_model import get_embeddings_from_mem

    models = p.models.data
    """ Models """
    derivatives = []
    for m in models:
        trg_gaits = m.u_mean
        granularity = m.phase_n
        derivative = [m.derivative_gait(trg_gaits, sensory_ph) for sensory_ph in range(granularity)]
        derivatives.append(np.asarray(derivative))

    motor_id = 6
    sensor_id = 0
    plt.rcParams["figure.figsize"] = (12, 6)
    map_fig = fig()
    axs = C.subplots(map_fig, 2, len(models) + 1, gridspec_kw={'width_ratios': [8] * len(models) + [1]})
    ##
    PV.ax_generate_colorbar(axs[1][-1], NEG_POS_CMAP, minmax=(-1, 1))
    PV.ax_generate_colorbar(axs[0][-1], POS_CMAP, minmax=(0, 1))
    ##
    abs_d = np.asarray([np.mean(np.abs(d), axis=(0, 3)) for d in derivatives])
    abs_d = abs_d - np.min(abs_d, axis=(0, 2))[None, :, None]
    abs_d = abs_d / np.max(abs_d, axis=(0, 2))[None, :, None]

    phph_d = np.asarray([der[:, sensor_id, motor_id, :] for der in derivatives])
    phph_d -= np.min(phph_d)
    phph_d /= np.max(phph_d)

    norm = plt.Normalize(vmin=0, vmax=1)
    # abs_norm = None

    for i in range(len(models)):
        ax_phph_d = axs[1][i]
        ax_all_d = axs[0][i]

        #
        ax_phph_d.matshow(phph_d[i], norm=norm, cmap=plt.cm.get_cmap(NEG_POS_CMAP))
        ax_phph_d.xaxis.set_ticks_position('bottom')
        if i == 0:
            ax_phph_d.set_ylabel("Sensory phase")
        ax_phph_d.set_yticks([i for i in range(4)])
        ax_phph_d.set_yticklabels([f"c{i + 1}" for i in range(4)])
        # else:
        #     ax_phph_d.set_yticklabels([])

        label = f"w^{{{SENSORY_LABELS[sensor_id]},c}}_{{{LEG_LABELS[motor_id // 3]}, d}}"
        ax_phph_d.text(-1, -0.8, r"${}$".format(label))

        ax_phph_d.set_xlabel("Motor phase")
        ax_phph_d.set_xticks([i for i in range(4)])
        ax_phph_d.set_xticklabels([f"d {i + 1}" for i in range(4)])

        #
        ax_all_d.matshow(abs_d[i][:5, :], norm=norm, cmap=plt.cm.get_cmap(POS_CMAP))

        if i == 0:
            ax_all_d.set_yticks([i for i, _ in enumerate(SENSORY_LABELS)])
            ax_all_d.set_yticklabels([lab for lab in SENSORY_LABELS])
            ax_all_d.set_ylabel("Sensory modality")
        else:
            ax_all_d.set_yticklabels([])

        ax_all_d.set_xticks([i * 3 for i in range(6)])
        # stage_f.set_xticklabels([event[1] for event in events], rotation=0, ha='center', minor=False)
        ax_all_d.set_xticklabels([LEG_LABELS[i] for i in range(6)], rotation=-45, ha='left', minor=False)
        # ax_all_d.set_xlabel("Motor")
        ax_all_d.xaxis.set_ticks_position('bottom')

        axs[0][i].set_title(MODEL_STYLES[i].label + " IM")
    ##

    plt.savefig(os.path.join(resutls_path, name + "models.png"), bbox_inches='tight')

    """Sensory-wise general model selection"""

    offset = 0.1
    shift = 0.4
    space = 2

    info_space = 0.4
    evol_markersize = 10
    map_markersize = 8

    model_score_band = (0, space)
    second_score_band = (model_score_band[1] + shift, model_score_band[1] + shift + space)
    # info_band = (second_score_band[1] + shift, second_score_band[1] + shift + info_space)
    ylim = (-offset, second_score_band[1] + offset)

    _scenario_phase = p.scenario_switch.data.phase
    _stages = p.navigation.data.stages

    seg_iter = p.segmented_model_selection.data.iter.astype(int)
    scenario_phase = _scenario_phase[seg_iter]
    stages = _stages[seg_iter]
    seg_n = len(seg_iter)
    itern_n = len(_scenario_phase)
    sg_p_it = seg_n / itern_n
    it_p_sg = itern_n / seg_n
    # iter_int = seg_iter.astype(dtype=int)

    # ZERO MODEL
    _confidence = p.segmented_model_selection.data.best_zero_confidence
    # confidence = np.exp(_confidence)
    # confidence = -np.tanh(_confidence / (np.std(_confidence, axis=0)[None, :, :]))
    confidence = np.clip((-_confidence / (np.std(_confidence, axis=0)[None, :, :]))/2, a_min=-0.9, a_max=0.9)
    # IM MODEL
    log_odds = p.segmented_model_selection.data.model_logodds
    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    los = np.zeros_like(log_odds) + log_odds[:, 0, :, :][:, None, :, :]
    for j in range(los.shape[1] - 1):
        st, en = prfs[j + 1]
        sgen = int((st + 100) * sg_p_it)
        los[sgen:, j + 1, :, :] = log_odds[sgen:, j + 1, :, :]
    # _odds = los[:, 0, :, :] - los[:, 1, :, :]
    # odds = np.tanh(_odds / (np.std(_odds, axis=0)[None, :, :]))
    _odds = np.exp(los * 100)
    odds = _odds[:, 0, :, :] / np.sum(_odds, axis=1) * 2 - 1
    st, en = prfs[0]
    odds[:en, :, :] = 1
    # Draw
    plt.rcParams["figure.figsize"] = (12, 6)
    ms_fig = fig()
    labels = SENSORY_LABELS[:5]
    ts = seg_iter * D_T
    figs = C.subplots(ms_fig, len(labels) + 1, 1, gridspec_kw={'height_ratios': [4] + [10] * len(labels)})
    for i in range(len(labels)):
        f = figs[i + 1][0]
        f.plot(ts, 1 + odds[:, i, 0], color=MODEL_STYLES[0].color, linewidth=2)
        f.plot(ts, 1 + confidence[:, i, 0] + second_score_band[0], color=PERFORMING_COLOR, linewidth=2)

        ##
        PV.fill_boolean(f, scenario_phase == 1., y1=ylim[0], y2=ylim[1], dt=it_p_sg * D_T, color=PARALYSIS_COLOR,
                        alpha=0.5)
        PV.fill_boolean(f, stages == int(BabblePerformanceAlternation.StageStates.BABBLING_STAGE.value[0]),
                        y1=ylim[0], y2=ylim[1], dt=it_p_sg * D_T, color=BABBLING_COLOR, alpha=0.5)

        f.axhline(y=space / 2, color='r', linestyle='--')
        f.axhline(y=second_score_band[0] + space / 2, color='r', linestyle='--')

        PV.fill_boolean(f, np.arange(len(scenario_phase)), dt=it_p_sg * D_T, y1=model_score_band[0],
                        y2=model_score_band[1],
                        color='k', alpha=0.1)
        PV.fill_boolean(f, np.arange(len(scenario_phase)), dt=it_p_sg * D_T, y1=second_score_band[0],
                        y2=second_score_band[1],
                        color='k', alpha=0.1)
        f.spines['right'].set_visible(False)
        f.spines['left'].set_visible(False)
        f.spines['top'].set_visible(False)
        f.spines['bottom'].set_visible(False)
        f.yaxis.set_ticks_position('right')
        f.set_yticks([space / 2, second_score_band[0] + space / 2])
        f.set_yticklabels(["competition", "zero-model"])
        f.set_ylabel(labels[i])
        f.set_xlim(np.min(ts), np.max(ts))
        if i != (len(labels) - 1):
            f.set_xticks([])
    figs[-1][0].set_xlabel(TIME_LABEL)
    ## STAGE
    stage_f = figs[0][0]
    # switch = p.scenario_switch.data.phase
    seg_it = p.segmented_model_selection.data.iter.astype(dtype=int)
    prfs = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)
    score_1 = p.segmented_model_selection.data.first_score
    min_sc = np.min(score_1)
    score_1f = np.zeros_like(score_1) + min_sc
    score_1f[:, 0] = score_1[:, 0]
    for j in range(los.shape[1] - 1):
        st, en = prfs[j]
        sgen = int(en * sg_p_it)
        score_1f[sgen:, j + 1] = score_1[sgen:, j + 1]
    sel_mod_id = p.segmented_model_selection.data.selected_model
    ## Draw
    stage_f.set_xlim(np.min(ts), np.max(ts))
    stage_f.set_ylim(0, 1)
    stage_f.yaxis.set_ticks_position('right')
    stage_f.xaxis.set_ticks_position('top')
    stage_f.set_yticks([0.5])
    stage_f.set_yticklabels(['stage'])
    stage_f.spines['right'].set_visible(False)
    stage_f.spines['left'].set_visible(False)
    stage_f.spines['top'].set_visible(False)
    stage_f.spines['bottom'].set_visible(False)

    time_path = np.asarray([ts, [0.5] * len(ts)]).T

    PV.fill_boolean(stage_f, _scenario_phase == 1., y1=0, y2=1, dt=D_T, color=PARALYSIS_COLOR, alpha=0.5)

    seg_location = time_path
    mod_times = []
    for i in range(2):
        _l = np.zeros_like(seg_location) + seg_location
        _l[sel_mod_id != i, :] = None
        mod_times.append(_l[:, 0][::20])
        stage_f.plot(_l[:, 0][::20], _l[:, 1][::20],
                     linestyle='-', linewidth=10,
                     color=MODEL_STYLES[i].color,
                     # marker=MODEL_STYLES[i].marker,
                     label=MODEL_STYLES[i].label + " IM", alpha=0.8)

    PV.draw_staged_path(ax=stage_f,
                        xy_translation=time_path,
                        yaw=np.zeros((len(ts),)),
                        stages=stages,
                        show_yaw=False,
                        vector_draw_granularity=2000,
                        babbling_style={"color": BABBLING_STAGE_STYLE.color, "linestyle": '-', "alpha": 0.8,
                                        "linewidth": 3},
                        performing_style={"color": PERFORMING_COLOR, "linestyle": '-', "alpha": 0.8, "linewidth": 3},
                        )
    rec_t = ts[scenario_phase == 1.][-1]
    walkim_t = ts[sel_mod_id == 1]
    events = [
        (0, f"{MODEL_STYLES[0].label} IM"),
        (ts[scenario_phase == 1.][0], "paralysis"),
        (ts[stages == 1][0], "babbling"),
        # (mod_times[1][~np.isnan(mod_times[1])][0], "Paralysis IM"),
        (ts[stages == 1][-1], f"{MODEL_STYLES[1].label} IM"),
        (rec_t, "recovery"),
        # (ts[sel_mod_id == 1][-1], f"{MODEL_STYLES[0].label} IM"),
        (walkim_t[walkim_t > rec_t][-1] + 30, f"{MODEL_STYLES[0].label} IM"),
    ]
    [stage_f.axvline(x=event[0], color='k', linestyle='-') for event in events]

    stage_f.set_xticks([event[0] for event in events])
    stage_f.set_xticklabels([event[1] for event in events], rotation=0, ha='center', minor=False)

    plt.savefig(os.path.join(resutls_path, name + "intermodel_competition.png"), bbox_inches='tight')

    """Gaits"""
    umem = p.uyt_mem.data.command
    tmem = p.uyt_mem.data.target
    amem = p.uyt_mem.data.segments
    ymem = p.uyt_mem.data.observation

    score_1 = p.segmented_model_selection.data.first_score
    u_embs = get_embeddings_from_mem(umem, amem)
    y_embs = get_embeddings_from_mem(ymem, amem)

    goes_forward = _is_any(y_embs)
    goes_left = _is_leftish(y_embs)
    goes_right = _is_rightish(y_embs)
    # best_models = np.argmax(score_1, axis=1)[3:]
    best_models = p.segmented_model_selection.data.selected_model[3:]

    prims = []
    for i in range(2):
        fwd = np.where(np.logical_and(goes_forward, best_models == i))[0]
        lft = np.where(np.logical_and(goes_left, best_models == i))[0]
        rgh = np.where(np.logical_and(goes_right, best_models == i))[0]
        prims.append([
            # np.std(u_embs[rgh[-100:], :, :], axis=0),
            # np.std(u_embs[fwd[-100:], :, :], axis=0),
            np.std(u_embs[lft[-100:], :, :], axis=0)
        ])

    plt.rcParams["figure.figsize"] = (12, 3)
    gait_fig = fig()
    axs = C.subplots(gait_fig, 1, len(prims))

    norm = plt.Normalize(vmin=0, vmax=1)

    for i, prim in enumerate(prims):
        for j in range(1):
            _ax = axs[j][i]
            _p = prim[j].T / np.max(prim[j])
            matxs = _ax.matshow(_p, norm=norm, cmap=plt.cm.get_cmap(POS_CMAP))

            _ax.set_xticks([i * 3 for i in range(6)])
            _ax.set_xticklabels([LEG_LABELS[i] for i in range(6)], rotation=-45, ha='left', minor=False)
            # if j == len(axs) - 1:
            #     _ax.set_xlabel("Motor")
            _ax.xaxis.set_ticks_position('bottom')

            _ax.set_yticks([i for i in range(4)])
            _ax.set_yticklabels([f"d{i + 1}" for i in range(4)])
    axs[0][0].set_ylabel("Motor phase")

    # _ax.yaxis.set_ticks_position('right')

    # for i, lab in enumerate(PRIMITIVE_LABELS):
    #     axs[i][0].set_ylabel(lab)
    for i in range(2):
        axs[0][i].set_title("Gait deviation during " + MODEL_STYLES[i].label)
    ##
    mlab = gait_fig.add_subplot(1, 1, 1)
    mlab.set_xticks([])
    mlab.set_yticks([])
    [mlab.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
    mlab.patch.set_visible(False)
    mlab.yaxis.set_label_position('right')
    # mlab.set_ylabel('Motor phase', labelpad=30)
    ##
    gait_fig.subplots_adjust(right=0.8)
    cbar_ax = gait_fig.add_axes([0.85, 0.15, 0.05, 0.7])
    gait_fig.colorbar(matxs, cax=cbar_ax)
    ##
    plt.savefig(os.path.join(resutls_path, name + "robot_gait.png"), bbox_inches='tight')

    """Map"""
    location = p.navigation.data.location
    goal = p.navigation.data.goal
    heading = p.navigation.data.heading
    stages = p.navigation.data.stages
    switch = p.scenario_switch.data.phase
    score_1 = p.segmented_model_selection.data.first_score
    seg_it = p.segmented_model_selection.data.iter.astype(dtype=int)

    prfs = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)
    min_sc = np.min(score_1)
    score_1f = np.zeros_like(score_1) + min_sc
    score_1f[:, 0] = score_1[:, 0]
    for j in range(los.shape[1] - 1):
        st, en = prfs[j]
        sgen = int(en * sg_p_it)
        score_1f[sgen:, j + 1] = score_1[sgen:, j + 1]

    sel_mod_id = np.argmax(score_1f, axis=1)

    plt.rcParams["figure.figsize"] = (6, 6)
    map_fig = fig()
    axs = C.subplots(map_fig, 1, 1)
    ax = axs[0][0]
    ax.plot(location[switch == 1, 0], location[switch == 1, 1], '-', color=PARALYSIS_COLOR, linewidth=20,
            label="paralysis", alpha=0.5)
    seg_location = location[seg_it, :]
    for i in range(2):
        _l = np.zeros_like(seg_location) + seg_location
        _l[sel_mod_id != i, :] = None
        ax.plot(_l[:, 0][::200], _l[:, 1][::200],
                linestyle='-', linewidth=10,
                color=MODEL_STYLES[i].color,
                # marker=MODEL_STYLES[i].marker,
                label=MODEL_STYLES[i].label + " IM", alpha=0.8)

    # ds = np.abs(switch[1:] - switch[:-1])
    # ax.plot(location[:-1][ds == 1, 0], location[:-1][ds == 1, 1], '^', color=PARALYSIS_COLOR, linewidth=8,
    #         label="paralysis")

    PV.xy_heading_stages_map(ax, xy_translation=location, xy_goal=goal)
    PV.draw_staged_path(ax=ax,
                        xy_translation=location,
                        yaw=heading,
                        stages=stages,
                        show_yaw=False,
                        vector_draw_granularity=2000,
                        babbling_style={"color": BABBLING_COLOR, "linestyle": '-', "alpha": 0.8, "linewidth": 3},
                        performing_style={"color": PERFORMING_COLOR, "linestyle": '-', "alpha": 0.8, "linewidth": 3},
                        )
    ax.plot(location[-1, 0], location[-1, 1], 'bo', label="reached")
    ax.set_ylim(-110, 10)
    ax.set_xlim(-110, 10)

    # axs[0][-1].legend(loc='upper left')
    ax.set_xlabel(LOCATION_LABELS[0])
    ax.set_ylabel(LOCATION_LABELS[1])
    plt.savefig(os.path.join(resutls_path, name + "navigation_detail.png"), bbox_inches='tight')
    ##
    # C.standalone_legend(C.plt, [
    #     {"color": BABBLING_COLOR, "linestyle": '-', "alpha": 0.8, "label": 'babbling'},
    #     {"color": PERFORMING_COLOR, "linestyle": '-', "alpha": 0.8, "label": 'performing'},
    #     {"color": 'r', "linestyle": '', "marker": "o", "alpha": 1., "label": 'goal'},
    #     {"color": 'b', "linestyle": '', "marker": "o", "alpha": 1., "label": 'reached'},
    #     {"color": PARALYSIS_COLOR, "linestyle": '-', "alpha": .5, "label": 'paralysis event', "linewidth": 10},
    #     {"color": MODEL_STYLES[0].color, "linestyle": '-', "alpha": 1., "label": MODEL_STYLES[0].label + "IM",
    #      "linewidth": 10},
    #     {"color": MODEL_STYLES[1].color, "linestyle": '-', "alpha": 1., "label": MODEL_STYLES[1].label + "IM",
    #      "linewidth": 10},
    # ])
    # plt.savefig(os.path.join(resutls_path, name + "detail_map_legend.png"), bbox_inches='tight')

def error_stats(pa: PortfolioAggregate[DecimJournalEvaluation], convolution_window=100):
    lifecycles = _get_ordered_lc_list(pa.lifecycle_range())
    trials = pa.trial_range()
    environments = pa.environment_range()
    scenarios = pa.scenario_range()
    env_id = environments[0]
    sc_id = scenarios[0]
    min_length = min([len(p.uyt_mem.data.target) for p, _ in pa.agg])

    styles = _get_lc_style_dictionary(lifecycles)
    first_tag = Tag(trial=trials[0], scenario=scenarios[0], lifecycle=lifecycles[0],
                    environment=environments[0])
    scenario_phases = pa.get(first_tag).scenario_switch.data.phase
    iter_len = len(pa.get(first_tag).uyt_mem.data.performing_stage)
    iter_seg_ratio = len(scenario_phases) // iter_len
    scenario_phases_seg = scenario_phases[::iter_seg_ratio]

    lifecycle_diff = []
    for i, lifecycle in enumerate(lifecycles):
        lifecycle_diff.append(get_performance_diffs(
            [pa.get(Tag(trial=trial, scenario=sc_id, lifecycle=lifecycle, environment=env_id)).uyt_mem.data for trial in
             trials], min_length))

    means, stds = PV.comparison_stats(lifecycle_diff, [styles[lc].label for lc in lifecycles], PERFORMANCE_METRICS,
                                      mean_window=convolution_window)

    lifecycle_diff = []
    for i, lifecycle in enumerate(lifecycles):
        nav_data = [pa.get(Tag(trial=trial, scenario=sc_id, lifecycle=lifecycle, environment=env_id)).navigation.data
                    for trial in trials]
        lifecycle_diff.append([(nav.location - nav.goal) for nav in nav_data])

    _means, _stds = PV.comparison_stats(lifecycle_diff, [styles[lc].label for lc in lifecycles], GOAL_METRICS,
                                        mean_window=convolution_window)

    means += _means
    stds += _stds

    return means, stds


def extract_stats(output_path, pas: List[PortfolioAggregate[DecimJournalEvaluation]], pa_labels=List[str], name=""):
    means = []
    stds = []
    lifecycles = _get_ordered_lc_list(pas[0].lifecycle_range())
    styles = _get_lc_style_dictionary(lifecycles)
    scenario_phases = pas[0].agg[0][0].scenario_switch.data.phase
    sc_diff = scenario_phases[1:] - scenario_phases[:-1]
    print(f"T> total:{len(sc_diff)}, paralysis_start: {np.argmax(sc_diff)}, paralysis_end: {np.argmin(sc_diff)}")

    for pa in pas:
        mean, std = error_stats(pa, convolution_window=100)
        means.append(mean)
        stds.append(std)

    metric_labels = [met[1] for met in (PERFORMANCE_METRICS + GOAL_METRICS)]

    for i, prf in enumerate(metric_labels):
        data = np.asarray([[means[k][i][j] for k in range(len(means))] for j in range(len(means[0][i]))])
        _argmin = np.argmin(data, axis=0)
        order = np.log10(means[0][i][0])
        scale = 1
        if order > 3:
            scale = np.power(10, 2 - np.round(order))
        print(f"--------------------{prf}[scale: {scale}]")
        print("method, " + ",".join(pa_labels))
        for j, lc in enumerate(lifecycles):
            line = f"{styles[lc].label}"
            for k in range(len(pas)):
                if _argmin[k] != j:
                    line += f" & ${means[k][i][j] * scale:.1f}({stds[k][i][j] * scale:.1f})$"
                else:
                    line += f" & $\\mathbf{{{means[k][i][j] * scale:.1f}({stds[k][i][j] * scale:.1f})}}$"
            print(line + "\\\\")


def generate_portfolio(output_path, pa: PortfolioAggregate[DecimJournalEvaluation], name=None):
    if name is None:
        lc_name = pa.lifecycle_range()[0]
    else:
        lc_name = name
    fig = C.FigProvider()

    error_figs(fig, pa, resutls_path=output_path, name=lc_name)
    model_selection_comparison(fig, pa, resutls_path=output_path, name=lc_name)


def generate_detail(output_path, dp: DecimJournalEvaluation, name):
    """Detail"""
    fig = C.FigProvider()
    detail(fig, dp, resutls_path=output_path, name=name)
    fig.close_all()


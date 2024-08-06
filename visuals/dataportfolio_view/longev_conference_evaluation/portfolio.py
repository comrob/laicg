from coupling_evol.data_process.postprocess.permanent_post_processing.common import ModelsWrap
from coupling_evol.data_process.postprocess.permanent_post_processing.data_portfolios import DecimJournalEvaluation, LongevEvaluation
from coupling_evol.data_process.postprocess.permanent_post_processing.helpers import parameter_parser, \
    PortfolioAggregate, Tag
import visuals.dataportfolio_view.decim_journal_evaluation.pretty_visuals as PV
import numpy as np
from coupling_evol.engine.embedded_control import EmbeddedTargetParameter
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
from visuals import common as C
##
from coupling_evol.agent.components.internal_model.forward_model import get_embeddings_from_mem
import coupling_evol.data_process.postprocess.permanent_post_processing.factories as PF


SEGMENT_DURATION = 0.2172
ITER_DURATION = 0.0104
SEGMENT_ITERATIONS = int(SEGMENT_DURATION / ITER_DURATION)

LABEL_SIZE = 20
TICK_SIZE = 15
PLOT_WIDTH = 18
PLOT_ROW_HEIGHT = 2

COLUMN_WIDTH = 12
BIT_SIZE = 3

VICINITY_CUT = 0.4
NEG_POS_CMAP = "coolwarm"
# NEG_POS_CMAP = "Spectral"
POS_CMAP = "Spectral"

class Style:
    def __init__(self, label, color, marker='.'):
        self.label: str = label
        self.color: str = color
        self.marker: str = marker


PERFORMING_COLOR = 'k'
BABBLING_COLOR = 'g'
PARALYSIS_COLOR = 'y'
LOGODDS_COLOR = 'k'

LEG_COLORS = ["darkorange","darkorange", "darkviolet","darkviolet", "darkgreen", "darkgreen"]

BABBLING_STAGE_STYLE = Style('babble', color=BABBLING_COLOR, marker='s')

MODEL_COLORS = ['darkorange', 'fuchsia', 'g', 'peru', 'y', 'darkviolet', 'orangered', 'navyblue', 'springgreen']
MODEL_MARKERS = ["^", "v", "o", "s", "x"]
MODEL_STYLES = [
    Style(str(i+1), color=MODEL_COLORS[i%len(MODEL_COLORS)], marker=MODEL_MARKERS[i%len(MODEL_MARKERS)])
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

SENSORY_LABELS = ["Head", "Roll", "Pitch", "Yaw", "Side"]
LEG_LABELS = ["L1", "R1", "L2", "R2", "L3", "R3"]
LEG_SERVO_IDS = [
    (0,1,2),
    (3,4,5),
    (6,7,8),
    (9,10,11),
    (12,13,14),
    (15,16,17)]

LOCATION_LABELS = ("X location (m)", "Y location (m)")

PRIMITIVE_LABELS = ["Left", "Forward", "Right"]

LEG_DISPLACEMENT_MAGNITUDE_LABEL = "Magnitude"
GOAL_DISTANCE_LABEL = "Distance"
PERFORMANCE_ERROR_LABEL = "Normalized difference"
CORRELATE_CMAP = "plasma"

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
GOAL_METRICS = [(lambda diff: 200 - np.linalg.norm(diff, axis=1), "Goal distance")]


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
        ret[lc] = Style(label=lc, color=LEG_LABELS[i])
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


def iter_n_of_reaching_vicinity(nav_data, vicinity):
    xy = nav_data.location[10:,:]
    start = xy[0]
    xy = xy - start
    goal = nav_data.goal[-1,:] - start
    inside = np.where(np.linalg.norm(goal - xy, axis=1) < vicinity)
    if len(inside) == 0 or len(inside[0]) == 0:
        return len(xy)-1
    return np.min(inside[0])+10


def reference_tracking(
        fig, results_path, name,
        uyt_mem: PF.dynamic_lifecycle.MotorSensorTargetMem, models: ModelsWrap,
        ensemble_dynamics: PF.model_selection.SegmentedScores,
        events: List[Tuple[str, int]],
        fep_signals: PF.controller.SegmentedFepSignals = None,
        vicinity_cut_segment: int = None
        ):
    sel_legs = [0,1,2,3,4,5]
    sel_sensors = [0,3]

    if vicinity_cut_segment is not None:
        vicinity_cut = vicinity_cut_segment
    else:
        vicinity_cut = len(uyt_mem.command-1)

    convolution_window = 10
    epsilon = 0.5
    ms = models.data
    u_mem = uyt_mem.command[:vicinity_cut]
    y_mem_obs = uyt_mem.observation[:vicinity_cut]
    y_mem_trg = uyt_mem.target[:vicinity_cut]
    phases = uyt_mem.segments[:vicinity_cut]
    if fep_signals is not None:
        y_mem_obs = fep_signals.y_estimation[:vicinity_cut]
    ##
    y_embs_obs = get_embeddings_from_mem(y_mem_obs, phases)
    y_embs_trg = get_embeddings_from_mem(y_mem_trg, phases)

    # u_mem to embeddings
    u_embs = get_embeddings_from_mem(u_mem, phases)
    u_embs = u_embs.reshape((u_embs.shape[0], 1, u_embs.shape[1], u_embs.shape[2]))
    y_mem_preds = [m.predict(u_embs, phases[phases.shape[1]:, :]) for m in ms]
    y_mem_obs = y_mem_obs[phases.shape[1]-1:, :]
    y_mem_trg = y_mem_trg[phases.shape[1]-1:, :] #/ phases.shape[1] # normalise target as the sum of the phase
    ##
    y_emb_responses = [np.asarray([m.predict_gait_response(np.asarray([gait[0, :, :]]))[0, :, :] for gait in u_embs]) for m in ms]
    ##
    y_preds_embs = [get_embeddings_from_mem(y_mem_pred, phases[phases.shape[1]:, :]) for y_mem_pred in y_mem_preds]
    y_embs_obs = y_embs_obs[phases.shape[1]:, :]
    y_embs_trg = y_embs_trg[phases.shape[1]:, :]

    ##

    t_axis = np.arange(0, len(u_mem))/phases.shape[1]

    def smooth_signal(signal):
        v = np.ones((convolution_window * 2,)) / convolution_window
        v[convolution_window:] = 0
        return np.convolve(signal, v=v, mode="valid")

    def out_epsilon(signal, _epsilon=epsilon):
        asig = np.abs(signal)
        return ((-_epsilon > asig) + (asig > _epsilon)).astype(np.float32)


    """----------------------Reference tacking evolution"""
    # visualise y_mems into figure
    plt.rcParams["figure.figsize"] = (PLOT_WIDTH, PLOT_ROW_HEIGHT * (len(sel_sensors)+1))

    f = fig()
    axs = C.subplots(f, len(sel_sensors) + 1, 1)
    for i, sns_id in enumerate(sel_sensors):
        ax = axs[i][0]
        # draw vertical line at stage switches
        for ev in events:
            segment, _ = ev
            ax.axvline((segment-convolution_window/2)/phases.shape[1], color="r", linestyle="--", alpha=0.5)
        # horizontal line on zero
        _y_mem_trg = smooth_signal(y_mem_trg[:, sns_id])
        ax.plot(t_axis[:len(_y_mem_trg)], _y_mem_trg, label="Reference", color="r", alpha=0.9)
        _y_mem_obs = smooth_signal(np.sum(y_embs_obs[:, sns_id, :], axis=1))
        ax.plot(t_axis[:len(_y_mem_obs)], _y_mem_obs, label="Estimation", color="k", alpha=0.9)

    ax_info = axs[-1][0]
    servo_magnitude = np.sqrt(np.sum(np.square(u_embs[:,0,:,:]), axis=(1,2)))
    legs_magnitude = smooth_signal(servo_magnitude)
    ax_info.plot(t_axis[:len(legs_magnitude)], legs_magnitude, color="b", alpha=0.9)
    for ev in events:
        segment, _ = ev
        ax_info.axvline((segment-convolution_window/2)/phases.shape[1], color="r", linestyle="--", alpha=0.5)

        # for j, y_mem_pred in enumerate(y_emb_responses):
            # _y_mem_pred = smooth_signal(np.sum(y_mem_pred[:, sns_id, :], axis=1))
            # ax.plot(_y_mem_pred, label=f"Model {j}", color=MODEL_COLORS[j], alpha=0.8)

    # styling
    for i in range(len(axs)):
        ax = axs[i][0]
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        if i < len(sel_sensors):
            ax.set_ylabel(SENSORY_LABELS[sel_sensors[i]])
        else:
            ax.set_ylabel("Control magnitude")
        ax.grid()
        ax.set_xlim(0, t_axis[len(_y_mem_trg)-1])
        if i < len(axs) - 1:
            # ax.set_xticks([])
            ax.set_xticklabels([])

    # f.suptitle("Reference evolution")
    axs[-1][0].set_xlabel(TIME_LABEL)
    # axs[-1][0].legend()

    plt.savefig(os.path.join(results_path, f"{name}reference_tracking.png"), bbox_inches='tight')

    """----------------------Per leg magnitude"""
    plt.rcParams["figure.figsize"] = (PLOT_WIDTH, PLOT_ROW_HEIGHT)

    f = fig()
    axs = C.subplots(f, 1, 1)
    ax = axs[0][0]
    for i, leg_id in enumerate(sel_legs):
        for ev in events:
            segment, _ = ev
            ax.axvline((segment-convolution_window/2)/phases.shape[1], color="r", linestyle="--", alpha=0.5)

        servo_magnitude = np.sqrt(np.sum(np.square(u_embs[:,0,:,:]), axis=2))
        _leg_magnitude = np.linalg.norm(servo_magnitude[:, leg_id*3:leg_id*3+3], axis=1)
        leg_magnitude = smooth_signal(_leg_magnitude)
        if leg_id%2 == 0:
            line_style = "--"
        else:
            line_style = "-"
        ax.plot(t_axis[:len(leg_magnitude)], leg_magnitude, label=LEG_LABELS[leg_id], color=LEG_COLORS[leg_id], alpha=0.9, linestyle=line_style, linewidth=2)

    ## styling
    # f.suptitle("Per leg magnitude")
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(LEG_DISPLACEMENT_MAGNITUDE_LABEL)
    ax.set_xlim(0, t_axis[len(leg_magnitude)-1])
    ax.grid()
    axs[0][0].legend()
    plt.savefig(os.path.join(results_path, f"{name}magnitude_per_leg.png"), bbox_inches='tight')

    """----------------------Preformance error"""
    prf_err = (y_mem_trg-y_mem_obs)[:,sel_sensors]
    prf_err_nrm = (prf_err)/np.std(prf_err, axis=0)

    u_prior_diff = u_embs[:,0,:,:] - ms[-1].u_mean
    u_prior_diff_nrm = (u_prior_diff)/np.std(u_prior_diff, axis=0)
    u_prior_diff_nrm /= 6

    prf_err_dist = np.sum(np.square(prf_err_nrm), axis=1)
    u_prior_diff_dist = np.sum(np.square(u_prior_diff_nrm), axis=(1,2))
    d_u_prior_diff = np.abs(u_prior_diff_dist[1:] - u_prior_diff_dist[:-1])

    smth_prf_err = smooth_signal(prf_err_dist)
    # smth_prf_err = prf_err_dist
    smth_u_prior_diff = smooth_signal(u_prior_diff_dist)
    # smth_u_prior_diff = u_prior_diff_dist

    plt.rcParams["figure.figsize"] = (PLOT_WIDTH, PLOT_ROW_HEIGHT)
    f = fig()
    axs = C.subplots(f, 1, 1)
    ax = axs[0][0]
    ###
    for ev in events:
        segment, _ = ev
        ax.axvline((segment-convolution_window/2)/phases.shape[1], color="r", linestyle="--", alpha=0.5)
    # horizontal line on zero
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.plot(t_axis[:len(smth_prf_err)],smth_prf_err, label="Performance error", color="r", alpha=0.9)
    ax.plot(t_axis[:len(smth_u_prior_diff)], smth_u_prior_diff, label="Gait change", color="k", alpha=0.9)

    ## styling
    # f.suptitle("Performance error")
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(PERFORMANCE_ERROR_LABEL)
    ax.set_xlim(0, t_axis[len(smth_prf_err)-1])
    ax.grid()
    axs[0][0].legend()
    plt.savefig(os.path.join(results_path, f"{name}performance_gait_change.png"), bbox_inches='tight')


def map_fig(
        fig, results_path, name,
        navigation: PF.coppelia_environment.Navigation,
        events: List[Tuple[str, int]],
        vicinity_cut_iter: int = None
        ):
    """
    MAP
    """
    box = (-1, 6, -3, 4)
    ##
    if vicinity_cut_iter is not None:
        vicinity_iter = vicinity_cut_iter
    else:
        vicinity_iter = len(navigation.location)
    ##
    nav = navigation
    start_location = nav.location[10]
    xy = (nav.location - start_location)[:vicinity_iter]
    head = nav.heading[:vicinity_iter]
    goal = nav.goal[-1, :] - start_location


    t_axis = np.arange(0, len(xy)) * D_T

    plt.rcParams["figure.figsize"] = (PLOT_WIDTH/2, PLOT_WIDTH/2)
    map_fig = fig()
    axs = C.subplots(map_fig, 1, 2, gridspec_kw={'width_ratios': [PLOT_WIDTH / 2] + [0.1]})
    ax = axs[0][0]
    # ax.plot(xy[:, 0], xy[:, 1], label="Location", color="k", alpha=0.9)
    # ax.plot(xy[:, 0], xy[:, 1], linestyle='', marker='o', label="Location", color="k", alpha=0.005)
    # for each event add marker with a labeled arrow
    for ev in events:
        _segment, label = ev
        segment = int((_segment / 6)/D_T)
        ax.plot(xy[segment, 0], xy[segment, 1], linestyle='', marker='x', markersize=10, label=label, color="r", alpha=0.9)
        # ax.arrow(xy[segment, 0], xy[segment, 1], np.cos(head[segment])*0.3, np.sin(head[segment])*0.3, head_width=0.1, head_length=0.1, fc="r", ec="r", alpha=0.7)


    # scatter location with the gradient coloring
    nrm_scale = np.pi/3
    nrm_head = np.clip((head + nrm_scale/2) / nrm_scale, 0, 1)
    cmap = plt.cm.get_cmap(POS_CMAP)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=[cmap(i) for i in nrm_head], alpha=1, s=4)
    # show goal with a circle of radius 0.5
    ax.add_patch(plt.Circle((goal[0], goal[1]), 0.5, color="r", alpha=0.5))
    # add large white goal marker
    ax.plot([goal[0]], [goal[1]], linestyle='', marker='o', markersize=10, label="Goal", color="w", alpha=0.9)

    PV.ax_generate_colorbar(axs[0][-1], POS_CMAP, minmax=(-nrm_scale, nrm_scale))
    # ax.scatter(xy[:, 0], xy[:, 1], color="k", alpha=0.005)
    ## styling
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_xticks([i for i in range(box[0], box[1]+1)])
    ax.set_yticks([i for i in range(box[2], box[3]+1)])
    # ax.set_xticklabels([str(i) for i in range(-5, 6)])
    ax.set_xlabel(LOCATION_LABELS[0])
    ax.set_ylabel(LOCATION_LABELS[1])
    ax.grid()
    #grey background
    ax.set_facecolor('grey')
    plt.savefig(os.path.join(results_path, f"{name}navigation.png"), bbox_inches='tight')

    """
    Goal distance
    """
    plt.rcParams["figure.figsize"] = (PLOT_WIDTH/2, PLOT_WIDTH/2)
    goal_fig = fig()
    axs = C.subplots(goal_fig, 1, 1)
    ax = axs[0][0]
    ax.plot(t_axis, np.linalg.norm(goal - xy, axis=1), color="k", alpha=0.9)
    #styling
    # goal_fig.suptitle("Goal distance")
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel("Distance (m)")
    ax.set_xlim(0, t_axis[-1])
    ax.grid()

    plt.savefig(os.path.join(results_path, f"{name}goal_distance.png"), bbox_inches='tight')


def joint_evol_map_reference_tracking(
        fig, results_path, name,
        uyt_mem: PF.dynamic_lifecycle.MotorSensorTargetMem,
        models: ModelsWrap,
        navigation: PF.coppelia_environment.Navigation,
        fep_signals: PF.controller.SegmentedFepSignals,
        events: List[Tuple[str, int]],
        time_record: PF.dynamic_lifecycle.SegmentedTimeAxis = None,
        ):
    ## PARAMS
    from matplotlib.gridspec import GridSpec

    box = (-1, 6, -3, 4)
    y_lims = [(0, 10), (-5, 5), (0, 7.5)]
    nav = navigation
    vicinity_cut_iter = iter_n_of_reaching_vicinity(nav, VICINITY_CUT)
    # vicinity_cut_segment = int((vicinity_cut_iter * D_T)* 6)
    vicinity_cut_segment = int(vicinity_cut_iter/SEGMENT_ITERATIONS)
    # sel_legs = [0,1,2,3,4,5]
    sel_sensors = [0,3]
    convolution_window = 3

    def smooth_signal(signal):
        v = np.ones((convolution_window * 2,)) / convolution_window
        v[convolution_window:] = 0
        return np.convolve(signal, v=v, mode="valid")
    ##
    ## DATA

    nav = navigation
    start_location = nav.location[10]
    xy = (nav.location - start_location)[:vicinity_cut_iter]
    head = nav.heading[:vicinity_cut_iter]
    goal = nav.goal[-1, :] - start_location

    ms = models.data
    u_mem = uyt_mem.command[:]
    y_mem_obs = uyt_mem.observation[:]
    y_mem_trg = uyt_mem.target[:]
    phases = uyt_mem.segments[:]
    if fep_signals is not None:
        y_mem_obs = fep_signals.y_estimation[:]
    ##
    y_embs_obs = get_embeddings_from_mem(y_mem_obs, phases)
    y_embs_trg = get_embeddings_from_mem(y_mem_trg, phases)

    # u_mem to embeddings
    u_embs = get_embeddings_from_mem(u_mem, phases)
    u_embs = u_embs.reshape((u_embs.shape[0], 1, u_embs.shape[1], u_embs.shape[2]))
    y_mem_preds = [m.predict(u_embs, phases[phases.shape[1]:, :]) for m in ms]
    y_mem_obs = y_mem_obs[phases.shape[1]-1:, :]
    y_mem_trg = y_mem_trg[phases.shape[1]-1:, :] #/ phases.shape[1] # normalise target as the sum of the phase
    ##
    y_emb_responses = [np.asarray([m.predict_gait_response(np.asarray([gait[0, :, :]]))[0, :, :] for gait in u_embs]) for m in ms]
    ##
    y_preds_embs = [get_embeddings_from_mem(y_mem_pred, phases[phases.shape[1]:, :]) for y_mem_pred in y_mem_preds]
    y_embs_obs = y_embs_obs[phases.shape[1]:, :]
    y_embs_trg = y_embs_trg[phases.shape[1]:, :]

    ##

    # else:
    t_axis_seg = np.arange(0, len(u_mem)) * SEGMENT_DURATION



    # plt.rcParams["figure.figsize"] = (COLUMN_WIDTH, COLUMN_WIDTH/3)
    plt.rcParams["figure.figsize"] = (COLUMN_WIDTH, COLUMN_WIDTH/3)
    f = fig()
    gs = GridSpec(3, 2, figure=f, width_ratios=[0.6, 0.35], wspace=0.03, hspace=0.25)
    axs = [[f.add_subplot(gspec)] for gspec in [gs[0, 0], gs[1, 0], gs[2, 0]]]
    """EVOLUTION"""
    # axs = V.common.subplots(f, len(sel_sensors) + 1, 1)
    for i, sns_id in enumerate(sel_sensors):
        ax = axs[i][0]
        _y_mem_trg = smooth_signal(y_mem_trg[:, sns_id])
        # clip by y_lims
        _y_mem_trg = np.clip(_y_mem_trg, y_lims[i][0], y_lims[i][1])
        ax.plot(t_axis_seg[:len(_y_mem_trg)], _y_mem_trg, label="Reference", color="r", alpha=0.9)

        _y_mem_obs = smooth_signal(np.sum(y_embs_obs[:, sns_id, :], axis=1))
        # _y_mem_obs = np.clip(_y_mem_obs, y_lims[i][0], y_lims[i][1])
        ax.plot(t_axis_seg[:len(_y_mem_obs)], _y_mem_obs, label="Estimation", color="b", alpha=0.9)

    ## servo magnitude
    # ax_magnitude = axs[-2][0]
    ax_magnitude = axs[-1][0]
    servo_magnitude = np.sqrt(np.sum(np.square(u_embs[:,0,:,:]), axis=(1,2)))
    legs_magnitude = smooth_signal(servo_magnitude)
    ax_magnitude.plot(t_axis_seg[:len(legs_magnitude)], legs_magnitude, linestyle="--", color="k", alpha=0.9)

    ax_magnitude.set_ylim(0, 6.5)


    ## distance from goal
    ax_distance = axs[-1][0]
    # ax_distance.plot(t_axis_ite, np.linalg.norm(goal - xy, axis=1), color="k", alpha=0.9)
    _loco = nav.location - start_location
    distance = np.linalg.norm(goal - _loco, axis=1)
    # _subsampled_distance = distance[::SEGMENT_ITERATIONS]
    # subsampled_distance = smooth_signal(_subsampled_distance)
    # shorten = np.maximum(len(subsampled_distance)-len(t_axis_seg), 0)
    # subsampled_distance = subsampled_distance[shorten:]
    t_axis_ite = np.arange(0, len(_loco)) * ITER_DURATION
    ax_distance.plot(t_axis_ite, distance, color="k", alpha=0.9)
    # red band at distance 0.5
    ax_distance.fill_between(t_axis_ite, 0, 0.5, color="r", alpha=0.1)


    # styling
    side_labels = [SENSORY_LABELS[sel_sensors[i]] for i in range(len(sel_sensors))] + [LEG_DISPLACEMENT_MAGNITUDE_LABEL]
    for i in range(len(axs)):
        ax = axs[i][0]
        # ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_ylabel(side_labels[i], fontsize=LABEL_SIZE)
        ax.grid()
        x_cut = min((vicinity_cut_segment-10)*SEGMENT_DURATION, np.max(t_axis_seg[:len(legs_magnitude)]))
        ax.set_xlim(0, x_cut)
        ax.set_ylim(y_lims[i])
        ## set y ticks
        # arange array of y ticks between the limits with 2.5 step
        ticks = np.arange(y_lims[i][0], y_lims[i][1]+.01, 2.5)
        ax.set_yticks([i for i in ticks])
        # every odd ticklabel is empty string
        ax.set_yticklabels([str(int(ticks[i])) if i%2 == 0 else "" for i in range(len(ticks))])
        # ax.set_yticklabels([str(i) for i in range(len(ticks))])


        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        ax.get_yaxis().set_label_coords(-0.1,0.5)

        if i < len(axs) - 1:
            # ax.set_xticks([])
            ax.set_xticklabels([])
        for ev in events:
            segment, _ = ev
            ax.axvline((segment-convolution_window/2)/phases.shape[1], color="r", linestyle="--", alpha=0.5)
    # f.suptitle("Reference evolution")
    axs[-1][0].set_xlabel(TIME_LABEL, fontsize=LABEL_SIZE)
    # axs[0][0].legend()

    # plt.savefig(os.path.join(results_path, f"{name}reference_tracking.png"), bbox_inches='tight')

    """MAP"""
    # plt.rcParams["figure.figsize"] = (COLUMN_WIDTH/3, COLUMN_WIDTH/3)
    # map_fig = fig()
    # axs = V.common.subplots(map_fig, 1, 2,gridspec_kw={'width_ratios': [0.95] + [0.05]})
    ax = f.add_subplot(gs[:, 1])
    # colorbar_ax = f.add_subplot(gs[:, 2])
    # ax = axs[0][0]
    # ax.plot(xy[:, 0], xy[:, 1], label="Location", color="k", alpha=0.9)
    # ax.plot(xy[:, 0], xy[:, 1], linestyle='', marker='o', label="Location", color="k", alpha=0.005)
    # for each event add marker with a labeled arrow
    for ev in events:
        _segment, label = ev
        segment = int((_segment / 6)/D_T)
        ax.plot(xy[segment, 0], xy[segment, 1], linestyle='', marker='x', markersize=10, label=label, color="r", alpha=0.9)
        # ax.arrow(xy[segment, 0], xy[segment, 1], np.cos(head[segment])*0.3, np.sin(head[segment])*0.3, head_width=0.1, head_length=0.1, fc="r", ec="r", alpha=0.7)


    # scatter location with the gradient coloring
    nrm_scale = np.pi/3
    nrm_head = np.clip((head + nrm_scale/2) / nrm_scale, 0, 1)
    cmap = plt.cm.get_cmap(POS_CMAP)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=[cmap(i) for i in nrm_head], alpha=1, s=4)
    # show goal with a circle of radius 0.5
    ax.add_patch(plt.Circle((goal[0], goal[1]), 0.5, color="r", alpha=0.5))
    # add large white goal marker
    ax.plot([goal[0]], [goal[1]], linestyle='', marker='o', markersize=10, label="Goal", color="w", alpha=0.9)

    # PV.ax_generate_colorbar(colorbar_ax, POS_CMAP, minmax=(-nrm_scale, nrm_scale))
    # ax.scatter(xy[:, 0], xy[:, 1], color="k", alpha=0.005)
    ## styling
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_xticks([i for i in range(0, box[1]+1)])
    ax.set_xticklabels([str(i) for i in range(0, box[1]+1)])

    ax.set_yticks([i for i in range(box[2], box[3]+1)])
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    # ax.set_xticklabels([str(i) for i in range(-5, 6)])
    ax.set_xlabel(LOCATION_LABELS[0], fontsize=LABEL_SIZE)
    ax.set_ylabel(LOCATION_LABELS[1], fontsize=LABEL_SIZE)
    ax.grid()
    #grey background
    ax.set_facecolor('grey')
    plt.savefig(os.path.join(results_path, f"{name}navigation.png"), bbox_inches='tight')

    ## standalone legends
    plt.rcParams["figure.figsize"] = (COLUMN_WIDTH*0.7, COLUMN_WIDTH/30)
    f = fig()
    gs = GridSpec(1, 2, figure=f, width_ratios=[0.5, 0.5], wspace=0.03, hspace=0.03)
    axs = [f.add_subplot(gspec) for gspec in [gs[0, 0], gs[0, 1]]]

    # standalone colorbar
    PV.ax_generate_colorbar(axs[1], POS_CMAP, minmax=(-nrm_scale, nrm_scale), orientation="horizontal")
    ax = axs[1]
    ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ##
    C.standalone_legend(fig, [
        {"color": 'r', "linestyle": '-', "alpha": 1., "label": 'Reference', "linewidth": 2},
        {"color": 'b', "linestyle": '-', "alpha": 1., "label": "Estimate", "linewidth": 2},
        {"color": 'k', "linestyle": '-', "alpha": 1., "label": "Distance", "linewidth": 2},
        {"color": 'k', "linestyle": '--', "alpha": 1., "label": "Control magnitude", "linewidth": 2}
        ],ax=axs[0], prop={'size': LABEL_SIZE})

    plt.savefig(os.path.join(results_path, f"label.png"), bbox_inches='tight')


def forward_model_derivatives(fig, results_path, name, _models: ModelsWrap):
    motor_id = 9
    sensor_id = 0
    models = _models.data
    ph_num = models[0].phase_n

    """ Models """
    derivatives = []
    for m in models:
        trg_gaits = m.u_mean
        granularity = m.phase_n
        derivative = [m.derivative_gait(trg_gaits, sensory_ph) for sensory_ph in range(granularity)]
        derivatives.append(np.asarray(derivative))


    plt.rcParams["figure.figsize"] = (COLUMN_WIDTH, COLUMN_WIDTH/3)
    map_fig = fig()
    axs = C.subplots(map_fig, 1, len(models) + 1, gridspec_kw={'width_ratios': [3] * len(models) + [.2]})
    ##
    PV.ax_generate_colorbar(axs[0][-1], NEG_POS_CMAP, minmax=(-1, 1))
    ax = axs[0][-1]
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    # PV.ax_generate_colorbar(axs[0][-1], POS_CMAP, minmax=(0, 1))
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
        ax_phph_d = axs[0][i]
        #
        ax_phph_d.matshow(phph_d[i], norm=norm, cmap=plt.cm.get_cmap(NEG_POS_CMAP))
        ax_phph_d.xaxis.set_ticks_position('bottom')
        if i == 0:
            ax_phph_d.set_ylabel("Sensory phase", size=LABEL_SIZE)
        ax_phph_d.set_yticks([i for i in range(ph_num)])
        ax_phph_d.set_yticklabels([f"c{i + 1}" for i in range(ph_num)])

        label = f"w^{{{SENSORY_LABELS[sensor_id]},c}}_{{{LEG_LABELS[motor_id // 3]}, d}}"
        ax_phph_d.text(-1, -0.8, r"${}$".format(label), size=TICK_SIZE)

        ax_phph_d.set_xlabel("Motor phase", size=LABEL_SIZE)
        ax_phph_d.set_xticks([i for i in range(ph_num)])
        ax_phph_d.set_xticklabels([f"d{i + 1}" for i in range(ph_num)])
        ax_phph_d.tick_params(axis='both', which='major', labelsize=TICK_SIZE)


        axs[0][i].set_title(MODEL_STYLES[i].label + " model", size=LABEL_SIZE)

    plt.savefig(os.path.join(results_path, f"{name}models.png"), bbox_inches='tight')

def path_map_aggregate(fig, results_path, navigations: List[
    PF.coppelia_environment.Navigation], labels: List[str], name):
    box = (-1, 5, -3, 3)
    arrow_lenght = 0.3
    skip = 100
    plt.rcParams["figure.figsize"] = (COLUMN_WIDTH/2, COLUMN_WIDTH/9)
    map_fig = fig()
    axs = C.subplots(map_fig, 1, 1)
    ax = axs[0][0]
    for i, navigation in enumerate(navigations):
        nav = navigation
        loc = nav.location[10:]
        start = loc[0]
        loc = loc - start
        ax.plot(loc[::skip, 0], loc[::skip, 1], color='k', alpha=0.4, linewidth=5)
        ## add arrows with heading
        # heading = nav.heading
        # for j in range(10, len(nav.location), len(nav.location)//10):
            # ax.arrow(nav.location[j, 0], nav.location[j, 1], np.cos(heading[j])*arrow_lenght, np.sin(heading[j])*arrow_lenght, head_width=0.1, head_length=0.1, fc=MODEL_COLORS[i], ec=MODEL_COLORS[i], alpha=0.7)
    ## styling
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_xticks([i for i in range(box[0], box[1]+1)])
    ax.set_yticks([i for i in range(box[2], box[3]+1)])
    # ax.set_xticklabels([str(i) for i in range(-5, 6)])
    ax.set_xlabel(LOCATION_LABELS[0])
    ax.set_ylabel(LOCATION_LABELS[1])
    ax.grid()
    plt.savefig(os.path.join(results_path, f"{name}navigation_aggregate.png"), bbox_inches='tight')

    """Distance aggreagate"""
    # plt.rcParams["figure.figsize"] = (PLOT_WIDTH/2, PLOT_WIDTH/2)
    plt.rcParams["figure.figsize"] = (COLUMN_WIDTH/2, COLUMN_WIDTH/6)
    goal_fig = fig()
    axs = C.subplots(goal_fig, 1, 1)
    ax = axs[0][0]
    for i, navigation in enumerate(navigations):
        nav = navigation
        xy = nav.location
        t_axis = np.arange(0, len(xy)) * ITER_DURATION

        goal = nav.goal
        ax.plot(t_axis[::skip], np.linalg.norm(goal - xy, axis=1)[::skip], color='k', alpha=0.9)
    #styling
    ax.set_xlabel(TIME_LABEL, size=LABEL_SIZE)
    ax.set_ylabel("Distance (m)", size=LABEL_SIZE)
    ax.set_xlim(0, len(xy)*ITER_DURATION)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    plt.savefig(os.path.join(results_path, f"{name}goal_distance_aggregate.png"), bbox_inches='tight')


def error_stats(pa: PortfolioAggregate[DecimJournalEvaluation], convolution_window=100):
    lifecycles = _get_ordered_lc_list(pa.lifecycle_range())
    trials = pa.trial_range()
    environments = pa.environment_range()
    scenarios = pa.scenario_range()
    env_id = environments[0]
    sc_id = scenarios[0]
    min_length = min([len(p.uyt_mem.target) for p, _ in pa.agg])

    styles = _get_lc_style_dictionary(lifecycles)
    first_tag = Tag(trial=trials[0], scenario=scenarios[0], lifecycle=lifecycles[0],
                    environment=environments[0])
    scenario_phases = pa.get(first_tag).scenario_switch.phase
    iter_len = len(pa.get(first_tag).uyt_mem.performing_stage)
    iter_seg_ratio = len(scenario_phases) // iter_len
    scenario_phases_seg = scenario_phases[::iter_seg_ratio]

    lifecycle_diff = []
    for i, lifecycle in enumerate(lifecycles):
        lifecycle_diff.append(get_performance_diffs(
            [pa.get(Tag(trial=trial, scenario=sc_id, lifecycle=lifecycle, environment=env_id)).uyt_mem for trial in
             trials], min_length))

    means, stds = PV.comparison_stats(lifecycle_diff, [styles[lc].label for lc in lifecycles], PERFORMANCE_METRICS,
                                      mean_window=convolution_window)

    lifecycle_diff = []
    for i, lifecycle in enumerate(lifecycles):
        nav_data = [pa.get(Tag(trial=trial, scenario=sc_id, lifecycle=lifecycle, environment=env_id)).navigation
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

    for pa in pas:
        mean, std = error_stats(pa, convolution_window=100)
        means.append(mean)
        stds.append(std)

    metric_labels = [met[1] for met in (PERFORMANCE_METRICS + GOAL_METRICS)]

    for i, prf in enumerate(metric_labels):
        data = np.asarray([[means[k][i][j] for k in range(len(means))] for j in range(len(means[0][i]))])
        # _argmin = np.argmin(data, axis=0)
        _argmin = np.argmax(data, axis=0)
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


def get_standalone_legends(fig, results_path):
    from matplotlib.gridspec import GridSpec
     ## standalone legends
    plt.rcParams["figure.figsize"] = (COLUMN_WIDTH*0.7, COLUMN_WIDTH/30)
    f = fig()
    gs = GridSpec(1, 2, figure=f, width_ratios=[0.5, 0.5], wspace=0.03, hspace=0.03)
    axs = [f.add_subplot(gspec) for gspec in [gs[0, 0], gs[0, 1]]]

    # standalone colorbar
    nrm_scale = np.pi/3
    PV.ax_generate_colorbar(axs[1], POS_CMAP, minmax=(-nrm_scale, nrm_scale), orientation="horizontal")
    ax = axs[1]
    ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Heading", fontsize=LABEL_SIZE)

    """xx"""
    C.standalone_legend(fig, [
        {"color": 'r', "linestyle": '-', "alpha": 1., "label": 'Reference', "linewidth": 2},
        {"color": 'b', "linestyle": '-', "alpha": 1., "label": "Estimate", "linewidth": 2},
        {"color": 'k', "linestyle": '-', "alpha": 1., "label": "Distance", "linewidth": 2},
        {"color": 'k', "linestyle": '--', "alpha": 1., "label": "Control magnitude", "linewidth": 2}
        ],ax=axs[0], prop={'size': LABEL_SIZE})

    plt.savefig(os.path.join(results_path, f"refmap_label.png"), bbox_inches='tight')

    ##
    plt.rcParams["figure.figsize"] = (COLUMN_WIDTH*0.3, COLUMN_WIDTH/30)
    f = fig()
    gs = GridSpec(1, 1, figure=f, wspace=0.03, hspace=0.03)
    axs = [f.add_subplot(gspec) for gspec in [gs[0, 0]]]
    PV.ax_generate_colorbar(axs[0], CORRELATE_CMAP, minmax=(0, 1), orientation="horizontal")
    ax = axs[0]
    ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Normalized weights", fontsize=LABEL_SIZE)

    plt.savefig(os.path.join(results_path, f"correlate_label.png"), bbox_inches='tight')


def generate_portfolio_aggregate(output_path, ps: List[LongevEvaluation],labels: List[str], name=""):
    fig = C.FigProvider()
    path_map_aggregate(fig, output_path, [p.navigation for p in ps], labels, name=name)
    fig.close_all()


def generate_portfolio_detail(output_path, pa: LongevEvaluation, name="", events: List[Tuple[str, int]] = []):
    fig = C.FigProvider()
    fep_signals = None
    if pa.fep_signals.data.segments.shape[0]>1:
        fep_signals = pa.fep_signals

    if pa.time_axis.data.iteration.shape[0] > 1:
        time_record = pa.time_axis.data
        mean_segment_duration = np.mean(time_record.timestamp[1:] - time_record.timestamp[:-1])
        mean_segment_iters = np.mean(time_record.iteration[1:] - time_record.iteration[:-1]) / 0.01
        mean_iter_duration = mean_segment_duration / mean_segment_iters
        print(f"Mean segment duration: {mean_segment_duration}s, Mean segment iters: {mean_segment_iters}, Mean iter duration: {mean_iter_duration}s")
    else:
        time_record = None

    joint_evol_map_reference_tracking(fig, output_path, name, pa.uyt_mem.data, pa.models, pa.navigation.data,
                                      fep_signals.data, events=events, time_record=time_record)
    fig.close_all()
    forward_model_derivatives(fig, output_path, name, pa.models)
    fig.close_all()



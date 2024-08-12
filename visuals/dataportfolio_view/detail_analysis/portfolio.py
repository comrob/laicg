import coupling_evol.data_process.postprocess.permanent_post_processing.factories as PF
from coupling_evol.data_process.postprocess.permanent_post_processing.common import ModelsWrap
from visuals import common as C
import visuals as V
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List
##
from coupling_evol.agent.components.internal_model.forward_model import get_embeddings_from_mem
from sklearn.decomposition import PCA

from coupling_evol.data_process.postprocess.permanent_post_processing.data_portfolios import DecimJournalEvaluation
from coupling_evol.data_process.postprocess.permanent_post_processing.helpers import Tag, parameter_parser, \
    PortfolioAggregate
from coupling_evol.engine.embedded_control import EmbeddedTargetParameter

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

MODEL_COLORS = ['darkorange', 'fuchsia', 'g', 'peru', 'y', 'darkviolet', 'orangered', 'navyblue', 'springgreen']
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


def general_control_consistency(
        fig, results_path, name, 
        uyt_mem: PF.dynamic_lifecycle.MotorSensorTargetMem,
        models: ModelsWrap,
        ensemble_dynamics: PF.model_selection.SegmentedScores
        ):
    convolution_window = 100
    epsilon = 0.5
    ms = models.data
    u_mem = uyt_mem.command
    y_mem_obs = uyt_mem.observation
    y_mem_trg = uyt_mem.target
    phases = uyt_mem.segments
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

    stage = uyt_mem.performing_stage
    d_stage = stage[1:] ^ stage[:-1]
    stage_switches = np.where(d_stage != 0)[0]
    ##
    confidence = ensemble_dynamics.second_score[phases.shape[1]:]

    def smooth_signal(signal):
        v = np.ones((convolution_window * 2,)) / convolution_window
        v[convolution_window:] = 0
        return np.convolve(signal, v=v, mode="valid")
    
    def out_epsilon(signal, _epsilon=epsilon):
        asig = np.abs(signal) 
        return ((-_epsilon > asig) + (asig > _epsilon)).astype(np.float32)
    
    sens_labels = ["head", "roll", "pitch", "yaw", "side"]

    """----------------------Raw data view"""
    # visualise y_mems into figure
    f = fig()
    f.suptitle("Sensory space")
    axs = C.subplots(f, len(sens_labels), 1)
    for i, lab in enumerate(sens_labels):
        ax = axs[i][0]
        # draw vertical line at stage switches
        for s in stage_switches:
            ax.axvline(s-convolution_window/2, color="k", linestyle="--", alpha=0.5)
        # horizontal line on zero
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        _y_mem_trg = smooth_signal(y_mem_trg[:, i])
        ax.plot(_y_mem_trg, label="Target", color="r", alpha=0.9)
        # _y_mem_obs = smooth_signal(y_mem_obs[:, i])
        _y_mem_obs = smooth_signal(np.sum(y_embs_obs[:, i, :], axis=1))
        # _y_mem_obs = np.sum(y_embs_obs[:, i, :], axis=1)
        
        ax.plot(_y_mem_obs, label="Observation", color="k", alpha=0.9)
        for j, y_mem_pred in enumerate(y_emb_responses):
            # _y_mem_pred = smooth_signal(y_mem_pred[:, i])
            _y_mem_pred = smooth_signal(np.sum(y_mem_pred[:, i, :], axis=1))
            # _y_mem_pred = np.sum(y_mem_pred[:, i, :], axis=1)
            ax.plot(_y_mem_pred, label=f"Model {j}", color=MODEL_COLORS[j], alpha=0.8)
        ax.set_ylabel(lab)
    axs[0][0].legend()
    plt.savefig(os.path.join(results_path, f"control_consistency.png"))

    """----------------------Controller optimization view"""
    d_y_mem_trg = y_mem_trg[1:] - y_mem_trg[:-1]
    e_pred = [y_mem_trg - y_mem_pred for y_mem_pred in y_mem_preds]
    e_obs = y_mem_trg - y_mem_obs

    f = fig()
    f.suptitle("Predicted performance error optimization")
    axs = C.subplots(f, len(sens_labels), 1)
    for i, lab in enumerate(sens_labels):
        ax = axs[i][0]
        # draw vertical line at stage switches
        for s in stage_switches:
            ax.axvline(s-convolution_window/2, color="k", linestyle="--", alpha=0.5)
        # horizontal line on zero
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)

        ax.plot(np.square(smooth_signal(e_obs[:, i])), label="obs", color="k", alpha=0.5)
        max_e = 0
        for j, e in enumerate(e_pred):
            _e = np.square(smooth_signal(e[:, i]))
            max_e = max(max_e, np.max(_e))
            ax.plot(_e, label=f"M {j}", color=MODEL_COLORS[j], alpha=0.8)
        _d_y_mem_trg = smooth_signal(np.square(d_y_mem_trg[:, i]))
        #scale d_y_mem_trg to match e_obs
        mx_trg = np.max(_d_y_mem_trg)
        if mx_trg > 0:
            _d_y_mem_trg = _d_y_mem_trg * (max_e / mx_trg)
        ax.plot(_d_y_mem_trg, label="d_trg", color="r", alpha=0.5)
        ax.set_ylabel(lab)
    axs[-1][0].legend()
    plt.savefig(os.path.join(results_path, f"control_optimization.png"))


    """----------------------Prediction error view"""
    e_obs_pred =  [y_mem_obs - y_mem_pred for y_mem_pred in y_mem_preds]
    f = fig()
    f.suptitle("Prediction error")
    axs = C.subplots(f, len(sens_labels), 1)
    for i, lab in enumerate(sens_labels):
        ax = axs[i][0]
        # draw vertical line at stage switches
        for s in stage_switches:
            ax.axvline(s-convolution_window/2, color="k", linestyle="--", alpha=0.5)
        # horizontal line on zero
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        for j, e in enumerate(e_obs_pred):
            _e = np.square(smooth_signal(e[:, i]))
            ax.plot(_e, label=f"M {j}", color=MODEL_COLORS[j], alpha=0.8)
        ax.set_ylabel(lab)
    axs[-1][0].legend()
    plt.savefig(os.path.join(results_path, f"prediction_error.png"))

    """----------------------Derivative Prediction error """
    d_y_embs_obs = y_embs_obs[1:, :, :] - y_embs_obs[:-1, :, :]
    d_y_embs_obs_strong = out_epsilon(d_y_embs_obs) * d_y_embs_obs
    d_y_embs_obs_magnitude = np.max(np.abs(d_y_embs_obs), axis=2) * stage[phases.shape[1]*2:, None]

    d_y_preds_embs = [y_preds_emb[1:, :, :] - y_preds_emb[:-1, :, :] for y_preds_emb in y_preds_embs]
    d_y_preds_embs_strong = [out_epsilon(d_y_preds_emb) * d_y_preds_emb for d_y_preds_emb in d_y_preds_embs]
    d_y_preds_embs_magnitude = [np.max(np.abs(d_y_preds_emb), axis=2) * stage[phases.shape[1]*2:, None] for d_y_preds_emb in d_y_preds_embs]
    d_e_obs_preds =  [np.sum(np.abs(np.sign(d_y_embs_obs_strong) - np.sign(d_y_preds_emb)), axis=2) * stage[phases.shape[1]*2:, None] for d_y_preds_emb in d_y_preds_embs_strong]
    d_e_zm_preds = np.sum(np.abs(np.sign(d_y_embs_obs_strong)), axis=2) * stage[phases.shape[1]*2:, None]
    filtered_confidence = smooth_signal(confidence[phases.shape[1]:] * stage[phases.shape[1]*2:])

    diff_obs_trg = (y_embs_obs - y_embs_trg)[1:, :, :]
    pred_towards_target = [np.mean(np.abs(np.sign(d_y_preds_emb) - np.sign(diff_obs_trg)), axis=2) * stage[phases.shape[1]*2:, None] for d_y_preds_emb in d_y_preds_embs_strong]

    f = fig()
    axs = C.subplots(f, len(sens_labels), 2)
    for i, lab in enumerate(sens_labels):
        ax = axs[i][0]
        # draw vertical line at stage switches
        for s in stage_switches:
            ax.axvline(s-convolution_window*2, color="k", linestyle="--", alpha=0.5)
        # horizontal line on zero
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        max_e = 0
        for j, e in enumerate(d_e_obs_preds):
            _e = smooth_signal(d_e_zm_preds[:, i] - e[:, i])
            max_e = max(max_e, np.max(_e))
            ax.plot(_e, label=f"dez - deM {j}", color=MODEL_COLORS[j], alpha=0.8)
        ax.set_ylabel(lab)
        _cnf = filtered_confidence * (max_e / np.max(filtered_confidence))
        ax.plot(_cnf, label="cnf", color="b", alpha=0.8)
        # ax.plot(smooth_signal(d_e_zm_preds[:, i]), label="zm", color="k", alpha=0.5)

        """Magnitude"""
        ax = axs[i][1]
        # derivative magnitude
        # red horizontal line on epsilon
        ax.axhline(epsilon/5, color="r", linestyle="--", alpha=0.5)
        ax.plot(smooth_signal(d_y_embs_obs_magnitude[:, i]), label="|obs|", color="k", alpha=0.5)
        for j, e in enumerate(d_y_preds_embs_magnitude):
            _e = smooth_signal(e[:, i])
            ax.plot(_e, label=f"|dM {j}|", color=MODEL_COLORS[j], alpha=0.8)
        # """Towards target"""
        # ax = axs[i][2]
        # # draw vertical line at stage switches
        # for s in stage_switches:
        #     ax.axvline(s-convolution_window*2, color="k", linestyle="--", alpha=0.5)
        # # horizontal line on zero
        # ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        # for j, e in enumerate(pred_towards_target):
        #     _e = smooth_signal(e[:, i])
        #     ax.plot(_e, label=f"tM {j}", color=MODEL_COLORS[j], alpha=0.8)
    axs[-1][0].legend()
    axs[-1][1].legend()
    axs[0][0].set_title("(Zero - Model) derivative error")
    axs[0][1].set_title("Derivative magnitude")
    plt.savefig(os.path.join(results_path, f"derivative_prediction_error.png"))

    """Absolute co-direction"""
    e_predicted_base_change = [(y_preds_emb - ms[i].y_mean[None, :, :])/ms[i].y_std[None, :, :] for i, y_preds_emb in enumerate(y_preds_embs)]
    e_pred_error_magnitude = [np.mean((y_preds_emb - y_embs_obs)/ms[i].y_std[None, :, :], axis=2) for i, y_preds_emb in enumerate(y_preds_embs)]
    e_true_base_change = [(y_embs_obs - m.y_mean[None, :, :])/m.y_std[None, :, :] for m in ms]

    ##
    e_pred_base_out_eps = [out_epsilon(e_pred_error_magnitude[i], _epsilon=epsilon)[:, :, None] * e_predicted_base_change[i] for i in range(len(ms))]
    e_true_base_out_eps = [out_epsilon(e_pred_error_magnitude[i], _epsilon=epsilon)[:, :, None] * e_true_base_change[i] for i in range(len(ms))]

    codirectional = [np.sum(np.abs(np.sign(e_pred_base_out_eps[i]) - np.sign(e_true_base_out_eps[i])), axis=2) for i in range(len(ms))]
    
    f = fig()
    axs = C.subplots(f, len(sens_labels), 2)
    for i, lab in enumerate(sens_labels):
        """Directional error"""
        ax = axs[i][0]
        # draw vertical line at stage switches
        for s in stage_switches:
            ax.axvline(s-convolution_window*2, color="k", linestyle="--", alpha=0.5)
        # horizontal line on zero
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        for j, e in enumerate(codirectional):
            _e = smooth_signal(e[:, i])
            ax.plot(_e, label=f"codir M {j}", color=MODEL_COLORS[j], alpha=0.8)
        ax.set_ylabel(lab)
        """Magnitude"""
        ax = axs[i][1]
        # draw vertical line at stage switches
        for s in stage_switches:
            ax.axvline(s-convolution_window*2, color="k", linestyle="--", alpha=0.5)
        # horizontal line on zero
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.axhline(epsilon, color='r', linestyle="--", alpha=0.5)
        for j, e in enumerate(e_pred_error_magnitude):
            _e = smooth_signal(e[:, i])
            ax.plot(np.abs(_e), label=f"pred {j}", color=MODEL_COLORS[j], alpha=0.8)
    
    axs[-1][0].legend()
    axs[-1][1].legend()
    axs[0][0].set_title("Co-directional error")
    axs[0][1].set_title("Prediction error magnitude")
    plt.savefig(os.path.join(results_path, f"absolute_co_directional.png"))

def internal_model_state_analysis(fig, results_path, name, uyt_mem: PF.dynamic_lifecycle.MotorSensorTargetMem,
                                  models: ModelsWrap,
                                  ensemble_dynamics: PF.model_selection.SegmentedScores):
    sfmx_pow = 1
    convolution_window = 50
    ms = models.data
    u_mem = uyt_mem.command
    y_mem_obs = uyt_mem.observation
    phases = uyt_mem.segments
    confidence = ensemble_dynamics.second_score[phases.shape[1]:]
    negentr = ensemble_dynamics.first_score[phases.shape[1]:]

    y_mem_trg = uyt_mem.target
    
    ##
    y_embs_obs = get_embeddings_from_mem(y_mem_obs, phases)
    u_embs = get_embeddings_from_mem(u_mem, phases)
    y_embs_trg = get_embeddings_from_mem(y_mem_trg, phases)

    # u_embs = u_embs.reshape((u_embs.shape[0], 1, u_embs.shape[1], u_embs.shape[2]))

    u_base = np.asarray([m.u_mean for m in ms])
    y_base = np.asarray([m.y_mean for m in ms])

    stage = uyt_mem.performing_stage
    d_stage = stage[1:] ^ stage[:-1]
    stage_switches = np.where(d_stage != 0)[0]
    ##

    def smooth_signal(signal):
        v = np.ones((convolution_window * 2,)) / convolution_window
        v[convolution_window:] = 0
        # return np.convolve(signal, v=v, mode="valid")
        return signal
    
    plt.rcParams["figure.figsize"] = (10, 5)

    """Negative entropy, u_base distance, y_base distance evolutions"""
    f = fig()
    f.suptitle("Internal model state analysis")
    axs = C.subplots(f, 3, 2)
    ## u_base distance
    for i, mid in enumerate(ms):
        ax = axs[0][0]
        ax.plot(smooth_signal(np.sum(np.square(u_embs - u_base[i]), axis=(1,2))), label=f"M{i}", color=MODEL_COLORS[i], alpha=0.9)
    ## y_base distance
    ax = axs[1][0]
    # y_trg evol
    ax.plot(smooth_signal(np.sum(np.square(y_embs_trg), axis=(1,2))), label=f"trg", color="k", alpha=0.9)
    for i, mid in enumerate(ms):
        ax.plot(smooth_signal(np.sum(np.square(y_embs_obs - y_base[i]), axis=(1,2))), label=f"M{i}", color=MODEL_COLORS[i], alpha=0.9)
    ## Negative entropy
    # softmax normalisation of negentr
    exp_negetr = np.exp(negentr * sfmx_pow)
    _negentr = exp_negetr / np.sum(exp_negetr, axis=1)[:, None]
    for i in range(_negentr.shape[1]):
        ax = axs[2][0]
        ax.plot(smooth_signal(_negentr[:, i]), label=f"M{i}", color=MODEL_COLORS[i], alpha=0.9)
    
    time_grad = np.arange(_negentr.shape[0])/_negentr.shape[0]
    axs[2][0].scatter(np.arange(_negentr.shape[0]), -np.ones((_negentr.shape[0],)) * 0.1, c=time_grad, alpha=0.01)
    
    # axs[2][0].legend()
    for i, lab in enumerate(["u_base distance", "y_base distance","Negative entropy"]):
        ax = axs[i][0]
        # draw vertical line at stage switches
        for s in stage_switches:
            ax.axvline(s-convolution_window/2, color="k", linestyle="--", alpha=0.5)
        # horizontal line on zero
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_ylabel(lab) 
    
    """PCA of u_base and u_embs"""
    pca = PCA(n_components=2)
    u_embs_m = np.asarray([m.u_mean.flatten() for m in ms] + [tre.flatten() for tre in u_embs])
    pca.fit(u_embs_m)
    u_pca = pca.transform([tre.flatten() for tre in u_embs])
    u_base_pca = pca.transform([m.u_mean.flatten() for m in ms])
    ax = axs[0][1]
    #scatter u_pca with gradient color evolving in time
    time_grad = np.arange(u_pca.shape[0])/u_pca.shape[0]
    ax.scatter(u_pca[:, 0], u_pca[:, 1], label="u_embs", c=time_grad, alpha=0.01)
    for i, mid in enumerate(ms):
        ax.text(u_base_pca[i, 0], u_base_pca[i, 1], f"M{i}", color=MODEL_COLORS[i], alpha=1)
        ax.scatter(u_base_pca[i, 0], u_base_pca[i, 1], color=MODEL_COLORS[i], alpha=1., marker='x')
    
    """PCA of y_base and y_embs"""
    pca = PCA(n_components=2)
    y_embs_m = np.asarray([m.y_mean.flatten() for m in ms] + [tre.flatten() for tre in y_embs_obs])
    pca.fit(y_embs_m)
    y_pca = pca.transform([tre.flatten() for tre in y_embs_obs])
    y_base_pca = pca.transform([m.y_mean.flatten() for m in ms])
    y_trg_pca = pca.transform([tre.flatten() for tre in y_embs_trg])
    ax = axs[1][1]
    #scatter y_pca with gradient color evolving in time
    time_grad = np.arange(y_pca.shape[0])/y_pca.shape[0]
    ax.scatter(y_pca[:, 0], y_pca[:, 1], c=time_grad, alpha=0.01)
    ax.scatter(y_trg_pca[:, 0], y_trg_pca[:, 1], c='k', alpha=0.1)
    for i, mid in enumerate(ms):
        ax.text(y_base_pca[i, 0], y_base_pca[i, 1], f"M{i}", color=MODEL_COLORS[i], alpha=0.9)
        ax.scatter(y_base_pca[i, 0], y_base_pca[i, 1], color=MODEL_COLORS[i], alpha=1., marker='x')


    """yu_distance negentropy correlation"""
    ax = axs[2][1]
    for i in range(_negentr.shape[1]):
        prior_dist = np.sum(np.square(y_embs_obs - y_base[i]), axis=(1,2)) + np.sum(np.square(u_embs - u_base[i]), axis=(1,2))
        ax.scatter(smooth_signal(prior_dist[1:]), smooth_signal(_negentr[:, i]), label=f"M{i}", color=MODEL_COLORS[i], alpha=0.1)
    
    plt.savefig(os.path.join(results_path, f"internal_model_state.png"))



def generate_portfolio(output_path, pa: DecimJournalEvaluation, name=None):
    fig = C.FigProvider()
    general_control_consistency(fig, output_path, name, pa.uyt_mem.data, pa.models, pa.segmented_model_selection.data)
    internal_model_state_analysis(fig, output_path, name, pa.uyt_mem.data, pa.models, pa.segmented_model_selection.data)
    # fig.show()
    fig.close_all()



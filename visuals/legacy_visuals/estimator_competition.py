from visuals import common as C
from typing import List
import numpy as np
from agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation
import matplotlib as plt


def competition_signals_evol(fig: C.plt.Figure, model_selections: np.ndarray,
                             inter_model_score: np.ndarray,
                             best_zero_score: np.ndarray, title):
    fig.suptitle(title)
    figs = C.subplots(fig, 3, 1)
    fig_st1 = figs[0][0]
    fig_st2 = figs[1][0]
    fig_decision = figs[2][0]
    ##
    model_n = inter_model_score.shape[1]
    colors = C.colors(model_n)

    fig_decision.plot(model_selections + 1)
    for i in range(model_n):
        fig_decision.axhline(y=i + 1, color=colors[i], linestyle='--')

    for i in range(model_n):
        fig_st1.plot(inter_model_score[:, i], color=colors[i], alpha=0.5, label=f"{i + 1}")
    fig_st1.legend()

    fig_st2.plot(best_zero_score)
    C.subplots_styling(
        figs, inner_x_ticks_off=False, xlabels=None,
        ylabels=["model logodds", "zero x best", "decision"]
    )


def competition_overview_evol(fig: C.plt.Figure, model_selections: np.ndarray,
                              inter_model_score: np.ndarray,
                              best_zero_score: np.ndarray,
                              is_dataset_driven,
                              stages,
                              performance_error,
                              title):
    fig.suptitle(title)
    figs = C.subplots(fig, 3, 1)
    fig_st1 = figs[0][0]
    fig_st2 = figs[1][0]
    fig_decision = figs[2][0]
    ##
    model_n = inter_model_score.shape[1]
    colors = C.colors(model_n)

    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)

    is_prior = 1 - is_dataset_driven
    chng = is_prior[1:] - is_prior[:-1]
    prior_st = [0] + [i for i in range(len(chng)) if chng[i] == 1]
    prior_en = [i for i in range(len(chng)) if chng[i] == -1]
    performance_error_max = np.max(performance_error)
    if len(prior_en) < len(prior_st):
        prior_en.append(len(chng))

    for st, en in bbls:
        fig_st1.axvspan(xmin=st, xmax=en, alpha=0.2, color='k')
        fig_st2.axvspan(xmin=st, xmax=en, alpha=0.2, color='k')
        fig_decision.axvspan(xmin=st, xmax=en, alpha=0.2, color='k')

    for st, en in prfs:
        fig_st1.axvspan(xmin=st, xmax=en, alpha=0.2, color='r')
        fig_st2.axvspan(xmin=st, xmax=en, alpha=0.2, color='r')
        fig_decision.axvspan(xmin=st, xmax=en, alpha=0.2, color='r')

    for i, st in enumerate(prior_st):
        fig_st1.axvspan(xmin=st, xmax=prior_en[i], alpha=0.4, color='r')
        fig_st2.axvspan(xmin=st, xmax=prior_en[i], alpha=0.4, color='r')
        fig_decision.axvspan(xmin=st, xmax=prior_en[i], alpha=0.4, color='r')

    fig_decision.plot(model_selections + 1)
    for i in range(model_n):
        fig_decision.axhline(y=i + 1, color=colors[i], linestyle='--')
    fig_decision.axhline(y=0, color='k', linestyle='-', alpha=0.8)
    fig_decision.set_yticks([i for i in range(model_n + 1)])
    fig_decision.set_yticklabels([str(i) for i in range(model_n + 1)])

    _ims = np.clip(inter_model_score, a_min=-100, a_max=1000)
    exps = np.exp(_ims)
    w = exps / np.sum(exps, axis=1)[:, None]

    for i in range(model_n):
        row_w = i * 1.1
        fig_st1.axhline(y=row_w - 0.5, color='k', linestyle='--', alpha=0.4)
        fig_st1.axhline(y=row_w + 0.5, color='k', linestyle='--', alpha=0.4)
        fig_st1.plot(w[:, i] - 0.5 + row_w, color=colors[i], alpha=0.9, label=f"{i + 1}")
    fig_st1.set_yticks([i * 1.1 for i in range(model_n)])
    fig_st1.set_yticklabels([str(i + 1) for i in range(model_n)])

    fig_st2.plot(best_zero_score)
    fig_st2.plot(performance_error / performance_error_max, 'k', alpha=0.2)

    C.subplots_styling(
        figs, inner_x_ticks_off=False, xlabels=None, legend_on=False,
        ylabels=["P(model)", "zero x best,p_e", "decision"]
    )


def sensor_wise_model_competition(
        fig: C.plt.Figure,
        model_logodds: np.ndarray,
        stages,
        title):
    fig.suptitle(title)
    n, model_n, sens_dim, ph_dim = model_logodds.shape
    figs = C.subplots(fig, sens_dim, 1)
    colors = C.colors(model_n)
    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    ##
    best_sens_mods = np.argmax(model_logodds, axis=1)
    ##
    for s in range(sens_dim):
        _fig = figs[s][0]
        for prf in prfs:
            _fig.axvline(x=prf[0], color='k', linestyle='--')
            _fig.axvline(x=prf[1], color='k', linestyle='-')
        for p in range(ph_dim):
            _fig.scatter(x=[i for i in range(n)], y=[p for i in range(n)],
                         c=[colors[best_sens_mods[i, s, p]] for i in range(n)])


def get_change_to_zero(model_selections: np.ndarray):
    x = np.zeros((len(model_selections),))
    x[model_selections == -1] = 1
    return np.where((x[1:] - x[:-1]) == 1)[0] + 1


def zero_best_log_odds_mat(fig: C.plt.Figure,
                           best_model_entropies: np.ndarray, zero_model_entropies: np.ndarray,
                           model_selections: np.ndarray,
                           selection=(0, -1), title=""):
    fig.suptitle(title)
    figs = C.subplots(fig, 1, len(selection))
    logodds = best_model_entropies - zero_model_entropies
    for i, s in enumerate(selection):
        _f = figs[0][i]
        _f.set_title(f"{model_selections[s]}[{s}]")
        _f.matshow(logodds[s])


def d_uy_std_evol(fig: C.plt.Figure, d_uy_std: np.ndarray, stages: np.ndarray,
                  zero_std: np.ndarray,
                  title="", use_log=False):
    fig.suptitle(title)
    sens_dim = d_uy_std.shape[2]
    mod_n = d_uy_std.shape[1]
    colors = C.colors(mod_n)
    figs = C.subplots(fig, sens_dim, 1)
    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)
    if use_log:
        d_uy_std_phavg = -np.max(np.log(d_uy_std), axis=3)
        zero_std_phavg = -np.max(np.log(zero_std), axis=2)
    else:
        d_uy_std_phavg = np.average(d_uy_std, axis=3)
        zero_std_phavg = np.average(zero_std, axis=2)

    for i in range(sens_dim):
        _f = figs[i][0]
        # _f.plot(zero_std_phavg[:, i], color='k', label="zero", alpha=0.8)
        for m in range(mod_n):
            _f.plot(np.clip(d_uy_std_phavg[:, m, i] - zero_std_phavg[:, i], a_max=0.5, a_min=-0.5), color=colors[m], label=f"m{m}", alpha=0.8)
            _f.axhline(y=0, linestyle="--", color="k", alpha=0.5)
        for st, en in bbls:
            _f.axvspan(xmin=st, xmax=en, alpha=0.2, color='k')
        for st, en in prfs:
            _f.axvspan(xmin=st, xmax=en, alpha=0.2, color='r')
        if use_log:
            _f.set_ylabel(f"E_std_s{i}")
        else:
            _f.set_ylabel(f"std_s{i}")
        _f.set_ylim(ymax=.6, ymin=-.6)
    figs[0][0].legend()


def best_v_zero_entropy_evol(fig: C.plt.Figure, best_model_entropy: np.ndarray, zero_model_entropy: np.ndarray,
                             stages: np.ndarray, title=""):
    fig.suptitle(title)
    sens_dim = best_model_entropy.shape[1]
    ph_dim = best_model_entropy.shape[2]
    colors = C.colors(ph_dim, "Set1")
    figs = C.subplots(fig, sens_dim, 1)
    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)

    log_odds = best_model_entropy - zero_model_entropy

    for i in range(sens_dim):
        _f = figs[i][0]
        for st, en in bbls:
            _f.axvspan(xmin=st, xmax=en, alpha=0.2, color='k')
        # for st, en in prfs:
        #     _f.axvspan(xmin=st, xmax=en, alpha=0.2, color='r')
        for m in range(ph_dim):
            _f.plot(log_odds[:, i, m], color=colors[m], label=f"ph{m}", alpha=0.8)
        _f.axhline(y=0, color='k', linestyle='--', alpha=0.4)
        _f.set_ylabel(f"s{i}")
    figs[0][0].legend()


def best_zero_logodds_evol(
        fig: C.plt.Figure, best_zero_logodds: np.ndarray,
        stages: np.ndarray, title="", convolution_window=1000):
    fig.suptitle(title)
    sens_dim = best_zero_logodds.shape[1]
    figs = C.subplots(fig, sens_dim, 1)
    bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)
    bz_logodds = np.average(best_zero_logodds, axis=2)

    for i in range(sens_dim):
        _f = figs[i][0]
        cnv_logodds = np.convolve(bz_logodds[:, i], v=np.ones((convolution_window,)) / convolution_window)
        for st, en in bbls:
            _f.axvspan(xmin=st, xmax=en, alpha=0.2, color='k')
        _f.plot(cnv_logodds, 'r', alpha=1.)
        _f.axhline(y=0, color='k', linestyle='--', alpha=0.4)
        _f.set_ylabel(f"s{i}")
    figs[0][0].legend()


def best_zero_logodds_evol_mat(
        fig: C.plt.Figure, best_zero_logodds: np.ndarray,
        stages: np.ndarray, title="", convolution_window=1000, cut=(0, -1), skip=100):
    fig.suptitle(title)
    sens_dim = best_zero_logodds.shape[1]
    figs = C.subplots(fig, 1, 1)
    _f = figs[0][0]
    bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)
    bz_logodds = np.average(best_zero_logodds, axis=2)
    best_win = (bz_logodds >= 0).astype(int)
    zero_win = (bz_logodds < 0).astype(int)
    x = best_win - zero_win
    cnvs = []
    for i in range(sens_dim):
        cnvs.append(np.convolve(x[:, i], v=np.ones((convolution_window,)) / convolution_window))
    mat = np.asarray(cnvs)[:, cut[0]:cut[1]]

    _f.matshow(mat[:, ::skip], cmap='coolwarm', norm=plt.colors.Normalize(vmin=-1, vmax=1))


def best_zero_logodds_evol_mat_iros23(
        fig: C.plt.Axes, best_zero_logodds: np.ndarray,
        convolution_window=1000, cut=(0, -1), skip=100):
    # sens_dim = best_zero_logodds.shape[1]
    _f = fig
    bz_logodds = np.average(best_zero_logodds, axis=2)
    best_win = (bz_logodds >= 0).astype(int)
    _x = best_win
    x = np.zeros((len(_x), 6))
    x[:, :5] = _x[:, :5]
    x[:, 5] = np.average(_x[:, 5:], axis=1)
    cnvs = []
    for i in range(6):
        cnvs.append(np.convolve(x[:, i], v=np.ones((convolution_window,)) / convolution_window))
    mat = np.asarray(cnvs)[:, cut[0]:cut[1]]
    return _f.matshow(mat[:, ::skip], cmap='coolwarm', norm=plt.colors.Normalize(vmin=0, vmax=1))


def modality_overview(fig: C.plt.Axes, confidence_aggregate: np.ndarray, colors: List,
                      ts: np.ndarray = None
                      ):
    # sens_dim = best_zero_logodds.shape[1]
    tn, cn = confidence_aggregate.shape
    # maxes = np.max(np.abs(confidence_aggregate), axis=0)
    # normed_conf_agg = confidence_aggregate / (maxes + 0.0001)[None, :]
    maxes = np.max(np.abs(confidence_aggregate))
    normed_conf_agg = confidence_aggregate / (maxes + 0.0001)
    normed_conf_agg = (normed_conf_agg + 1) / 2
    is_confident_element = confidence_aggregate >= 0
    is_confident_modality = np.mean(is_confident_element, axis=1)

    if ts is None:
        iters = np.arange(len(is_confident_element), )
    else:
        iters = ts

    fig.plot(iters, is_confident_modality, 'k', label="avg(conf)")
    fig.set_ylim(-0.1, 1.1)
    fig.axhline(y=0.5, color='r', linestyle='--')

    for c in range(cn):
        fig.plot(iters, normed_conf_agg[:, c], color=colors[c], alpha=0.6, label=f"ph{c}")


def bz_competition_sum(fig: C.plt.Axes, confidence_aggregate: np.ndarray, target_weights: np.ndarray,
                       weighted_score: np.ndarray, colors: List, labels: List, aggregate_beyond=None,
                       ts: np.ndarray = None
                       ):
    is_confident_element = confidence_aggregate >= 0
    weights_sum = np.sum(target_weights, axis=(1, 2))
    elem_contrib = is_confident_element * target_weights / weights_sum[:, None, None]
    nrm_weights = target_weights/weights_sum[:, None, None]

    nrm_weight_contrib = np.sum(nrm_weights, axis=2)
    modality_contrib = np.sum(elem_contrib, axis=2)

    if aggregate_beyond is not None:
        _modality_contrib = np.zeros((len(modality_contrib), aggregate_beyond + 1))
        _modality_contrib[:, :aggregate_beyond] = modality_contrib[:, :aggregate_beyond]
        _modality_contrib[:, aggregate_beyond] = np.sum(modality_contrib[:, aggregate_beyond:], axis=1)
        modality_contrib = _modality_contrib

        _nrm_weight_contrib = np.zeros((len(modality_contrib), aggregate_beyond + 1))
        _nrm_weight_contrib[:, :aggregate_beyond] = nrm_weight_contrib[:, :aggregate_beyond]
        _nrm_weight_contrib[:, aggregate_beyond] = np.sum(nrm_weight_contrib[:, aggregate_beyond:], axis=1)
        nrm_weight_contrib = _nrm_weight_contrib

    if ts is None:
        iters = np.arange(len(is_confident_element), )
    else:
        iters = ts
    for y in range(modality_contrib.shape[1]):
        if y > 0:
            lower_bound = np.sum(nrm_weight_contrib[:,:y], axis=1)
        else:
            lower_bound = 0
        fig.fill_between(iters, y1=lower_bound, y2=modality_contrib[:, y] + lower_bound, color=colors[y], alpha=0.5)
        fig.plot(iters, modality_contrib[:, y] + lower_bound, color=colors[y], alpha=0.8, label=labels[y])
    fig.plot(iters, weighted_score, 'k', label="bz_score")


def best_zero_competition_hexapod_odom_all(
        fig: C.plt.Figure, confidence_aggregate: np.ndarray, stages: np.ndarray, title):
    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)
    tn, mn, cn = confidence_aggregate.shape
    fig.suptitle(title)
    figs = C.subplots(fig, 5, 1)
    labels = ["fwd_v", "roll", "pitch", "yaw", "sid_v", "eff"]
    ph_colors = C.colors(cn, cmap='Dark2')
    for i in range(5):
        f = figs[i][0]
        modality_overview(f, confidence_aggregate[:, i, :], colors=ph_colors)
        f.set_ylabel(labels[i])
        for st, en in bbls:
            f.axvspan(xmin=st, xmax=en, alpha=0.2, color='k')
        for st, en in prfs:
            f.axvspan(xmin=st, xmax=en, alpha=0.2, color='r')
    figs[0][0].legend()


def best_zero_competition_hexapod_eff_all(
        fig: C.plt.Figure, confidence_aggregate: np.ndarray, stages: np.ndarray, title):
    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)
    tn, mn, cn = confidence_aggregate.shape
    fig.suptitle(title)
    figs = C.subplots(fig, 18, 1)
    ph_colors = C.colors(cn, cmap='Dark2')
    for i in range(18):
        f = figs[i][0]
        modality_overview(f, confidence_aggregate[:, i + 5, :], colors=ph_colors)
        f.set_ylabel(f"eff{i}")
        for st, en in bbls:
            f.axvspan(xmin=st, xmax=en, alpha=0.2, color='k')
        for st, en in prfs:
            f.axvspan(xmin=st, xmax=en, alpha=0.2, color='r')
    figs[0][0].legend()


def best_zero_competition_contributions_all(
        fig: C.plt.Figure,
        confidence_aggregate: np.ndarray, target_weights: np.ndarray,
        weighted_score: np.ndarray, stages: np.ndarray, title):

    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)

    fig.suptitle(title)
    _f = C.subplots(fig, 1, 1)
    fs = _f[0][0]

    for st, en in bbls:
        fs.axvspan(xmin=st, xmax=en, alpha=0.2, color='k')
    for st, en in prfs:
        fs.axvspan(xmin=st, xmax=en, alpha=0.2, color='r')

    labels = ["fwd_v", "roll", "pitch", "yaw", "sid_v", "eff"]
    m_colors = C.colors(6, cmap='jet')
    bz_competition_sum(fs, confidence_aggregate, target_weights, weighted_score, colors=m_colors,
                       labels=labels,
                       aggregate_beyond=5)
    fs.legend()
    fs.set_ylabel("weighted score")


def best_zero_competition_contributions_detail(
        fig: C.plt.Figure,
        confidence_aggregate: np.ndarray, target_weights: np.ndarray,
        weighted_score: np.ndarray, stages: np.ndarray, title, cut_before=500):

    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)

    fig.suptitle(title)
    _f = C.subplots(fig, 1, len(prfs))

    iters = np.arange(len(confidence_aggregate), )
    for i, sten in enumerate(prfs):
        st, en = sten
        s, e = (np.maximum(en - cut_before, 0), en)
        fs = _f[0][i]

        labels = ["fwd_v", "roll", "pitch", "yaw", "sid_v", "eff"]
        m_colors = C.colors(6, cmap='jet')
        bz_competition_sum(fs, confidence_aggregate[s:e], target_weights[s:e], weighted_score[s:e],
                           colors=m_colors,
                           labels=labels,
                           aggregate_beyond=5,
                           ts=iters[s:e]
                           )
        fs.set_title(f"ctx{i}")
    _f[0][0].legend()
    _f[0][0].set_ylabel("weighted score")


def best_zero_competition_hexapod_odom_detail(
        fig: C.plt.Figure, confidence_aggregate: np.ndarray, stages: np.ndarray, title,
        cut_before=500
):
    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    # bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)
    tn, mn, cn = confidence_aggregate.shape
    fig.suptitle(title)
    figs = C.subplots(fig, 5, len(prfs))
    labels = ["fwd_v", "roll", "pitch", "yaw", "sid_v", "eff"]
    ph_colors = C.colors(cn, cmap='Dark2')

    iters = np.arange(len(confidence_aggregate), )
    for j, sten in enumerate(prfs):
        st, en = sten
        s, e = (np.maximum(en - cut_before, 0), en)
        figs[0][j].set_title(f"ctx{j}")
        for i in range(5):
            f = figs[i][j]
            modality_overview(f, confidence_aggregate[s:e, i, :],
                              ts=iters[s:e],
                              colors=ph_colors)
            if j == 0:
                f.set_ylabel(labels[i])
    figs[0][-1].legend()


def best_zero_competition_hexapod_eff_detail(
        fig: C.plt.Figure, confidence_aggregate: np.ndarray, stages: np.ndarray, title,
        cut_before=500
):
    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    tn, mn, cn = confidence_aggregate.shape
    fig.suptitle(title)
    figs = C.subplots(fig, 18, len(prfs))
    ph_colors = C.colors(cn, cmap='Dark2')
    iters = np.arange(len(confidence_aggregate), )
    for j, sten in enumerate(prfs):
        st, en = sten
        s, e = (np.maximum(en - cut_before, 0), en)
        figs[0][j].set_title(f"ctx{j}")
        for i in range(18):
            f = figs[i][j]
            modality_overview(f, confidence_aggregate[s:e, i + 5, :],
                              ts=iters[s:e],
                              colors=ph_colors)
            if j == 0:
                f.set_ylabel(f"e{i}")

    figs[0][-1].legend()

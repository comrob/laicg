import numpy as np

from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation
from visuals import common as C
from typing import List, Tuple
import matplotlib as mpl


def fill_boolean(ax: C.plt.Axes, is_fill: np.ndarray, y1, y2, dt=1., **kwargs):
    vals = np.arange(len(is_fill))[is_fill] * dt
    ax.fill_between(vals, y1=y1, y2=y2, **kwargs)


def _filter_nonperforming(mem, performing, gait_n, avg_window=10):
    avg_mem = np.mean(mem[:avg_window], axis=0)
    was_performing = False
    ret = np.zeros(mem.shape)
    segments_rest = 0
    for i in range(len(mem)):
        if performing[i]:
            if not was_performing:
                segments_rest = i
            if i - segments_rest < gait_n * 3:
                ret[i] = avg_mem
            else:
                ret[i] = mem[i]
        else:
            if was_performing:
                avg_mem = np.mean(mem[i - avg_window - 2:i - 2], axis=0)
                ret[i - 1] = avg_mem
            ret[i] = avg_mem
        was_performing = performing[i]
    return ret


def comparison(figs: List[Tuple[C.plt.Axes, C.plt.Axes]], diffs: List[List[np.ndarray]], method_labels, colors,
               scenario_phases: np.ndarray, metrics, ts: np.ndarray = None, mean_window=100, time_label=None,
               legend_on=True, has_first=True, has_last=True
               ):
    """

    @param fig:
    @param diffs: [alg[trial]]
    @param title:
    @param method_labels: alg[label]
    @return:
    @rtype:
    """
    if ts is None:
        ts = np.arange(diffs[0][0].shape[0])

    for i, metric_lab in enumerate(metrics):
        metric, metric_label = metric_lab
        fig_evol = figs[i][0]
        fig_bp = figs[i][1]
        alg_means = []
        ylim = (np.inf, -np.inf)
        for j in range(len(diffs)):
            errs = np.asarray([metric(trial) for k, trial in enumerate(diffs[j])])
            err_means = np.mean(errs, axis=0)
            # err_min = np.min(errs, axis=0)
            # err_max = np.max(errs, axis=0)
            if i == 0:
                _label = method_labels[j]
            else:
                _label = None
            # fig_evol.plot(err_min, '--', color=colors[j], alpha=0.12)
            # fig_evol.plot(err_max, '--', color=colors[j], alpha=0.12)
            # fig_evol.fill_between(np.arange(len(err_means), ), y1=err_min, y2=err_max, color=colors[j], alpha=0.1)
            offset = len(ts) - len(err_means)
            fig_evol.plot(ts[offset:], err_means, color=colors[j], label=_label, alpha=0.8)
            alg_means.append(np.mean(errs[:, -mean_window:], axis=1))
            # figs[i][0].axvline(x=float(ts[-mean_window]), linestyle='--', color='r')
            ylim = (np.minimum(np.min(errs), ylim[0]), np.maximum(np.max(errs), ylim[1]))
        ylim = (ylim[0], ylim[1] * 1.1)
        fill_boolean(fig_evol, scenario_phases[offset:] == 1., y1=ylim[0], y2=ylim[1],
                     dt=float(ts[1] - ts[0]),
                     color='y', alpha=0.5)
        fig_bp.boxplot(alg_means)
        fig_evol.set_ylabel(metric_label)
        fig_evol.set_ylim(*ylim)
        # fig_bp.set_ylim(*ylim)
    if has_first:
        figs[0][0].set_title("Evolution")
        figs[0][1].set_title("Final statistics")

    if has_last:
        figs[-1][0].set_xlabel(xlabel=time_label)
        [figs[i][0].set_xticks([]) for i in range(len(metrics) - 1)]
    else:
        [figs[i][0].set_xticks([]) for i in range(len(metrics))]
    for i in range(len(metrics)):
        figs[i][1].set_xticks([i + 1 for i in range(len(diffs))])
        figs[i][1].set_xticklabels([method_labels[i] for i in range(len(diffs))])
    if legend_on:
        figs[0][0].legend()


def draw_staged_path(ax: C.plt.Axes, xy_translation: np.ndarray,
                     yaw: np.ndarray,
                     stages: np.ndarray,
                     babbling_style={"color": 'r', "linestyle": '-', "alpha": 0.8},
                     performing_style={"color": 'b', "linestyle": '-', "alpha": 0.8},
                     babbling_label="babbling",
                     performing_label="performing",
                     vector_draw_granularity=2000,
                     show_yaw=False):
    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)

    for i, st_en in enumerate(bbls):
        st, en = st_en
        if i == 0:
            label = babbling_label
        else:
            label = None
        ax.plot(xy_translation[st:en, 0], xy_translation[st:en, 1], **babbling_style,
                label=label)

    for i, st_en in enumerate(prfs):
        st, en = st_en
        if i == 0:
            label = performing_label
        else:
            label = None
        ax.plot(xy_translation[st:en, 0], xy_translation[st:en, 1], **performing_style,
                label=label)

    if show_yaw:
        vec_x = np.cos(yaw) * 1
        vec_y = np.sin(yaw) * 1
        ax.quiver(xy_translation[::vector_draw_granularity, 0], xy_translation[::vector_draw_granularity, 1],
                  vec_x[::vector_draw_granularity], vec_y[::vector_draw_granularity], color='g',
                  width=0.005, scale=10, scale_units='width', alpha=0.1
                  )


def xy_heading_stages_map(
        ax: C.plt.Axes, xy_translation: np.ndarray, xy_goal: np.ndarray):
    furthest_dim = max(np.max(np.abs(xy_goal)), np.max(np.abs(xy_translation)))
    bound = furthest_dim * 1.1
    map_fig = ax
    map_fig.axhline(y=0., color='r', linestyle='--', alpha=0.1)
    map_fig.axvline(x=0., color='r', linestyle='--', alpha=0.1)
    map_fig.plot(xy_goal[:, 0], xy_goal[:, 1], 'ro', label='goal')
    map_fig.set_ylim(-bound, bound)
    map_fig.set_xlim(-bound, bound)


def draw_staged_evol(ax: C.plt.Axes, y: np.ndarray,
                     stages: np.ndarray,
                     babbling_style={"color": 'r', "linestyle": '-', "alpha": 0.8},
                     performing_style={"color": 'b', "linestyle": '-', "alpha": 0.8},
                     babbling_label="babbling",
                     performing_label="performing"):
    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)
    t = np.arange(len(y))
    for i, st_en in enumerate(bbls):
        st, en = st_en
        if i == 0:
            label = babbling_label
        else:
            label = None
        ax.plot(t[st:en], y[st:en], **babbling_style, label=label)

    for i, st_en in enumerate(prfs):
        st, en = st_en
        if i == 0:
            label = performing_label
        else:
            label = None
        ax.plot(t[st:en], y[st:en], **performing_style, label=label)


def modality_overview(fig: C.plt.Axes, confidence_aggregate: np.ndarray,
                      ts: np.ndarray = None, **kwargs
                      ):
    # tn = len(confidence_aggregate)
    # maxes = np.max(np.abs(confidence_aggregate))
    # normed_conf_agg = confidence_aggregate / (maxes + 0.0001)
    # normed_conf_agg = (normed_conf_agg + 1) / 2
    # is_confident_element = confidence_aggregate >= 0
    normed_conf_agg = confidence_aggregate
    # is_confident_modality = np.mean(is_confident_element, axis=1)

    if ts is None:
        iters = np.arange(len(confidence_aggregate), )
    else:
        iters = ts

    # fig.plot(iters, is_confident_modality, 'k', label="avg(conf)")
    # fig.set_ylim(-0.1, 1.1)
    fig.axhline(y=0.0, color='r', linestyle='--')
    fig.plot(iters, normed_conf_agg[:], **kwargs)


def generate_colorbar(fig: C.plt.Figure, cmap, minmax=(0, 1), **kwargs):
    ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=mpl.colors.Normalize(minmax[0], minmax[1]), **kwargs)


def ax_generate_colorbar(ax: C.plt.Axes, cmap, minmax=(0, 1), **kwargs):
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=mpl.colors.Normalize(minmax[0], minmax[1]), **kwargs)


def comparison_stats(diffs: List[List[np.ndarray]], method_labels, metrics, mean_window=100):
    """

    @param fig:
    @param diffs: [alg[trial]]
    @param title:
    @param method_labels: alg[label]
    @return:
    @rtype:
    """
    metric_means = []
    metric_stds = []
    for i, metric_lab in enumerate(metrics):
        alg_means = []
        alg_stds = []
        for j in range(len(diffs)):
            metric, metric_label = metric_lab
            errs = np.asarray([metric(trial) for k, trial in enumerate(diffs[j])])
            # if i == 0:
            #     _label = method_labels[j]
            # else:
            #     _label = None
            means = np.mean(errs[:, -mean_window:], axis=1)
            # otp = plt.(means)
            alg_means.append(np.mean(means))
            alg_stds.append(np.std(means))
        metric_means.append(alg_means)
        metric_stds.append(alg_stds)
    return metric_means, metric_stds




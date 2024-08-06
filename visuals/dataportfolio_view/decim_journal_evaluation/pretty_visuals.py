import numpy as np
import matplotlib.pyplot as plt

from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation
from visuals import common as C
from typing import List, Tuple, Union
import matplotlib as mpl
import scipy.stats as stats



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



class SampleHitScore(object):
    def __init__(self, colors, model_labels: List[str], row_labels: List[str], time_label, sensor_label, odds_label,
                 evol_label, tick_size, label_size, fancy_model_label: List[str], sensor_unit_label: List[str],
                 sensor_lims: List[tuple], histogram_labels: List[str],
                 log_odds_lims, score2_lims
                 ):
        self.colors = colors
        self.model_labels = model_labels
        self.row_labels = row_labels
        self.time_label = time_label
        self.sensor_label = sensor_label
        self.odds_label = odds_label
        self.evol_label = evol_label
        self.threshold_name = "thr"
        self.confidence_label = "Zero-model"
        self.odds_label = "Model\nprobability"
        self.tick_size = tick_size
        self.label_size = label_size
        self.pivot_line_width = 7
        self.pivot_color = 'b'
        self.bell_alpha = 0.7
        self.fancy_model_label = fancy_model_label
        self.competition_window_label = "Model comparison"
        self.sensor_unit_label = sensor_unit_label
        self.sensor_lims = sensor_lims
        self.histogram_labels = histogram_labels
        self.threshold = 0.0
        self.log_odds_lims = log_odds_lims
        self.score2_lims = score2_lims

    def model_windows(self, model_axs: List[C.plt.Axes], hist_axs: List[C.plt.Axes],
                      emb_obs: List[float], emb_preds: List[List[float]], emb_pred_stds: List[List[float]],
                      model_is_active: np.ndarray
                      ):

        for i in range(len(model_axs)):
            means = [emb_pred[i] for emb_pred in emb_preds]
            stds = [emb_pred_std[i] for emb_pred_std in emb_pred_stds]
            value = emb_obs[i]
            self.model_window(model_axs[i], hist_axs[i], value=value, means=means, stds=stds,
                              model_is_active=model_is_active)
            model_axs[i].set_ylabel(self.row_labels[i], fontsize=self.label_size)

        for i in range(len(model_axs) - 1):
            # model_axs[i].set_xticklabels([])
            hist_axs[i].set_xticklabels([])
        # hist_axs[-1].set_xticks([0, 0.5, 1])
        # hist_axs[-1].set_xticklabels(["0", "", "1"])
        # model_axs[-1].tick_params(axis='x', which='major', labelsize=self.tick_size)
        # hist_axs[-1].tick_params(axis='x', which='major', labelsize=self.tick_size)

        ##
        for i in range(len(model_axs)):
            model_axs[i].set_xlabel(self.sensor_unit_label[i], fontsize=self.tick_size * 0.8)
            model_axs[i].set_xlim(self.sensor_lims[i][0] * 1.2, self.sensor_lims[i][1] * 1.2)
            model_axs[i].set_xticks([self.sensor_lims[i][0], 0, self.sensor_lims[i][1]])
            model_axs[i].set_xticklabels([
                f"{self.sensor_lims[i][0]:1.2f}", "0.00", f"{self.sensor_lims[i][1]:1.2f}"
            ])
            hist_axs[i].set_xlabel(self.histogram_labels[i], fontsize=self.tick_size * 0.8)
            hist_axs[i].set_xticks([0, 0.5, 1])
            hist_axs[i].set_xticklabels(["0", "0.5", "1"])
            # model_axs[-1].tick_params(axis='x', which='major', labelsize=self.tick_size//2)

        ##
        # model_axs[-1].set_xlabel(self.sensor_label, fontsize=self.label_size)
        # hist_axs[-1].set_xlabel(self.odds_label, fontsize=self.label_size)
        model_axs[0].set_title("            " + self.competition_window_label, fontsize=self.label_size)

    def model_window(self, ax_model: C.plt.Axes, ax_hist: C.plt.Axes,
                     value: float, means: List[float], stds: List[float],
                     model_is_active: np.ndarray
                     ):
        ax_model.axvline(x=value, color=self.pivot_color, linestyle='-', linewidth=self.pivot_line_width, alpha=0.8)
        shift = -1.1
        ylines = [i * shift for i in range(len(means))]

        # Data analysis
        odds = []
        bell_axis = []
        bells = []
        for i in range(len(means)):
            m, s, c = means[i], stds[i], self.colors[i]
            x = np.linspace(m - 5 * s, m + 5 * s, 100)
            xpdf = stats.norm.pdf(x, m, s)
            bell_axis.append(x)
            bells.append(xpdf)
            odds.append(-np.square((value - m) / s) - np.log(s))
        # bell_max = -np.log(0.1)

        # Drawing
        for i in range(len(means)):
            # yline = i * 1.1
            ax_model.axhline(y=ylines[i], color='k', linestyle='-')
            _xpdf = bells[i] / np.max(bells[i])
            if i == 0 or model_is_active[i - 1] > 0.5:
                ax_model.fill_between(bell_axis[i], _xpdf + ylines[i], y2=ylines[i], color=self.colors[i],
                                      alpha=self.bell_alpha)
            ##
        odds = np.exp(np.asarray(odds))
        odds[1:] *= model_is_active
        odds /= np.sum(odds)
        ax_hist.stairs(odds, color=self.pivot_color, orientation="horizontal", fill=True)
        ##
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])
        ax_hist.spines['left'].set_visible(False)
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.invert_yaxis()
        ax_hist.grid()

        ##
        ax_model.set_yticks([yline - shift / 2 for yline in ylines])
        ax_model.set_yticklabels(self.model_labels[:len(ylines)])
        ax_model.tick_params(axis='y', rotation=60, labelsize=self.tick_size * 0.8)

        ax_model.spines['right'].set_visible(False)
        ax_model.spines['top'].set_visible(False)
        ax_model.grid()

    def signal_evols(self, axs_odds: List[List[C.plt.Axes]], axs_conf: List[C.plt.Axes],
                     t_axis_seg: np.ndarray, scores: List[np.ndarray], confidences: List[np.ndarray],
                     scenario_phase: np.ndarray, is_babbling: np.ndarray):
        ##
        switch_times = t_axis_seg[scenario_phase == 1.]
        switch_start_end = (t_axis_seg[0], t_axis_seg[0])
        if len(switch_times) > 0:
            switch_start_end = np.min(switch_times), np.max(switch_times)

        babbling_times = t_axis_seg[is_babbling]
        babbling_start_end = (t_axis_seg[0], t_axis_seg[0])
        if len(babbling_times) > 0:
            babbling_start_end = (np.min(babbling_times), np.max(babbling_times))

        for i in range(len(axs_odds)):
            # if switch_start < switch_end:
            #     axs_odds[i].fill_between([switch_start, switch_end], y1=-2, y2=2, color='y', alpha=0.5)
            #     axs_conf[i].fill_between([switch_start, switch_end], y1=-2, y2=2, color='y', alpha=0.5)
            for j in range(len(axs_odds[i])):
                ReferenceAndMap.color_evol_with_stages(
                    axs_odds[i][j], babbling_start_end=babbling_start_end, switch_start_end=switch_start_end,
                    ylim=(-2, 2)
                )
            ReferenceAndMap.color_evol_with_stages(
                axs_conf[i], babbling_start_end=babbling_start_end, switch_start_end=switch_start_end,
                ylim=(-2, 2)
            )

            self.signal_evol(axs_conf[i], t_axis_seg, [confidences[i]],
                             scenario_phase=scenario_phase, threshold=0, colors=[self.colors[0]])
            axs_conf[i].spines['bottom'].set_visible(False)
            for j in range(len(axs_odds[i])):
                self.signal_evol(axs_odds[i][j], t_axis_seg, [scores[i][:, j]],
                                 scenario_phase=scenario_phase, threshold=self.threshold, colors=[self.colors[j + 1]])

        for i in range(len(axs_conf)):
            axs_conf[i].set_xticklabels([])
            axs_conf[i].set_yticks([self.score2_lims[0], 0, self.score2_lims[1]])
            axs_conf[i].set_yticklabels([f"{self.score2_lims[0]}", "", f"{self.score2_lims[1]}"])
            axs_conf[i].set_ylabel(self.fancy_model_label[0],
                                   rotation='horizontal',
                                   fontsize=self.tick_size,
                                   labelpad=35, va='center')
            axs_conf[i].set_ylim(self.score2_lims[0] * 1.2, self.score2_lims[1] * 1.2)

            ##

        for i in range(len(axs_odds)):
            for j in range(len(axs_odds[i])):
                axs_odds[i][j].set_yticks([self.log_odds_lims[0], 0, self.log_odds_lims[1]])
                axs_odds[i][j].set_yticklabels([f"{self.log_odds_lims[0]}", "", f"{self.log_odds_lims[1]}"])
                axs_odds[i][j].set_ylabel(self.fancy_model_label[j + 1],
                                          rotation='horizontal',
                                          fontsize=self.tick_size,
                                          labelpad=35, va='center')
                axs_odds[i][j].set_ylim(self.log_odds_lims[0] * 1.2, self.log_odds_lims[1] * 1.2)
            ##

        for i in range(len(axs_odds)):
            for j in range(len(axs_odds[i])):
                if i == len(axs_odds) - 1 and j == len(axs_odds[i]) - 1:
                    continue
                axs_odds[i][j].set_xticklabels([])
                # axs_odds[i][j].set_xticks([])
        axs_odds[-1][-1].set_xlabel(self.time_label, fontsize=self.label_size)
        axs_odds[-1][-1].tick_params(axis='x', which='major', labelsize=self.tick_size)
        axs_conf[0].set_title(self.evol_label, fontsize=self.label_size)

    @staticmethod
    def signal_evol(ax: C.plt.Axes, t_axis_seg: np.ndarray, score: List[np.ndarray], scenario_phase: np.ndarray,
                    threshold, colors: List[str]):
        ax.set_facecolor('lightgrey')
        # fill_boolean(ax, scenario_phase == 1., y1=-2., y2=2., color='y', alpha=0.2)
        t_mid = float(t_axis_seg[-1] + t_axis_seg[0]) / 2
        ax.axvline(x=t_mid, color='k', linestyle='-', alpha=0.5)
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.8, linewidth=2)
        for i in range(len(score)):
            ax.plot(t_axis_seg, score[i], color=colors[i])
            ax.fill_between(t_axis_seg, score[i], y2=threshold - 2, color=colors[i], alpha=0.5)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(np.min(t_axis_seg), np.max(t_axis_seg))

        # ax.grid()


class ReferenceAndMap():
    def __init__(
            self, sensory_labels: List[str],
            y_lims: List[Tuple[float, float]],
            label_size: float, tick_size: float, x_cut: Tuple[float, float], time_label: str, map_cmap: str,
            map_box_limits: List[float],
            map_labels: List[str],
            score2_threshold: float, score2_label: str, score2_lims: Tuple[float, float],
            odds_label: str, odds_lims: Tuple[float, float],
            threshold_label: str, threshold_style: dict, score2_style: dict,
            model_colors: List[str], goal: np.ndarray
    ) -> None:
        self.sensory_labels = sensory_labels
        self.y_lims = y_lims
        self.label_size = label_size
        self.tick_size = tick_size
        self.x_cut = x_cut
        self.time_label = time_label
        self.map_cmap = map_cmap
        self.map_box_limits = map_box_limits
        self.map_labels = map_labels
        self.score2_threshold = score2_threshold
        self.score2_label = score2_label
        self.score2_lims = score2_lims
        self.score2_style = score2_style
        self.odds_label = odds_label
        self.odds_lims = odds_lims
        self.threshold_label = threshold_label
        self.threshold_style = threshold_style
        self.model_colors = model_colors
        self.evol_linewidth = 3
        self.xdiff = (((self.map_box_limits[1] - self.map_box_limits[0]) * 10) // 4) / 10
        self.ydiff = (((self.map_box_limits[3] - self.map_box_limits[2]) * 10) // 4) / 10

        self.xcenter = (self.map_box_limits[1] + self.map_box_limits[0]) * 10 // 2 / 10
        self.ycenter = (self.map_box_limits[3] + self.map_box_limits[2]) * 10 // 2 / 10

        self.xticks = [self.map_box_limits[0] + self.xdiff * i for i in range(0, 5)]
        self.yticks = [self.map_box_limits[2] + self.ydiff * i for i in range(0, 5)]

        self.goal = goal

    @staticmethod
    def color_evol_with_stages(ax: C.plt.Axes, babbling_start_end: Tuple[int, int], switch_start_end: Tuple[int, int],
                               ylim: Tuple[float, float]
                               ):
        babbling_start, babbling_end = babbling_start_end
        if babbling_start < babbling_end:
            ax.fill_between([babbling_start, babbling_end], y1=ylim[0], y2=ylim[1], color='g', alpha=0.3)

        switch_start, switch_end = switch_start_end
        _babbling_end = np.maximum(switch_start, babbling_end)
        if switch_start < switch_end:
            if switch_start < babbling_start:
                ax.fill_between([switch_start, babbling_start], y1=ylim[0], y2=ylim[1], color='y', alpha=0.3)
            ax.fill_between([_babbling_end, switch_end], y1=ylim[0], y2=ylim[1], color='y', alpha=0.3)

    @staticmethod
    def _add_evol_style(ax: C.plt.Axes, side_label, label_size,
                        xlim: Tuple[float, float], ylim: Tuple[float, float],
                        tick_size, switch_start_end: Tuple[int, int],
                        babbling_start_end: Tuple[int, int],
                        inflate_lims=1.
                        ):
        # add vertical line in the middle of the cut
        ax.set_ylabel(side_label, fontsize=label_size * 0.65)
        ax.set_xlim(*xlim)
        ax.axvline(x=(xlim[0] + xlim[1]) / 2, color="k", linestyle="-", alpha=0.5)

        _ylim = ylim[0] * inflate_lims, ylim[1] * inflate_lims
        ax.set_ylim(*_ylim)
        ## set y ticks
        # arange array of y ticks between the limits with 2.5 step
        ax.set_yticks([ylim[0], (ylim[0] + ylim[1]) / 2, ylim[1]])
        # every odd ticklabel is empty string
        # ax.set_yticklabels([str(int(ticks[i])) if i % 2 == 0 else "" for i in range(len(ticks))])
        ax.set_yticklabels([f"{ylim[0]}", "", f"{ylim[1]}"])

        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)

        # babbling_start, babbling_end = babbling_start_end
        # if babbling_start < babbling_end:
        #     ax.fill_between([babbling_start, babbling_end], y1=ylim[0], y2=ylim[1], color='g', alpha=0.3)
        #
        # switch_start, switch_end = switch_start_end
        # _babbling_end = np.maximum(switch_start, babbling_end)
        # if switch_start < switch_end:
        #     ax.fill_between([switch_start, babbling_start], y1=ylim[0], y2=ylim[1], color='y', alpha=0.3)
        #     ax.fill_between([_babbling_end, switch_end], y1=ylim[0], y2=ylim[1], color='y', alpha=0.3)

        ReferenceAndMap.color_evol_with_stages(ax, babbling_start_end=babbling_start_end,
                                               switch_start_end=switch_start_end, ylim=_ylim)
        # if i < len(evol_axs) - 1:
        #     ax.set_xticklabels([])

    @staticmethod
    def _map_path_colors(odds, second_score: np.ndarray, second_score_threshold,
                         model_colors, second_score_color):
        zm_win = second_score > second_score_threshold
        model_winners = np.argmax(odds, axis=1)
        color_sequence = []
        for i in range(len(second_score)):
            if zm_win[i]:
                color_sequence.append(second_score_color)
            else:
                color_sequence.append(model_colors[model_winners[i]])
        return color_sequence

    def reference_and_map_plot(self, evol_axs: List[C.plt.Axes], map_ax: C.plt.Axes, confidence_ax: C.plt.Axes,
                               t_axis_seg, y_trg_signal, y_obs_signal, heading_signal,
                               location_signal, goal_point, second_score, odds: np.ndarray,
                               scenario_phase: np.ndarray, map_colors: list, is_babbling: np.ndarray
                               ):
        # gs = GridSpec(3, 2, figure=fig, width_ratios=[0.6, 0.35], wspace=0.03, hspace=0.25)
        # axs = [[fig.add_subplot(gspec)] for gspec in [gs[0, 0], gs[1, 0], gs[2, 0]]]
        """EVOLUTION"""
        switch_times = t_axis_seg[scenario_phase == 1.]
        switch_start, switch_end = (t_axis_seg[0], t_axis_seg[0])

        if len(switch_times) > 0:
            switch_start = np.min(switch_times)
            switch_end = np.max(switch_times)

        babbling_times = t_axis_seg[is_babbling]
        babbling_start_end = (t_axis_seg[0], t_axis_seg[0])
        if len(babbling_times) > 0:
            babbling_start_end = (np.min(babbling_times), np.max(babbling_times))

        """sensors"""
        # axs = V.common.subplots(f, len(sel_sensors) + 1, 1)
        for i in range(len(y_obs_signal)):
            ax = evol_axs[i]
            _y_trg = np.clip(y_trg_signal[i], a_min=self.y_lims[i][0], a_max=self.y_lims[i][1])
            _y_obs = np.clip(y_obs_signal[i], a_min=self.y_lims[i][0], a_max=self.y_lims[i][1])

            ax.plot(t_axis_seg, _y_trg, label="Reference", color="r", alpha=0.8,
                    linewidth=self.evol_linewidth)
            ax.plot(t_axis_seg, _y_obs, label="Estimation", color="b", alpha=0.8,
                    linewidth=self.evol_linewidth)
            ax.axhline(y=0, color='k')

        xlim = (np.min(t_axis_seg), np.max(t_axis_seg))

        # scores

        for i in range(odds.shape[1]):
            confidence_ax.plot(t_axis_seg, odds[:, i], color=self.model_colors[i],
                               linewidth=self.evol_linewidth, alpha=1.)
            confidence_ax.fill_between(t_axis_seg, odds[:, i], y2=self.score2_lims[0],
                                       color=self.model_colors[i], linewidth=self.evol_linewidth,
                                       alpha=0.3)
        confidence_ax.plot(t_axis_seg, second_score,
                           linewidth=self.evol_linewidth, alpha=1, **self.score2_style)
        confidence_ax.fill_between(t_axis_seg, second_score, y2=self.score2_lims[0],
                                   linewidth=self.evol_linewidth, alpha=0.3, **self.score2_style)
        confidence_ax.axhline(y=self.score2_threshold, **self.threshold_style)

        # styling
        side_labels = [self.sensory_labels[i] for i in range(len(y_trg_signal))]
        for i in range(len(y_obs_signal)):
            ax = evol_axs[i]
            # add vertical line in the middle of the cut
            self._add_evol_style(ax, side_label=side_labels[i], label_size=self.label_size,
                                 xlim=xlim, ylim=self.y_lims[i], tick_size=self.tick_size,
                                 switch_start_end=(switch_start, switch_end),
                                 babbling_start_end=babbling_start_end)
            ax.set_xticklabels([])
        self._add_evol_style(confidence_ax, side_label=self.score2_label, label_size=self.label_size,
                             xlim=xlim, ylim=self.score2_lims, tick_size=self.tick_size,
                             switch_start_end=(switch_start, switch_end),
                             babbling_start_end=babbling_start_end, inflate_lims=1.2)
        # confidence_ax.set_yticks([-0.5, 0, 0.5])
        # confidence_ax.set_yticklabels(["0", "", "1"])
        confidence_ax.set_xlabel(self.time_label, fontsize=self.label_size)
        """MAP"""
        # arrow
        # goal arrow
        dist = np.linalg.norm(self.goal - location_signal[-1, :])
        dx = (self.goal[0] - location_signal[-1, 0]) / dist * self.xdiff * 0.9
        dy = (self.goal[1] - location_signal[-1, 1]) / dist * self.ydiff * 0.9
        map_ax.arrow(location_signal[-1, 0], location_signal[-1, 1], dx, dy,
                     length_includes_head=True, head_width=0.2 * self.xdiff,
                     head_length=0.2 * self.xdiff, color='r', alpha=0.8,
                     width=0.04 * self.xdiff)
        # heading arrow
        dxh = np.cos(heading_signal[-1]) * self.xdiff * 0.5
        dyh = np.sin(heading_signal[-1]) * self.ydiff * 0.5
        map_ax.arrow(location_signal[-1, 0], location_signal[-1, 1], dxh, dyh,
                     length_includes_head=True, head_width=0.2 * self.xdiff,
                     head_length=0.2 * self.xdiff, color='b', alpha=0.8, width=0.04 * self.xdiff)
        # scatter location with the gradient coloring
        map_ax.scatter(location_signal[:, 0], location_signal[:, 1], c=map_colors, alpha=1, s=4)
        # show goal with a circle of radius 0.5
        map_ax.add_patch(plt.Circle((goal_point[0], goal_point[1]), 0.5, color="r", alpha=0.5))
        # add large white goal marker
        map_ax.plot([goal_point[0]], [goal_point[1]], linestyle='', marker='o', markersize=10, label="Goal", color="w",
                    alpha=0.9)

        ## styling
        map_ax.yaxis.tick_right()
        map_ax.yaxis.set_label_position("right")
        map_ax.set_xlim(self.map_box_limits[0], self.map_box_limits[1])
        map_ax.set_ylim(self.map_box_limits[2], self.map_box_limits[3])

        map_ax.set_xticks(self.xticks)
        map_ax.set_yticks(self.yticks)
        map_ax.set_xlim(self.map_box_limits[0] - self.xdiff * .1, self.map_box_limits[1] + self.xdiff * .1)
        map_ax.set_ylim(self.map_box_limits[2] - self.ydiff * .1, self.map_box_limits[3] + self.ydiff * .1)
        # map_ax.set_xticklabels([str(i) for i in range(0, self.map_box_limits[1] + 1, 20)])

        # map_ax.set_yticks([i for i in range(self.map_box_limits[2], self.map_box_limits[3] + 1, 20)])
        map_ax.tick_params(axis='both', which='major', labelsize=self.tick_size)

        # ax.set_xticklabels([str(i) for i in range(-5, 6)])
        map_ax.set_xlabel(self.map_labels[0], fontsize=self.label_size)
        map_ax.set_ylabel(self.map_labels[1], fontsize=self.label_size)
        map_ax.grid()
        # grey background
        map_ax.set_facecolor('grey')


class ModelLearning(object):
    def __init__(self, motor_label: str, sensor_label: str, motor_lims: List[Tuple[float, float]],
                 sensori_labels,
                 motor_labels,
                 tick_size, label_size, weight_cmap,
                 weights_norm, model_labels
                 ) -> None:
        self.motor_label = motor_label
        self.motor_lims = motor_lims
        self.motor_labels = motor_labels
        self.sensori_labels = sensori_labels
        self.sensor_label = sensor_label
        self.tick_size = tick_size
        self.label_size = label_size
        self.weight_cmap = weight_cmap
        self.weights_norm = weights_norm
        self.err_max = 5
        self.time_label = "Time (s)"
        self.model_labels = model_labels
        self.error_label = "MSE"

    def sensori_motor_hists(self, model_axs: List[C.plt.Axes], evol_axs: List[C.plt.Axes], hist_axs: List[C.plt.Axes],
                            motor_commands: np.ndarray, t_axis_seg: np.ndarray,
                            models: List[np.ndarray], scenario_phase: np.ndarray, is_babbling: np.ndarray,
                            pred_errors: List[np.ndarray],
                            ):
        t_min_max = np.min(t_axis_seg), np.max(t_axis_seg)
        switch_times = t_axis_seg[scenario_phase == 1.]
        switch_start_end = (t_axis_seg[0], t_axis_seg[0])
        if len(switch_times) > 0:
            switch_start_end = np.min(switch_times), np.max(switch_times)

        babbling_times = t_axis_seg[is_babbling]
        babbling_start_end = (t_axis_seg[0], t_axis_seg[0])
        if len(babbling_times) > 0:
            babbling_start_end = (np.min(babbling_times), np.max(babbling_times))

        ##
        for i, model_weights in enumerate(models):
            model_axs[i].matshow(model_weights, cmap=self.weight_cmap, aspect="auto", norm=self.weights_norm)
            hist_axs[i].stairs(pred_errors[i], color="r", fill=True)
            hist_axs[i].set_ylim(0, self.err_max)
            hist_axs[i].set_xlim(0, pred_errors[i].shape[0])

        for i in range(motor_commands.shape[1]):
            evol_axs[i].plot(t_axis_seg, motor_commands[:, i], 'b')
            ReferenceAndMap.color_evol_with_stages(
                evol_axs[i], babbling_start_end=babbling_start_end,
                switch_start_end=switch_start_end, ylim=(-5, 5))
            evol_axs[i].set_ylim(*self.motor_lims[i])
            evol_axs[i].set_xlim(*t_min_max)
            evol_axs[i].axvline(x=(t_min_max[0] + t_min_max[1]) / 2, color='k', linestyle='-', alpha=0.5)

        ##
        for i in range(motor_commands.shape[1] - 1):
            evol_axs[i].set_xticklabels([])
        evol_axs[-1].set_xlabel(self.time_label, fontsize=self.label_size)
        evol_axs[0].set_title(self.motor_label, fontsize=self.label_size)

        for i in range(len(models)):
            if i < len(models) - 1:
                hist_axs[i].set_yticks([])
            hist_axs[i].yaxis.tick_right()
            hist_axs[i].set_xticks([])
            hist_axs[i].set_title(self.model_labels[i], fontsize=self.label_size)

            model_axs[i].tick_params(axis='x', rotation=70, labelsize=self.tick_size)
            model_axs[i].tick_params(axis='y', labelsize=self.tick_size)
            model_axs[i].set_xticks([i for i in range(len(self.sensori_labels))])
            model_axs[i].set_xticklabels(self.sensori_labels)
            model_axs[i].set_yticks([i for i in range(len(self.motor_labels))])
            model_axs[i].set_yticklabels(self.motor_labels)
            model_axs[i].xaxis.tick_bottom()
            model_axs[i].yaxis.tick_right()

            # model_axs[i].set_yticks([i for i in range(len(performance_error_value))])
            # model_axs[i].set_yticklabels([self.sensori_labels[i] for i in range(len(performance_error_value))])
        hist_axs[-1].yaxis.set_label_position("right")
        hist_axs[-1].set_ylabel(self.error_label, fontsize=self.label_size)


class AggregateSignal(object):
    def __init__(
            self,
            signal_styles: List[dict], ylabel: str,
            ylim: Tuple[Union[float, np.ndarray], Union[float, np.ndarray]],
            xlabel: str, x_axis_on: bool,
            xlabel_style: dict, ylabel_style: dict, x_ticks_size: float, y_ticks_size: float,
            threshold: float = None, clip_ylim_on = False, ylim_clip_pad = .1, no_center_line=False
    ):
        self.signal_styles = signal_styles
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.ylim = ylim
        self.x_axis_on = x_axis_on
        self.xlabel_style = xlabel_style
        self.ylabel_style = ylabel_style
        self.x_ticks_size = x_ticks_size
        self.y_ticks_size = y_ticks_size
        self.threshold = threshold
        self.clip_ylim_on = clip_ylim_on
        yeps = (self.ylim[1] - self.ylim[0]) * ylim_clip_pad
        self.clip_ylim = (self.ylim[0] + yeps, self.ylim[1] - yeps)
        self.no_center_line=no_center_line
    def signal_evol(self, ax: C.plt.Axes, t_signal: np.ndarray, y_signals: List[np.ndarray]):
        for i, sig in enumerate(y_signals):
            _sig = sig
            if self.clip_ylim_on:
                _sig = np.clip(sig, a_min=self.clip_ylim[0], a_max=self.clip_ylim[1])
            ax.plot(t_signal, _sig, **self.signal_styles[i])
        xlim = np.min(t_signal), np.max(t_signal)
        if not self.no_center_line:
            ax.axvline(x=(xlim[0] + xlim[1]) / 2, color="k", linestyle="-", alpha=0.5)
        if self.threshold is not None:
            ax.axhline(y=self.threshold, color='r', linestyle='--', alpha=0.8, linewidth=2)

        ax.set_ylabel(self.ylabel, **self.ylabel_style)
        ax.set_ylim(self.ylim)
        ax.tick_params(axis='y', which='major', labelsize=self.y_ticks_size)
        ax.set_xlim(*xlim)

        ##
        if self.x_axis_on:
            ax.set_xlabel(self.xlabel, **self.xlabel_style)
            ax.tick_params(axis='x', which='major', labelsize=self.x_ticks_size)
        else:
            ax.set_xticklabels([])

    def signal_stages(self, ax: C.plt.Axes, t_signal: np.ndarray,
                      is_babbling: np.ndarray, scenario_phase: np.ndarray):
        switch_times = t_signal[scenario_phase == 1.]
        switch_start_end = (t_signal[0], t_signal[0])

        if len(switch_times) > 0:
            switch_start_end = np.min(switch_times), np.max(switch_times)

        babbling_times = t_signal[is_babbling]
        babbling_start_end = (t_signal[0], t_signal[0])
        if len(babbling_times) > 0:
            babbling_start_end = (np.min(babbling_times), np.max(babbling_times))

        ReferenceAndMap.color_evol_with_stages(
            ax, babbling_start_end=babbling_start_end,
            switch_start_end=switch_start_end, ylim=self.ylim)



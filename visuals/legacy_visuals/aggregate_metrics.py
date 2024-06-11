import numpy as np

from visuals import common as C
from typing import List


def _filter_nonperforming(mem, performing, gait_n, avg_window=10):
    avg_mem = np.mean(mem[:avg_window], axis=0)
    was_performing = False
    ret = np.zeros(mem.shape)
    segments_rest = 0
    for i in range(len(mem)):
        if performing[i]:
            if not was_performing:
                segments_rest = i
            if i - segments_rest < gait_n*3:
                ret[i] = avg_mem
            else:
                ret[i] = mem[i]
        else:
            if was_performing:
                avg_mem = np.mean(mem[i-avg_window-2:i-2], axis=0)
                ret[i-1] = avg_mem
            ret[i] = avg_mem
        was_performing = performing[i]
    return ret


def mse_mae_comparison(fig: C.plt.Figure, diffs: List[List[np.ndarray]],
                       title, labels, colors, is_performing: List[List[np.ndarray]],
                       filter_babbling=False,
                       convolution_window=100, mean_window=100):
    """

    @param fig:
    @param diffs: [alg[trial]]
    @param title:
    @param labels: alg[label]
    @return:
    @rtype:
    """
    fig.suptitle(title)
    v = np.ones((convolution_window*2,)) / convolution_window
    v[convolution_window:] = 0

    if filter_babbling:
        def _abs(diff, is_performing):
            return _filter_nonperforming(np.abs(diff), is_performing, gait_n=4, avg_window=10)

        def _sqr(diff, is_performing):
            return _filter_nonperforming(np.square(diff), is_performing, gait_n=4, avg_window=10)
    else:
        def _abs(diff, is_performing):
            return np.abs(diff)

        def _sqr(diff, is_performing):
            return np.square(diff)

    def _int(err):
        return np.cumsum(np.mean(err, axis=1))

    def _smh(err):
        return np.convolve(np.mean(err, axis=1), v=v)[convolution_window*2:-convolution_window*2]

    metrics = [
        (_int, _sqr, "int(mse)"), (_int, _abs, "int(mae)"),
        (_smh, _sqr, "avg(mse)"), (_smh, _abs, "avg(mae)")
    ]

    figs = C.subplots(fig, len(metrics), 2)

    for i, metric_lab in enumerate(metrics):
        aggregator, metric, metric_label = metric_lab
        fig_evol = figs[i][0]
        fig_bp = figs[i][1]
        alg_means = []
        for j in range(len(diffs)):
            errs = np.asarray([aggregator(metric(trial, is_performing[j][k])) for k, trial in enumerate(diffs[j])])
            err_means = np.mean(errs, axis=0)
            err_min = np.min(errs, axis=0)
            err_max = np.max(errs, axis=0)
            if i == 0:
                _label = labels[j]
            else:
                _label = None
            fig_evol.plot(err_min, '--', color=colors[j],alpha=0.12)
            fig_evol.plot(err_max, '--', color=colors[j],alpha=0.12)
            fig_evol.fill_between(np.arange(len(err_means),), y1=err_min, y2=err_max, color=colors[j], alpha=0.1)
            fig_evol.plot(err_means, color=colors[j], label=_label, alpha=0.8)


            ##
            alg_means.append(np.mean(errs[:, -mean_window:], axis=1))
        fig_bp.boxplot(alg_means)
        fig_evol.set_ylabel(metric_label)
    figs[0][0].set_title("Evolution")
    figs[0][1].set_title(f"Stats [-{mean_window}:]")
    fig.legend()

from visuals import common as C
import numpy as np
from agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation

def time_stamp_analysis(fig: C.plt.Figure, time_stamps: np.ndarray,duration_average: np.ndarray,
                              stages, title, time_step=0.015):
    fig.suptitle(title)
    figs = C.subplots(fig, 1, 2)
    fig_evol = figs[0][0]
    fig_bp = figs[0][1]
    ##

    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)

    d_t = time_stamps[1:] - time_stamps[:-1]
    dt_m = np.mean(d_t)
    dt_std = np.std(d_t)

    for st, en in bbls:
        fig_evol.axvspan(xmin=st, xmax=en, alpha=0.2, color='k')
        fig_evol.axvspan(xmin=st, xmax=en, alpha=0.2, color='k')

    for st, en in prfs:
        fig_evol.axvspan(xmin=st, xmax=en, alpha=0.2, color='r')
        fig_evol.axvspan(xmin=st, xmax=en, alpha=0.2, color='r')

    fig_evol.plot(d_t, 'k', alpha=0.7)
    fig_evol.plot(duration_average, '-b')
    fig_evol.axhline(y=time_step, alpha=0.5, color='r')
    fig_evol.set_ylim(bottom=0, top=dt_m + 2 * dt_std)

    x_tick_labels = []
    data = []
    for i, prf in enumerate(prfs):
        # fig_bp.boxplot(d_t[prf[0]:prf[1]])
        data.append(d_t[prf[0]:prf[1]])
        x_tick_labels.append(f"prf{i+1}")

    for i, prf in enumerate(bbls):
        # fig_bp.boxplot(d_t[prf[0]:prf[1]])
        data.append(d_t[prf[0]:prf[1]])
        x_tick_labels.append(f"bbl{i+1}")
    fig_bp.boxplot(data)
    fig_bp.axhline(y=time_step, alpha=0.5, color='r')
    fig_bp.set_xticks([i+1 for i in range(len(x_tick_labels))])
    fig_bp.set_xticklabels(x_tick_labels)
    fig_bp.set_ylim(bottom=dt_m - 2*dt_std, top=dt_m + 2*dt_std)







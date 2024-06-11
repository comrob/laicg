from matplotlib.pyplot import Figure

from agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation
from visuals import common as C
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def xy_heading_map(
        fig: Figure, xy_translation: np.ndarray,
        yaw: np.ndarray,title, xy_goal: np.ndarray = None,
        vector_draw_granularity=2000
):

    if xy_goal is None:
        xy_goal = np.zeros((1, 2))

    furthest_dim = max(np.max(np.abs(xy_goal)), np.max(np.abs(xy_translation)))
    bound = furthest_dim * 1.1

    fig.suptitle(title)
    figs = C.subplots(fig, 1, 1)
    map_fig = figs[0][0]
    map_fig.plot(xy_translation[:, 0], xy_translation[:, 1], 'k')
    map_fig.plot(xy_translation[-1, 0], xy_translation[-1, 1], 'bo')
    map_fig.plot(xy_goal[:, 0], xy_goal[:, 1], 'r.')
    map_fig.axhline(y=0., color='k', linestyle='--', alpha=0.2)
    map_fig.axvline(x=0., color='k', linestyle='--', alpha=0.2)

    vec_x = np.cos(yaw) * 1
    vec_y = np.sin(yaw) * 1
    map_fig.quiver(xy_translation[::vector_draw_granularity, 0], xy_translation[::vector_draw_granularity, 1],
                 vec_x[::vector_draw_granularity], vec_y[::vector_draw_granularity], color='g',
                 width=0.005, scale=10, scale_units='width', alpha=0.5
                 )

    map_fig.set_ylim(-bound, bound)
    map_fig.set_xlim(-bound, bound)


def xy_heading_stages_map(
        fig: Figure, xy_translation: np.ndarray,
        yaw: np.ndarray,title, xy_goal: np.ndarray,
        stages: np.ndarray,
        vector_draw_granularity=2000
):
    prfs = BabblePerformanceAlternation.StageStates.PERFORMANCE_STAGE.get_intervals(stages)
    bbls = BabblePerformanceAlternation.StageStates.BABBLING_STAGE.get_intervals(stages)
    furthest_dim = max(np.max(np.abs(xy_goal)), np.max(np.abs(xy_translation)))
    bound = furthest_dim * 1.1
    colors = C.colors(len(prfs), cmap='tab10')
    vec_x = np.cos(yaw) * 1
    vec_y = np.sin(yaw) * 1

    fig.suptitle(title)
    figs = C.subplots(fig, 1, 1)
    map_fig = figs[0][0]

    map_fig.axhline(y=0., color='r', linestyle='--', alpha=0.1)
    map_fig.axvline(x=0., color='r', linestyle='--', alpha=0.1)
    map_fig.plot(xy_goal[:, 0], xy_goal[:, 1], 'ro', label='goal')

    for i, st_en in enumerate(bbls):
        st, en = st_en
        if i == 0:
            label = "bbl"
        else:
            label = None
        map_fig.plot(xy_translation[st:en, 0], xy_translation[st:en, 1], color='k', linestyle='-', alpha=0.8, label=label)

    for i, st_en in enumerate(prfs):
        st, en = st_en
        map_fig.plot(xy_translation[st:en, 0], xy_translation[st:en, 1], color=colors[i], linestyle='-', alpha=0.8, label=f"prf{i}")

    map_fig.quiver(xy_translation[::vector_draw_granularity, 0], xy_translation[::vector_draw_granularity, 1],
                 vec_x[::vector_draw_granularity], vec_y[::vector_draw_granularity], color='g',
                 width=0.005, scale=10, scale_units='width', alpha=0.1
                 )

    map_fig.plot(xy_translation[-1, 0], xy_translation[-1, 1], 'o', color='b', alpha=0.8,
                 label=f"stop")
    map_fig.set_ylim(-bound, bound)
    map_fig.set_xlim(-bound, bound)


    ###
    furthest_dim_prf = np.max(np.abs(xy_translation[prfs[0][0]:prfs[0][1], :]))
    bound_prf = furthest_dim_prf * 1.1
    zoom = max(int((bound/3)/bound_prf), 1)
    if zoom > 4:
        axins = zoomed_inset_axes(map_fig, zoom, loc=1)
        axins.axhline(y=0., color='r', linestyle='--', alpha=0.1)
        axins.axvline(x=0., color='r', linestyle='--', alpha=0.1)
        for i, st_en in enumerate(bbls):
            st, en = st_en
            axins.plot(xy_translation[st:en, 0], xy_translation[st:en, 1], color='k', linestyle='-', alpha=0.8)

        for i, st_en in enumerate(prfs):
            st, en = st_en
            axins.plot(xy_translation[st:en, 0], xy_translation[st:en, 1], color=colors[i], linestyle='-', alpha=0.8,
                         label=f"prf{i}")

        axins.quiver(xy_translation[::vector_draw_granularity, 0], xy_translation[::vector_draw_granularity, 1],
                       vec_x[::vector_draw_granularity], vec_y[::vector_draw_granularity], color='g',
                       width=0.005, scale=10, scale_units='width', alpha=0.1
                       )

        axins.set_xlim(-bound_prf, bound_prf)
        axins.set_ylim(-bound_prf, bound_prf)
        mark_inset(map_fig, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    map_fig.legend(loc='upper left')


def navigation_signals(
        fig: Figure, xyz_translation: np.ndarray, roll_pitch_yaw: np.ndarray,
        forward_ref: np.ndarray, side_ref: np.ndarray, turn_ref: np.ndarray,
        title, xy_goal: np.ndarray = None, cut_start=50
):

    if xy_goal is None:
        xy_goal = np.zeros((len(xyz_translation), 2))
    xyz_goal = np.zeros((len(xyz_translation), 3))
    xyz_goal[:, :2] = xy_goal

    yaw_goal = np.arctan2(xy_goal[:, 1]-xyz_translation[:, 1], xy_goal[:, 0]-xyz_translation[:, 0])
    rpy_goal = np.zeros((len(xyz_translation), 3))
    rpy_goal[:, 2] = yaw_goal

    reference = np.zeros((len(xyz_translation), 3))
    reference[:, 0] = forward_ref
    reference[:, 1] = side_ref
    reference[:, 2] = turn_ref

    fig.suptitle(title)
    figs = C.subplots(fig, 9, 1)

    ##
    for i, dim_label in enumerate(["x", "y", "z"]):
        f = figs[i][0]
        if i == 0:
            _obs_label = "position"
            _trg_label = "goal"
        else:
            _obs_label = None
            _trg_label = None

        f.plot(xyz_translation[cut_start:, i], 'b', label=_obs_label)
        f.plot(xyz_goal[cut_start:, i], 'k', label=_trg_label)
        f.set_ylabel(dim_label)

    for i, dim_label in enumerate(["roll", "pitch", "yaw"]):
        f = figs[i+3][0]
        if i == 0:
            _obs_label = "angle"
            _trg_label = "goal"
        else:
            _obs_label = None
            _trg_label = None

        f.plot(roll_pitch_yaw[cut_start:, i], 'b', label=_obs_label)
        f.plot(rpy_goal[cut_start:, i], 'k', label=_trg_label)
        f.set_ylabel(dim_label)

    for i, dim_label in enumerate(["fwd_ref", "side_ref", "turn_ref"]):
        f = figs[i+6][0]
        if i == 0:
            _obs_label = "reference"
        else:
            _obs_label = None
        f.plot(reference[cut_start:, i], 'r', label=_obs_label)
        f.set_ylabel(dim_label)
    ##


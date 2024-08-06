import numpy as np
from typing import List, Tuple
from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation
import coupling_evol.data_process.postprocess.permanent_post_processing.factories as F


def detect_first_move(signal):
    dsignal = signal[1:] - signal[:-1]
    o = np.arange(len(dsignal))
    x = o[np.abs(dsignal) > 0.0001]
    if len(x) == 0:
        return 0
    return x[1]


def is_in_init_(is_learning, raw_second_score):
    is_inactive = np.abs(raw_second_score) < 0.0001
    return np.logical_and(is_inactive, np.logical_not(is_learning))


def filter_by_stage(_odds, _second_score, is_learning, is_in_init, threshold=0.5, lower_limit=-0.5, starts_with_learning=False):
    odds = np.zeros_like(_odds) + _odds
    second_score = np.zeros_like(_second_score) + _second_score
    ##
    learning_n = np.minimum(
        np.sum(np.logical_and(np.logical_not(is_learning[:-1]), is_learning[1:])),
        np.sum(np.logical_and(np.logical_not(is_learning[1:]), is_learning[:-1])))

    if starts_with_learning:
        learning_n += 1
    odds[is_learning] = lower_limit
    second_score[is_learning] = -lower_limit

    second_score[is_in_init] = lower_limit
    odds[is_in_init] = lower_limit

    learned_counter = 0
    was_learning = False
    for t in range(len(is_in_init)):
        if not was_learning and is_learning[t]:
            learned_counter += 1
        if is_in_init[t]:
            odds[t, -1 - learning_n + learned_counter] = -lower_limit
        was_learning = is_learning[t]

    return odds, second_score


class SignalDataProcess(object):
    def __init__(
            self,
            navigation: F.coppelia_environment.Navigation,
            time_record: F.dynamic_lifecycle.SegmentedTimeAxis,
            scenario: F.dynamic_lifecycle.ScenarioSwitch,
            scores: F.model_selection.SegmentedScores,
            learning_threshold: float, sfmx_pow=100.,
            common_distance_lim: Tuple[np.ndarray, np.ndarray] = None,
    ):
        self._navigation = navigation
        self._time_record = time_record
        self._scenario = scenario
        self._scores = scores
        self._learning_threshold = learning_threshold
        self._common_distance_lim = common_distance_lim
        self._sfmx_pow = sfmx_pow
        ##
        self._model_log_odds_cmp = [(0, 1), (1, 0)]
        ##
        self.goal_distance: np.ndarray
        self.nrm_odds: np.ndarray
        self.second_score: np.ndarray
        self.t_axis_seg: np.ndarray
        self.scenario_phase: np.ndarray
        self.is_learning: np.ndarray

    def process(self):
        _stages = self._navigation.stages
        #########
        timestamps = self._time_record.timestamp - self._time_record.timestamp[0]
        iters_seg = (self._time_record.iteration * 100).astype(int)
        self.is_learning = _stages[iters_seg] == BabblePerformanceAlternation.StageStates.BABBLING_STAGE.value

        ##
        second_score = self._scores.second_score
        model_loggods = self._scores.model_logodds
        odds = np.mean(model_loggods, axis=(2, 3))
        t_iter = self._scores.iter
        _scenario_phase = self._scenario.phase
        t_iter_ids = t_iter.astype(int)
        self.scenario_phase = _scenario_phase[t_iter_ids]
        # t_axis_seg = np.arange(0, len(y_mem_obss[0]))
        self.t_axis_seg = timestamps
        ##
        learning_threshold = self._learning_threshold
        is_in_init = is_in_init_(self.is_learning, raw_second_score=second_score)
        second_score = -(second_score - learning_threshold)
        # norming between -.5, .5

        """DISTANCE"""
        nav = self._navigation
        start_location = nav.location[10]
        goal = nav.goal[-1, :] - start_location
        _xy = (nav.location - start_location)
        xy = _xy[iters_seg]

        self.goal_distance = np.linalg.norm(xy - goal[None, :], axis=1)

        """LOG ODDS"""
        nrm_odds = np.zeros_like(odds) + np.min(odds)
        for i in range(odds.shape[1]):
            offset = detect_first_move(odds[:, i])
            nrm_odds[offset:, i] = odds[offset:, i]
        self.second_score = second_score
        self.nrm_odds, _ = filter_by_stage(nrm_odds, second_score, self.is_learning, is_in_init,
                                                 threshold=0, lower_limit=-1)
        return self

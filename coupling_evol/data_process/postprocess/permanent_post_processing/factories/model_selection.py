from coupling_evol.data_process.postprocess.lifecycle_data import LifeCycleRawData
from coupling_evol.data_process.postprocess.permanent_post_processing.common import ProcessedData
from coupling_evol.engine.common import RecordNamespace
import numpy as np
from coupling_evol.data_process.inprocess.record_logger import select_prefix, postfix_adder


class TwoStageScores(ProcessedData):
    def __init__(self):
        super().__init__()
        self.first_score: np.ndarray = np.zeros((1,))
        self.second_score: np.ndarray = np.zeros((1,))
        self.selected_model = np.zeros((1,))

    def process(self, dlcdp: LifeCycleRawData):
        from coupling_evol.agent.components.ensemble_dynamics.estimator_competition import \
            EstimatorTwoStageCompetition as E
        rec = select_prefix(dlcdp.get_raw_record(), str(RecordNamespace.LIFE_CYCLE.key))
        self.first_score = rec[E.R_FIRST_STAGE_SCORE]
        self.second_score = rec[E.R_SECOND_STAGE_SCORE]
        self.selected_model = rec[E.R_MODEL_SELECTION]

def sum_by_segments(signal, phase_segments):
    segment_start = 0
    current_segment = np.argmax(phase_segments[0, :])
    ret = []
    seg = []
    for i in range(signal.shape[0]):
        if np.argmax(phase_segments[i, :]) != current_segment:  # and (i > segment_start):
            ret.append(np.sum(signal[segment_start: i], axis=0))
            seg.append(phase_segments[segment_start, :])
            current_segment = np.argmax(phase_segments[i, :])
            segment_start = i
    return np.asarray(ret), np.asarray(seg)


def fill_non_active_with_prev(segmented_signal, segments, active_phase_id):
    """
    Replaces the default value in signal with the last valid value.
    This is for cases when the record is stored once per gait.
    """
    ret = np.zeros_like(segmented_signal)
    current_value = segmented_signal[0]
    for i in range(len(segmented_signal)):
        if segments[i][active_phase_id] == 1:
            current_value = segmented_signal[i]
        ret[i] = current_value
    return ret, segments


class SegmentedScores(ProcessedData):
    def __init__(self):
        super().__init__()
        self.first_score: np.ndarray = np.zeros((1,))
        self.second_score: np.ndarray = np.zeros((1,))
        self.best_zero_confidence: np.ndarray = np.zeros((1,))
        self.model_logodds: np.ndarray = np.zeros((1,))
        self.iter: np.ndarray = np.zeros((1,))
        self.segments: np.ndarray = np.zeros((1,))
        self.selected_model = np.zeros((1,))


    def process(self, dlcdp: LifeCycleRawData):
        from coupling_evol.agent.components.ensemble_dynamics.estimator_competition import \
            EstimatorTwoStageCompetition as E
        from coupling_evol.agent.components.embedding.cpg_rbf_embedding import CpgRbf
        from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation as LC_CONST
        from coupling_evol.agent.components.internal_model.forward_model import mean_by_segments, median_by_segments

        rec = select_prefix(dlcdp.get_raw_record(), str(RecordNamespace.LIFE_CYCLE.key))

        a = rec[postfix_adder(LC_CONST.CPG_PSFX)(CpgRbf.ACTIVATION_NAME)]

        # in the estimator competitions, the values are valid only at last phase of the gait.
        active_phase_id = a.shape[1] - 1

        self.first_score, segments = fill_non_active_with_prev(
            *sum_by_segments(rec[E.R_FIRST_STAGE_SCORE], a), active_phase_id)
        self.best_zero_confidence, _ = fill_non_active_with_prev(
            *sum_by_segments(rec[E.R_CONFIDENCE_AGGREGATE], a), active_phase_id)
        self.second_score, _ = fill_non_active_with_prev(
            *sum_by_segments(rec[E.R_SECOND_STAGE_SCORE], a), active_phase_id)
        self.model_logodds, _ = fill_non_active_with_prev(
            *sum_by_segments(rec[E.R_MODEL_LOGODDS], a), active_phase_id)

        self.iter, _ = mean_by_segments(np.arange(len(a)), a)
        self.selected_model, _ = median_by_segments(rec[E.R_MODEL_SELECTION], a)
        self.segments = segments

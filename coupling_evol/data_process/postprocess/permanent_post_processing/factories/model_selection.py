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

        self.first_score, segments = mean_by_segments(rec[E.R_FIRST_STAGE_SCORE], a)
        self.best_zero_confidence, _ = mean_by_segments(rec[E.R_CONFIDENCE_AGGREGATE],a)
        self.second_score, _ = mean_by_segments(rec[E.R_SECOND_STAGE_SCORE], a)
        self.model_logodds, _ = mean_by_segments(rec[E.R_MODEL_LOGODDS], a)
        self.iter, _ = mean_by_segments(np.arange(len(a)), a)
        self.selected_model, _ = median_by_segments(rec[E.R_MODEL_SELECTION], a)
        self.segments = segments

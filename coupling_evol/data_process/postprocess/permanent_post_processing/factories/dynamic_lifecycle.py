from coupling_evol.agent.components.internal_model.forward_model import MultiPhaseModel
from coupling_evol.data_process.postprocess.lifecycle_data import LifeCycleRawData
from coupling_evol.data_process.postprocess.permanent_post_processing.common import ProcessedData
from coupling_evol.engine.embedded_control import EmbeddedTargetParameter
from coupling_evol.engine.common import RecordNamespace
import numpy as np
from coupling_evol.data_process.inprocess.record_logger import select_prefix, postfix_adder
from typing import List


class SegmentedTimeAxis(ProcessedData):
    def __init__(self):
        super().__init__()
        self.iteration: np.ndarray = np.zeros((1,))
        self.duration: np.ndarray = np.zeros((1,))
        self.timestamp: np.ndarray = np.zeros((1,))
        self.segments: np.ndarray = np.zeros((1,))
    
    def process(self, dlcdp: LifeCycleRawData):
        from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation as LC_CONST
        from coupling_evol.engine.experiment_executor import R_DURAVG, R_T, R_TIMESTAMP
        from coupling_evol.agent.components.embedding.cpg_rbf_embedding import CpgRbf
        from coupling_evol.agent.components.internal_model.forward_model import mean_by_segments

        rec_lc = select_prefix(dlcdp.get_raw_record(), RecordNamespace.LIFE_CYCLE.key)
        rec_ex = select_prefix(dlcdp.get_raw_record(), RecordNamespace.EXECUTOR.key)

        a = rec_lc[postfix_adder(LC_CONST.CPG_PSFX)(CpgRbf.ACTIVATION_NAME)]

        self.iteration, self.segments = mean_by_segments(rec_ex[R_T], a)
        self.duration, _ = mean_by_segments(rec_ex[R_DURAVG], a)
        self.timestamp, _  = mean_by_segments(rec_ex[R_TIMESTAMP], a)


class MotorSensorTargetMem(ProcessedData):
    def __init__(self):
        super().__init__()
        self.observation: np.ndarray = np.zeros((1,))
        self.command: np.ndarray = np.zeros((1,))
        self.target: np.ndarray = np.zeros((1,))
        self.metrics: np.ndarray = np.zeros((1,))
        self.weights: np.ndarray = np.zeros((1,))
        self.segments: np.ndarray = np.zeros((1,))
        self.performing_stage: np.ndarray = np.zeros((1,))

    @staticmethod
    def _reduce_embeddings(embedding_signal, a):
        return np.sum(embedding_signal * a[:, None, :], axis=2)

    def process(self, dlcdp: LifeCycleRawData):
        from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation as LC_CONST
        from coupling_evol.agent.components.embedding.cpg_rbf_embedding import CpgRbf
        from coupling_evol.agent.components.internal_model.forward_model import mean_by_segments
        rec_lc = select_prefix(dlcdp.get_raw_record(), RecordNamespace.LIFE_CYCLE.key)
        rec_ex = select_prefix(dlcdp.get_raw_record(), RecordNamespace.EXECUTOR.key)

        a = rec_lc[postfix_adder(LC_CONST.CPG_PSFX)(CpgRbf.ACTIVATION_NAME)]

        target_values_r = self._reduce_embeddings(rec_lc[LC_CONST.R_TARGET], a)
        target_metrics_r = self._reduce_embeddings(rec_lc[LC_CONST.R_TARGET + EmbeddedTargetParameter.METRIC_NAME], a)
        target_weights_r = self._reduce_embeddings(rec_lc[LC_CONST.R_TARGET + EmbeddedTargetParameter.WEIGHT_NAME], a)
        observation = rec_lc[LC_CONST.R_OBSERVATION]
        command = rec_lc[LC_CONST.R_CMD]
        performing_stage = rec_lc[LC_CONST.R_STAGE] == LC_CONST.StageStates.PERFORMANCE_STAGE.value[0]

        self.target, self.segments = mean_by_segments(target_values_r, a)
        self.metrics, _ = mean_by_segments(target_metrics_r, a)
        self.weights, _ = mean_by_segments(target_weights_r, a)
        self.observation, _ = mean_by_segments(observation, a)
        self.command, _ = mean_by_segments(command, a)
        prf, _ = mean_by_segments(performing_stage, a)
        self.performing_stage = prf > 0.9


class ScenarioSwitch(ProcessedData):
    def __init__(self):
        super().__init__()
        self.phase: np.ndarray = np.zeros((1,))

    def process(self, dlcdp: LifeCycleRawData):
        from coupling_evol.engine.scenarios.combiners import R_SWITCH
        rec = select_prefix(dlcdp.get_raw_record(), str(RecordNamespace.SCENARIO_CONTROLLER.key))
        self.phase = rec[R_SWITCH]


class Models(ProcessedData):
    def __init__(self):
        super().__init__()
        self.models: List[MultiPhaseModel] = []

    @staticmethod
    def _reduce_embeddings(embedding_signal, a):
        return np.sum(embedding_signal * a[:, None, :], axis=2)

    def process(self, dlcdp: LifeCycleRawData):
        self.models = dlcdp.get_models()
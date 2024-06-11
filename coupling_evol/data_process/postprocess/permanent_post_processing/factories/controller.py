from typing import List
import numpy as np
from coupling_evol.data_process.inprocess.record_logger import select_prefix, postfix_adder
from coupling_evol.data_process.postprocess.lifecycle_data import LifeCycleRawData
from coupling_evol.data_process.postprocess.permanent_post_processing.common import ProcessedData
from coupling_evol.engine.common import RecordNamespace

LIST_NP = List[np.ndarray]

class SegmentedFepSignals(ProcessedData):
    def __init__(self):
        super().__init__()
        self.y_estimation: np.ndarray = np.zeros((1,))
        self.segments: np.ndarray = np.zeros((1,))

    @staticmethod
    def _reduce_embeddings(embedding_signal, a):
        return np.sum(embedding_signal * a[:, None, :], axis=2)

    def process(self, dlcdp: LifeCycleRawData):
        from coupling_evol.agent.components.controllers.fep_controllers import SENSORY_ESTIMATE
        from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation as LC_CONST
        from coupling_evol.agent.components.embedding.cpg_rbf_embedding import CpgRbf
        from coupling_evol.agent.components.internal_model.forward_model import mean_by_segments

        rec_lc = select_prefix(dlcdp.get_raw_record(), RecordNamespace.LIFE_CYCLE.key)
        estimation = rec_lc[postfix_adder(LC_CONST.CONTROL_PSFX)(SENSORY_ESTIMATE)]
        a = rec_lc[postfix_adder(LC_CONST.CPG_PSFX)(CpgRbf.ACTIVATION_NAME)]
        estimation_red = self._reduce_embeddings(estimation, a)
        self.y_estimation, self.segments = mean_by_segments(estimation_red, a)





import numpy as np
from coupling_evol.data_process.inprocess.record_logger import select_prefix
from coupling_evol.data_process.postprocess.lifecycle_data import LifeCycleRawData
from coupling_evol.data_process.postprocess.permanent_post_processing.common import ProcessedData


class Navigation(ProcessedData):
    def __init__(self):
        super().__init__()
        self.location: np.ndarray = np.zeros((1,))
        self.heading: np.ndarray = np.zeros((1,))
        self.goal: np.ndarray = np.zeros((1,))
        self.stages: np.ndarray = np.zeros((1,))

    def process(self, dlcdp: LifeCycleRawData):
        from coupling_evol.environments.coppeliasim.target_providers.navigation import R_GOAL_XY, R_POS_XYZ, R_POS_RPY
        from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation as LC_CONST

        lcy_rec = select_prefix(dlcdp.get_raw_record(), "lcy")
        rec = select_prefix(dlcdp.get_raw_record(), "trg")

        self.location = rec[R_POS_XYZ][:, :2]
        self.heading = rec[R_POS_RPY][:, 2]
        self.goal = rec[R_GOAL_XY]
        self.stages = lcy_rec[LC_CONST.R_STAGE]
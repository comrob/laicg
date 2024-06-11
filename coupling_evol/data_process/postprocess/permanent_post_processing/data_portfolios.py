from coupling_evol.data_process.postprocess.permanent_post_processing import common as C
from coupling_evol.data_process.postprocess.permanent_post_processing import factories as F
import os


class DecimJournalEvaluation(C.DataPortfolio):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.uyt_mem = C.ProcessedDataWrap[
            F.dynamic_lifecycle.MotorSensorTargetMem](
            os.path.join(data_path, "uyt_mem"), F.dynamic_lifecycle.MotorSensorTargetMem()
        )
        self.navigation = C.ProcessedDataWrap[
            F.coppelia_environment.Navigation](
            os.path.join(data_path, "navigation"), F.coppelia_environment.Navigation()
        )
        self.scenario_switch = C.ProcessedDataWrap[
            F.dynamic_lifecycle.ScenarioSwitch](
            os.path.join(data_path, "scenario_switch"), F.dynamic_lifecycle.ScenarioSwitch()
        )
        self.segmented_model_selection = C.ProcessedDataWrap[
            F.model_selection.SegmentedScores](
            os.path.join(data_path, "segmented_model_selection"), F.model_selection.SegmentedScores())

        self.models = C.ModelsWrap(os.path.join(data_path, "models"))


class LongevEvaluation(C.DataPortfolio):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.uyt_mem = C.ProcessedDataWrap[
            F.dynamic_lifecycle.MotorSensorTargetMem](
            os.path.join(data_path, "uyt_mem"), F.dynamic_lifecycle.MotorSensorTargetMem()
        )
        self.navigation = C.ProcessedDataWrap[
            F.coppelia_environment.Navigation](
            os.path.join(data_path, "navigation"), F.coppelia_environment.Navigation()
        )
        self.scenario_switch = C.ProcessedDataWrap[
            F.dynamic_lifecycle.ScenarioSwitch](
            os.path.join(data_path, "scenario_switch"), F.dynamic_lifecycle.ScenarioSwitch()
        )
        self.segmented_model_selection = C.ProcessedDataWrap[
            F.model_selection.SegmentedScores](
            os.path.join(data_path, "segmented_model_selection"), F.model_selection.SegmentedScores())

        self.models = C.ModelsWrap(os.path.join(data_path, "models"))

        self.time_axis = C.ProcessedDataWrap[
            F.dynamic_lifecycle.SegmentedTimeAxis](
            os.path.join(data_path, "time_axis"), F.dynamic_lifecycle.SegmentedTimeAxis()
        )

        self.fep_signals = C.ProcessedDataWrap[
            F.controller.SegmentedFepSignals](
            os.path.join(data_path, "fep_signals"), F.controller.SegmentedFepSignals()
        )

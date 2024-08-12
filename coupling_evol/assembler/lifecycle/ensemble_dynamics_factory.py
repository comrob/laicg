import coupling_evol.agent.components.ensemble_dynamics.estimator_competition as EC
import coupling_evol.agent.components.ensemble_dynamics.experimental as ECE
from coupling_evol.agent.components.ensemble_dynamics import EnsembleDynamics
from abc import ABC, abstractmethod
from coupling_evol.assembler.common import *
from typing import Tuple


class EnsembleDynamicsType(Enum):
    TWO_STAGE_RAPID = 1
    TWO_STAGE_AGGREGATE = 2
    TWO_STAGE_AGGREGATE_COMPOSITION = 3
    TWO_STAGE_AGGREGATE_ODOMETRYSUM = 4
    SUBMODES = 5
    TWO_STAGE_AGGREGATE_ODOMETRYSUM_COMPOSITION = 6


class EnsembleDynamicsConfiguration(FactoryConfiguration[EnsembleDynamicsType]):
    def __init__(self):
        super().__init__()
        self.created_type: EnsembleDynamicsType = EnsembleDynamicsType.TWO_STAGE_AGGREGATE_ODOMETRYSUM
        # self.force_last_model = False


_FULL_CONFIG = Tuple[EnsembleDynamicsConfiguration, EssentialParameters, LifeCycleParameters]


class EnsembleDynamicsFactory(ABC):
    def __init__(self, configuration: _FULL_CONFIG):
        self.configuration = configuration[0]
        self.essential = configuration[1]
        self.lifecycle = configuration[2]

    @abstractmethod
    def __call__(self, sensory_dim, motor_dim, phase_dim, models) -> EnsembleDynamics:
        # FIXME the dimensions come from essentials or at least can be extracted from models so it is quite useless here
        pass


class TwoStageRapidFactory(EnsembleDynamicsFactory):
    def __call__(self, sensory_dim, motor_dim, phase_dim, models):
        cnf = self.configuration.arguments
        return EC.EstimatorTwoStageCompetition(
            sensory_dim=sensory_dim, motor_dim=motor_dim, phase_dim=phase_dim,
            models=models, zero_neighbourhood_epsilon=cnf["zero_neighbourhood_epsilon"],
            log_best_zero_odds_threshold=0.01, min_confident_elements_rate=cnf["min_confident_elements_rate"],
            zero_model_standard_deviation=cnf["zero_model_standard_deviation"], gait_buffer_size=10,
            zero_model_flip_threshold_in=(cnf["zero_model_flip_thr"], cnf["zero_model_filp_hist"]),
            max_performing_periods=cnf["max_performing_periods"]
        )


class TwoStageAggregateFactory(EnsembleDynamicsFactory):
    def __call__(self, sensory_dim, motor_dim, phase_dim, models):
        args = self.configuration.arguments
        return EC.TwoStageAggregatedScoreCompetition(
            sensory_dim=sensory_dim, motor_dim=motor_dim, phase_dim=phase_dim,
            models=models, zero_neighbourhood_epsilon=args["zero_neighbourhood_epsilon"],
            log_best_zero_odds_threshold=0.01, min_confident_elements_rate=args["min_confident_elements_rate"],
            zero_model_standard_deviation=args["zero_model_standard_deviation"], gait_buffer_size=10,
            ensemble_confidence_lr=args["score_lr"], log_ensemble_confidence_combiner=args["score_is_log"],
            max_performing_periods=args["max_performing_periods"],
            # force_last_model_selection=self.configuration.force_last_model
            force_last_model_selection=self.lifecycle.force_keep_same_model
        )


class TwoStageAggregateCompositionFactory(EnsembleDynamicsFactory):
    def __call__(self, sensory_dim, motor_dim, phase_dim, models):
        args = self.configuration.arguments
        return EC.TwoStageAggregatedScoreComposition(
            sensory_dim=sensory_dim, motor_dim=motor_dim, phase_dim=phase_dim,
            models=models, zero_neighbourhood_epsilon=args["zero_neighbourhood_epsilon"],
            log_best_zero_odds_threshold=0.01, min_confident_elements_rate=args["min_confident_elements_rate"],
            zero_model_standard_deviation=args["zero_model_standard_deviation"], gait_buffer_size=10,
            ensemble_confidence_lr=args["score_lr"], log_ensemble_confidence_combiner=args["score_is_log"],
            max_performing_periods=args["max_performing_periods"]
        )


class TwoStageAggregateOdometrysumFactory(EnsembleDynamicsFactory):
    def __call__(self, sensory_dim, motor_dim, phase_dim, models):
        args = self.configuration.arguments
        return EC.TwoStageAggregatedScoreCompetitionOdometrySum(
            sensory_dim=sensory_dim, motor_dim=motor_dim, phase_dim=phase_dim,
            models=models, zero_neighbourhood_epsilon=args["zero_neighbourhood_epsilon"],
            log_best_zero_odds_threshold=0.01, min_confident_elements_rate=args["min_confident_elements_rate"],
            zero_model_standard_deviation=args["zero_model_standard_deviation"], gait_buffer_size=100,
            ensemble_confidence_lr=args["score_lr"], log_ensemble_confidence_combiner=args["score_is_log"],
            max_performing_periods=args["max_performing_periods"],
            max_zero_suggestions=args["max_selection_eval_time"],
            max_candidate_suggestions=args["min_selection_eval_time"],
            direct_confidence_score=args["direct_confidence_score"],
            # force_last_model_selection=self.configuration.force_last_model
            force_last_model_selection=self.lifecycle.force_keep_same_model

        )


class TwoStageAggregateOdometrysumCompositionFactory(EnsembleDynamicsFactory):
    def __call__(self, sensory_dim, motor_dim, phase_dim, models):
        args = self.configuration.arguments
        return EC.TwoStageAggregatedScoreCompositionOdometrySum(
            sensory_dim=sensory_dim, motor_dim=motor_dim, phase_dim=phase_dim,
            models=models, zero_neighbourhood_epsilon=args["zero_neighbourhood_epsilon"],
            log_best_zero_odds_threshold=0.01, min_confident_elements_rate=args["min_confident_elements_rate"],
            zero_model_standard_deviation=args["zero_model_standard_deviation"], gait_buffer_size=100,
            ensemble_confidence_lr=args["score_lr"], log_ensemble_confidence_combiner=args["score_is_log"],
            max_performing_periods=args["max_performing_periods"],
            # force_last_model_selection=self.configuration.force_last_model,
            force_last_model_selection=self.lifecycle.force_keep_same_model,
            max_zero_suggestions=args["max_selection_eval_time"],
            max_candidate_suggestions=args["min_selection_eval_time"],
            direct_confidence_score=args["direct_confidence_score"],
            softmax_power=args["softmax_power"],
            composite_weight_normer=args["composite_weight_normer"],
            continual_model_update=args["continual_model_update"],
            submodel_combination=args["submodel_combination"]
        )


class SubmodesFactory(EnsembleDynamicsFactory):
    def __call__(self, sensory_dim, motor_dim, phase_dim, models):
        args = self.configuration.arguments
        return ECE.SubmodesModelSelection(
            sensory_dim=sensory_dim, motor_dim=motor_dim, phase_dim=phase_dim,
            models=models, zero_neighbourhood_epsilon=args["zero_neighbourhood_epsilon"],
            log_best_zero_odds_threshold=0.01, min_confident_elements_rate=args["min_confident_elements_rate"],
            zero_model_standard_deviation=args["zero_model_standard_deviation"], gait_buffer_size=100,
            ensemble_confidence_lr=args["score_lr"], log_ensemble_confidence_combiner=args["score_is_log"],
            max_performing_periods=args["max_performing_periods"],
            # force_last_model_selection=self.configuration.force_last_model,
            force_last_model_selection=self.lifecycle.force_keep_same_model,
            min_max_selection_eval_time=(args["min_selection_eval_time"], args["max_selection_eval_time"])
        )


_ENUM_FACTORY_MAP = {
    EnsembleDynamicsType.TWO_STAGE_RAPID: TwoStageRapidFactory,
    EnsembleDynamicsType.TWO_STAGE_AGGREGATE: TwoStageAggregateFactory,
    EnsembleDynamicsType.TWO_STAGE_AGGREGATE_COMPOSITION: TwoStageAggregateCompositionFactory,
    EnsembleDynamicsType.TWO_STAGE_AGGREGATE_ODOMETRYSUM: TwoStageAggregateOdometrysumFactory,
    EnsembleDynamicsType.TWO_STAGE_AGGREGATE_ODOMETRYSUM_COMPOSITION: TwoStageAggregateOdometrysumCompositionFactory,
    EnsembleDynamicsType.SUBMODES: SubmodesFactory
}


def factory(configuration: EnsembleDynamicsConfiguration, essentials: EssentialParameters,
            lifecycle: LifeCycleParameters) -> EnsembleDynamicsFactory:
    return _ENUM_FACTORY_MAP[configuration.created_type]((configuration, essentials, lifecycle))

from coupling_evol.agent.components.internal_model import regressors as R
import numpy as np

from coupling_evol.assembler import *
from coupling_evol.assembler.experiment_assembly import AssemblyConfiguration

"""
Here will be high level functions that build experiment instances from primitives (+ enums).
"""


class GoalType(Enum):
    STATIC_LINX = 0
    STATIC_ROTX = 1
    STATIC_ROTY = 2
    STATIC_ROTZ = 3
    MOTION_DEFAULT = 4
    NAVIGATION_TURN_AND_GO = 5
    NAVIGATION_TRANSLATE = 6
    NAVIGATION_TURN_AND_GO_BOUNDED = 7

    def is_static_goal(self) -> bool:
        return self in [
            self.STATIC_ROTY,
            self.STATIC_ROTZ,
            self.STATIC_LINX,
            self.STATIC_ROTX
        ]

    def is_navigation_goal(self) -> bool:
        return self in [
            self.NAVIGATION_TURN_AND_GO,
            self.NAVIGATION_TRANSLATE,
            self.NAVIGATION_TURN_AND_GO_BOUNDED
        ]


class LifeCycleRunParameters:
    def __init__(self, babble_iters, delta_search_iters, performance_measure_iters, babbling_rate, model_builder):
        self.babble_iters = babble_iters
        self.delta_search_iters = delta_search_iters
        self.performance_measure_iters = performance_measure_iters
        self.babbling_rate = babbling_rate
        self.model_builder = model_builder
        self.max_iters = 0

    def set_max_iters_from_context_steps(self, ctx_num):
        self.max_iters = (self.babble_iters + self.delta_search_iters + 2) * ctx_num + 1
        return self


class _EssentialParameters:
    def __init__(self,
                 motor_dimension: int, sensory_dimension: int, granularity: int, natural_frequency: float,
                 transgait_window_size: int, integration_step_size: float, rbf_epsilon: float,
                 step_sleep: float = 0.01):
        self.motor_dimension = motor_dimension
        self.sensory_dimension = sensory_dimension
        self.granularity = granularity
        self.natural_frequency = natural_frequency
        self.transgait_window_size = transgait_window_size
        self.integration_step_size = integration_step_size
        self.rbf_epsilon = rbf_epsilon
        self.step_sleep = step_sleep


class ScenarioBuild(object):
    def __init__(self):
        self.args: Dict[str, Union[int, float]] = {}
        self.scenario_controller_type = ScenarioControllerType.NONE
        self.switch_fractions = (-1,)
        self.switch_distances = (-1,)
        self.center = (0, 0)

    @classmethod
    def init_default(cls):
        return cls()

    def set_scenario_controller_type(self, typ: ScenarioControllerType):
        self.scenario_controller_type = typ
        return self

    def switch_scenario_at_fractions(self, switch_fraction: float, switch_back_fraction: float):
        """
        @param switch_fraction: fraction of total iterations when the switch should occur.
        @param switch_back_fraction: fraction of total iterations when the switch should occur.
        @return:
        @rtype:
        """
        assert 0 <= switch_fraction < switch_back_fraction <= 1
        self.switch_fractions = (switch_fraction, switch_back_fraction)
        return self

    def switch_scenario_at_distances_to(self, switch_distance: float, switch_back_distance: float, distance_to: tuple):
        self.switch_distances = (switch_distance, switch_back_distance)
        self.center = distance_to
        return self

    def build(self) -> ScenarioControllerConfiguration:
        scc = ScenarioControllerConfiguration()
        scc.created_type = self.scenario_controller_type
        scc.center = self.center
        scc.switch_distances = self.switch_distances
        scc.switch_fractions = self.switch_fractions
        scc.arguments = self.args
        return scc


class DynamicLifeCycleConfiguration(object):
    def __init__(self,
                 strategy: LifeCycleType,
                 args: Dict[str, Union[int, float]],
                 ):
        self.strategy = strategy
        self.args = args
        self.start_with_babbling = True
        self.ensemble_dynamics_strategy = EnsembleDynamicsType.TWO_STAGE_RAPID
        self.performance_babble_rate = 0.
        self.force_keep_same_model = False

    @classmethod
    def init_default(cls):
        return cls.init(LifeCycleType.REPEATED_SCHEDULE)

    @classmethod
    def init(cls, strategy: LifeCycleType):
        return cls(strategy, {
            "max_performing_periods": np.inf
        })

    def zero_neighbourhood_eps(self, val: float):
        self.args["zero_neighbourhood_epsilon"] = val
        return self

    def min_confident_elements_rate(self, val: float):
        self.args["min_confident_elements_rate"] = val
        return self

    def zero_model_standard_deviation(self, val: float):
        self.args["zero_model_standard_deviation"] = val
        return self

    def zero_model_flip_threshold_in(self, threshold, window_size):
        self.args["zero_model_flip_thr"] = threshold
        self.args["zero_model_filp_hist"] = window_size
        return self

    def ensemble_dynamics_score_lr(self, lr: float):
        self.args["score_lr"] = lr
        return self

    def log_score_combination(self, is_log=True):
        self.args["score_is_log"] = is_log
        return self

    def direct_confidence_score(self, is_direct_confidence=True):
        self.args["direct_confidence_score"] = is_direct_confidence
        return self

    def set_start_with_babbling(self):
        self.start_with_babbling = True
        return self

    def set_start_with_performing(self):
        self.start_with_babbling = False
        return self

    def set_ensemble_strategy(self, strat: EnsembleDynamicsType):
        self.ensemble_dynamics_strategy = strat
        return self

    def set_performance_babble_rate(self, rate: float):
        self.performance_babble_rate = rate
        return self

    def max_performing_periods(self, periods: Union[int, np.ndarray]):
        self.args["max_performing_periods"] = periods
        return self

    def set_force_keep_same_model(self, is_forced: bool = True):
        self.force_keep_same_model = is_forced
        return self

    def model_selection_evaluation_time(self, min_time: int, max_time: int):
        self.args["min_selection_eval_time"] = min_time
        self.args["max_selection_eval_time"] = max_time
        return self

    def model_selection_composition(self, normer: ModelCompositionType, softmax_power: float,
                                    continual_model_update: bool = False, submodel_combination: bool = True):
        self.args["composite_weight_normer"] = normer.value
        self.args["softmax_power"] = softmax_power
        self.args["continual_model_update"] = continual_model_update
        self.args["submodel_combination"] = submodel_combination
        return self


class ComplexLifecycleConfiguration(object):
    def __init__(self,
                 ep: _EssentialParameters,
                 rp: LifeCycleRunParameters,
                 babbler_param: BabblerParameterization,
                 babbler_args: Dict[str, Union[int, float]],
                 controller_search_strategy: ControllerType,
                 controller_search_args: Dict[str, Union[int, float]],
                 ):
        # model

        self.ep = ep
        self.rp = rp
        self.babbler_param = babbler_param
        self.babbler_args = babbler_args
        self.controller_search_strategy = controller_search_strategy
        self.controller_search_args = controller_search_args
        self.dynamic_lifecycle = DynamicLifeCycleConfiguration.init_default()
        self.scenario = ScenarioControllerConfiguration()

    def motor_sens_dim(self, motor_dim: int, sens_dim: int):
        self.ep.sensory_dimension = sens_dim
        self.ep.motor_dimension = motor_dim
        return self

    def babble_iters(self, iters: int):
        self.rp.babble_iters = iters
        return self

    def control_iters(self, iters: int):
        self.rp.delta_search_iters = iters
        return self

    def babble_scale(self, scale):
        self.rp.babbling_rate = scale
        return self

    def log_prior_strength(self, weight):
        self.controller_search_args["log_prior_strength"] = weight
        return self

    def amplitude_symmetry_strength(self, weight):
        self.controller_search_args["amplitude_symmetry_strength"] = weight
        return self

    def amplitude_energy_regularization(self, sum_amplitude_energy, regularization_strength):
        self.controller_search_args["sum_amplitude_energy"] = sum_amplitude_energy
        self.controller_search_args["sum_amplitude_energy_strength"] = regularization_strength
        return self

    def scale_learning_rates(self, scale=1.):
        for k in self.controller_search_args:
            if "learning_rate" in k:
                # print(f"Scaling {k} from {self.controller_search_args[k]} to {self.controller_search_args[k] * scale}")
                self.controller_search_args[k] *= scale
        return self

    def scale_gait_learning_rate(self, scale=1.):
        self.controller_search_args["gait_learning_rate"] *= scale
        return self

    def target_variance_learning_rate(self, lr: float):
        self.controller_search_args["target_error_variance_learning_rate"] = lr
        return self

    def prediction_variance_learning_rate(self, lr: float):
        self.controller_search_args["likelihood_variance_learning_rate"] = lr
        return self

    def scale_fep_fusion_learning_granularity(self, scale=1.):
        self.scale_learning_rates(1 / scale)
        self.controller_search_args["observed_sensory_variance"] /= scale
        self.controller_search_args["estimation_prior_variance"] /= scale
        self.controller_search_args["prediction_variance_lower_bound"] /= scale
        return self

    def static_variance(self, observed: float, predicted: float, prior: float):
        self.controller_search_args["observed_sensory_variance"] = observed
        self.controller_search_args["estimation_prior_variance"] = prior
        self.controller_search_args["prediction_variance_lower_bound"] = predicted
        ## zero the variance learning rates
        self.target_variance_learning_rate(0.)
        self.prediction_variance_learning_rate(0.)
        return self

    def scale_fep_fusion_sensory_to_prediction_variance(self, scale=.1):
        self.controller_search_args["observed_sensory_variance"] = self.controller_search_args[
                                                                       "prediction_variance_lower_bound"] * scale
        return self

    def variance_precision_scale(self, target=1., prediction=1.):
        self.controller_search_args["prediction_precision_scale"] = prediction
        self.controller_search_args["target_precision_scale"] = target
        return self

    def switch_to_simplified_gradient(self, use=True):
        self.controller_search_args["simplified_gradient_switch"] = use
        return self

    def set_target_error_processing(self, sum_on=False, typ=None):
        if typ is None:
            if sum_on is True:
                typ = 1
            else:
                typ = 0
        self.controller_search_args["target_error_processing"] = typ
        return self

    def set_step_sleep(self, step_sleep):
        self.ep.step_sleep = step_sleep
        return self

    def set_dynamic_lifecycle_config(self, dccl: DynamicLifeCycleConfiguration):
        self.dynamic_lifecycle = dccl
        return self

    def get_max_iters_from_context_steps(self, ctx_num):
        return (self.rp.babble_iters + self.rp.delta_search_iters + 2) * ctx_num + 1

    def use_fep_controller_container(self, use=False):
        self.controller_search_args["use_fep_controller_container"] = use
        return self


def build_assembly_configuration(clc: ComplexLifecycleConfiguration,
                                 scc: ScenarioControllerConfiguration,
                                 pc: ProviderConfiguration,
                                 context_num: int) -> AssemblyConfiguration:
    ##
    ep = EssentialParameters(
        motor_dimension=clc.ep.motor_dimension,
        sensory_dimension=clc.ep.sensory_dimension,
        integration_step_size=clc.ep.integration_step_size
    )
    ##
    lcp = LifeCycleParameters(granularity=clc.ep.granularity)
    lcp.force_keep_same_model = clc.dynamic_lifecycle.force_keep_same_model
    ##
    esp = ExperimentSetupParameters(
        max_iters=clc.rp.max_iters,
        step_sleep=clc.ep.step_sleep

    )
    ##
    bp = clc.babbler_param
    ##
    cc = ControllerConfiguration()
    cc.created_type = clc.controller_search_strategy
    cc.arguments = clc.controller_search_args
    ##
    edc = EnsembleDynamicsConfiguration()
    edc.created_type = clc.dynamic_lifecycle.ensemble_dynamics_strategy
    edc.arguments = clc.dynamic_lifecycle.args
    ##
    lcc = LifeCycleConfiguration(
        natural_frequency=clc.ep.natural_frequency,
        rbf_epsilon=clc.ep.rbf_epsilon,
        babble_iters=clc.rp.babble_iters,
        delta_search_iters=clc.rp.delta_search_iters,
        babbling_rate=clc.rp.babbling_rate
    )
    lcc.created_type = clc.dynamic_lifecycle.strategy
    lcc.arguments = clc.dynamic_lifecycle.args
    lcc.start_with_babbling = clc.dynamic_lifecycle.start_with_babbling
    esp.set_max_iters_from_context_steps(ctx_num=context_num,
                                         babble_iters=lcc.babble_iters, performance_iters=lcc.delta_search_iters)

    ##
    return AssemblyConfiguration(
        essential_parameters=ep,
        lifecycle_parameters=lcp,
        experiment_setup_parameters=esp,
        babbler_parametrization=bp,
        controller_configuration=cc,
        ensemble_dynamics_configuration=edc,
        life_cycle_configuration=lcc,
        scenario_controller_configuration=scc,
        provider_configuration=pc
    )


def canonic_life_cycle_configuration() -> ComplexLifecycleConfiguration:
    ep = _EssentialParameters(
        motor_dimension=4,
        sensory_dimension=4,
        granularity=4,
        natural_frequency=5.,
        transgait_window_size=1,
        integration_step_size=0.01,
        rbf_epsilon=1.
    )
    rp = LifeCycleRunParameters(
        babble_iters=20000,
        delta_search_iters=20000,
        performance_measure_iters=1000,
        babbling_rate=0.2,
        model_builder=lambda: R.StdNormedLinearRegressor()
    )

    babbler_param = BabblerParameterization.DYNAMIC_BABBLING
    babbling_args = {"weight_learning_rate": 0.}
    controller_search_strategy = ControllerType.WAVE_FEP_FUSION
    controller_search_args = {"estimation_learning_rate": 0.01, "gait_learning_rate": 0.01,
                              "likelihood_variance_learning_rate": 0.01, "observed_sensory_variance": 0.01,
                              "estimation_prior_variance": np.inf, "prediction_variance_lower_bound": .1,
                              "log_prior_strength": 0.0001, "amplitude_symmetry_strength": 0.1,
                              "target_error_variance_learning_rate": 0.01, "prediction_precision_scale": 1.,
                              "target_precision_scale": 1., "target_error_processing": 1,
                              "simplified_gradient_switch": False
                              }

    return ComplexLifecycleConfiguration(
        ep=ep, rp=rp, babbler_param=babbler_param, babbler_args=babbling_args,
        controller_search_strategy=controller_search_strategy, controller_search_args=controller_search_args,
    )


class DLCSelectionVariant(Enum):
    ONE_MODEL = 1
    LAST_MODEL = 2
    REACTIVE = 3


class DLCLearningDecisionVariant(Enum):
    PURE_SCHEDULED = 1
    PURE_REACTIVE = 2
    REACTIVE_SCHEDULED = 3


class DLCCompositionVariant(Enum):
    LEAVE_ORIGINAL = 0
    ENSEMBLED = 1
    MODALITY_PHASE_WISE = 2


DLC_VARIANTS = [
    (DLCSelectionVariant.ONE_MODEL, DLCLearningDecisionVariant.PURE_SCHEDULED, DLCCompositionVariant.LEAVE_ORIGINAL),
    (DLCSelectionVariant.LAST_MODEL, DLCLearningDecisionVariant.PURE_SCHEDULED, DLCCompositionVariant.LEAVE_ORIGINAL),
    (DLCSelectionVariant.LAST_MODEL, DLCLearningDecisionVariant.PURE_REACTIVE, DLCCompositionVariant.LEAVE_ORIGINAL),
    (DLCSelectionVariant.REACTIVE, DLCLearningDecisionVariant.PURE_REACTIVE, DLCCompositionVariant.LEAVE_ORIGINAL),
    # (DLCSelectionVariant.REACTIVE, DLCLearningDecisionVariant.REACTIVE_SCHEDULED, DLCCompositionVariant.ENSEMBLED),
    # (DLCSelectionVariant.REACTIVE, DLCLearningDecisionVariant.REACTIVE_SCHEDULED, DLCCompositionVariant.MODALITY_PHASE_WISE),
]


def switch_dynamic_life_cycle_variant(
        config: DynamicLifeCycleConfiguration,
        selection: DLCSelectionVariant,
        learning_decision: DLCLearningDecisionVariant,
        composition: DLCCompositionVariant) -> DynamicLifeCycleConfiguration:
    """
    Complex parametrization setter which turns off or keeps functionalities depending to given variants.
    Used for ablation study. This function should be terminal for the config
    (i.e. the returned config should be treated as read-only).

    @return:
    @rtype:
    """
    if selection is DLCSelectionVariant.ONE_MODEL:
        # keep the same model
        # turn off the TTL
        # turn off the competition against zero-model
        return config.set_force_keep_same_model(True).max_performing_periods(np.inf).min_confident_elements_rate(-.1)

    ##
    # Selection variant
    if selection is DLCSelectionVariant.LAST_MODEL:
        config = config.set_force_keep_same_model(True)
    elif selection is DLCSelectionVariant.REACTIVE:
        config = config.set_force_keep_same_model(False)
    else:
        raise NotImplemented("Such selection variant does not exist.")

    # Learning decision variant
    if learning_decision is DLCLearningDecisionVariant.PURE_REACTIVE:
        # turn off the TTL
        config = config.max_performing_periods(periods=np.inf)
    elif learning_decision is DLCLearningDecisionVariant.PURE_SCHEDULED:
        # turn off the competition against zero-model
        config = config.min_confident_elements_rate(-.1)
    elif learning_decision is DLCLearningDecisionVariant.REACTIVE_SCHEDULED:
        # keep the parameters as they are
        pass
    else:
        raise NotImplemented("Such learning variant does not exist.")

    # Composition variant
    if composition is DLCCompositionVariant.ENSEMBLED:
        config = config.set_ensemble_strategy(EnsembleDynamicsType.TWO_STAGE_AGGREGATE)
    elif composition is DLCCompositionVariant.MODALITY_PHASE_WISE:
        config = config.set_ensemble_strategy(EnsembleDynamicsType.TWO_STAGE_AGGREGATE_COMPOSITION)
    elif composition is DLCCompositionVariant.LEAVE_ORIGINAL:
        pass

    return config

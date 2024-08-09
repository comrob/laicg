import copy
from coupling_evol.assembler import ProviderConfiguration
from experiment_helpers import configuration_build as C
from experiment_helpers.configuration_build import GoalType, EnvironmentType
from coupling_evol.engine.environment import Environment
from typing import Dict
import os
import re
STEP_SLEEP = 0.01
GAIT_ANGULAR_VELOCITY = 6.

MOTOR_DIMENSION = 18
EFFORT_IN_RAW = True

if EFFORT_IN_RAW is True:
    SENSORY_DIMENSION = 23
else:
    SENSORY_DIMENSION = 5

EXP_ROOT_PATH = os.path.join("results", "simulation_data")

TRIAL_MATCH = re.compile(r'_r(.+)$')
LIFECYCLE_MATCH = re.compile(r'_lc(.+)_env')
SCENARIO_MATCH = re.compile(r'_trg(.+)_r')
ENVIRONMENT_MATCH = re.compile(r'_env(.+)_trg')

DATA_OUTPUT_DICT = os.path.join("results", "simulation_data")


# if not os.path.exists(DATA_OUTPUT_DICT):
#     os.mkdir(DATA_OUTPUT_DICT)

def create_provider_configuration(goal_type: GoalType, target_args: dict):
    from coupling_evol.assembler import ProviderType

    pc = ProviderConfiguration()

    pc.arguments["max_linear_velocity"] = 2.
    pc.arguments["max_turn_velocity"] = 2.
    pc.arguments["zero_collapse_epsilon"] = 0.1

    for k in target_args:
        pc.arguments[k] = target_args[k]

    pc.arguments["vel_w"] = target_args["vel_weight"]
    pc.arguments["ang_z_w"] = target_args["rotz_weight"]
    pc.arguments["ang_w"] = target_args["rotxy_weight"]
    pc.arguments["eff_w"] = target_args["eff_weight"]

    if goal_type is GoalType.NAVIGATION_TRANSLATE:
        pc.created_type = ProviderType.NAVIGATE_TRANSLATE
    elif goal_type is GoalType.NAVIGATION_TURN_AND_GO:
        pc.created_type = ProviderType.NAVIGATE_TURN_AND_GO
        pc.arguments["goal_x"] = target_args["goal_x"]
        pc.arguments["goal_y"] = target_args["goal_y"]
    elif goal_type is GoalType.NAVIGATION_TURN_AND_GO_BOUNDED:
        pc.created_type = ProviderType.NAVIGATE_TURN_AND_GO
        pc.arguments["goal_x"] = target_args["goal_x"]
        pc.arguments["goal_y"] = target_args["goal_y"]

    return pc


def get_environment(env_type: EnvironmentType, args) -> Environment:
    if env_type is EnvironmentType.DUMMY:
        from coupling_evol.engine.environment import DummyEnvironment
        return DummyEnvironment(u_dim=MOTOR_DIMENSION, y_dim=SENSORY_DIMENSION)

    elif env_type is EnvironmentType.SIMULATION:
        from coupling_evol.environments.coppeliasim import coppeliasim_environment as SIM_ENV
        return SIM_ENV.CoppeliaSimEnvironment(ampl_min=[-0.32, 0.2, -0.2], ampl_max=[0.32, 0.7, 0.3])
    else:
        raise NotImplemented(f"Environment {env_type} option is not implemented.")


ENVIRONMENT_CONFIGS = {
    # "dummy": (EnvironmentType.DUMMY, {}),
    "Vrep": (EnvironmentType.SIMULATION, {})
}


# 0.05
DYNAMIC_LIFECYCLE_CONFIGS_SUBMODES = {
    "232410-sumbodes-econ14+3-msevalt10-30-lr5+3":
        C.canonic_life_cycle_configuration().
        motor_sens_dim(motor_dim=MOTOR_DIMENSION, sens_dim=SENSORY_DIMENSION).
        amplitude_symmetry_strength(0.01).
        amplitude_energy_regularization(sum_amplitude_energy=9, regularization_strength=1.).
        scale_fep_fusion_learning_granularity(.5).variance_precision_scale(target=0, prediction=10).
        babble_iters(30000).control_iters(5000).
        scale_fep_fusion_sensory_to_prediction_variance(scale=10.).log_prior_strength(weight=1).
        scale_learning_rates(scale=1.).babble_scale(.2).scale_gait_learning_rate(0.1).set_dynamic_lifecycle_config(
            C.DynamicLifeCycleConfiguration.init(C.LifeCycleType.MODEL_COMPETITION_DRIVEN).
            zero_model_flip_threshold_in(6, 10).zero_model_standard_deviation(9.).min_confident_elements_rate(0.014)
            .zero_neighbourhood_eps(0.01).set_ensemble_strategy(C.EnsembleDynamicsType.SUBMODES)
            .ensemble_dynamics_score_lr(0.005).log_score_combination(is_log=True)
            .set_performance_babble_rate(0.01).max_performing_periods(10000).model_selection_evaluation_time(10, 30)
        ).set_step_sleep(STEP_SLEEP).set_gait_natural_angvel(GAIT_ANGULAR_VELOCITY),
}

DYNAMIC_LIFECYCLE_CONFIGS_CHOSEN = {
    # "232410-econ7+1-odomsum-pri1-sclr5+3-ampe9-1":
    #     C.switch_to_wave_fep_fusion(C.canonic_life_cycle_configuration()).
    #     motor_sens_dim(motor_dim=MOTOR_DIMENSION, sens_dim=SENSORY_DIMENSION).granularity(4).
    #     amplitude_symmetry_strength(0.01).
    #     amplitude_energy_regularization(sum_amplitude_energy=9, regularization_strength=1.).
    #     scale_fep_fusion_learning_granularity(.5).variance_precision_scale(target=0, prediction=10).
    #     babble_iters(30000).control_iters(5000).
    #     scale_fep_fusion_sensory_to_prediction_variance(scale=10.).log_prior_strength(weight=1).
    #     scale_learning_rates(scale=1.).babble_scale(.2).scale_gait_learning_rate(0.1).set_dynamic_lifecycle_config(
    #         C.DynamicLifeCycleConfiguration.init(C.LifeCycleType.MODEL_COMPETITION_DRIVEN).
    #         zero_model_flip_threshold_in(6, 10).zero_model_standard_deviation(9.).min_confident_elements_rate(0.7)
    #         .zero_neighbourhood_eps(0.01).set_ensemble_strategy(C.EnsembleDynamicsType.TWO_STAGE_AGGREGATE_ODOMETRYSUM)
    #         .ensemble_dynamics_score_lr(0.005).log_score_combination(is_log=True)
    #         .set_performance_babble_rate(0.01).max_performing_periods(1000)
    #     ),
    "231129-econ22+3-sclr5+3-msevalt10-20-zeps43+10-dlca":
        C.canonic_life_cycle_configuration().
        motor_sens_dim(motor_dim=MOTOR_DIMENSION, sens_dim=SENSORY_DIMENSION).
        amplitude_symmetry_strength(0.01).
        amplitude_energy_regularization(sum_amplitude_energy=9, regularization_strength=1.).
        scale_fep_fusion_learning_granularity(.5).variance_precision_scale(target=0, prediction=10).
        babble_iters(30000).control_iters(5000).
        scale_fep_fusion_sensory_to_prediction_variance(scale=10.).log_prior_strength(weight=1).
        scale_learning_rates(scale=1.).babble_scale(.2).scale_gait_learning_rate(0.1).set_dynamic_lifecycle_config(
            C.DynamicLifeCycleConfiguration.init(C.LifeCycleType.MODEL_COMPETITION_DRIVEN).
            zero_model_flip_threshold_in(6, 10).zero_model_standard_deviation(9.).min_confident_elements_rate(0.022)
            .zero_neighbourhood_eps(0.43).set_ensemble_strategy(C.EnsembleDynamicsType.TWO_STAGE_AGGREGATE_ODOMETRYSUM)
            .ensemble_dynamics_score_lr(0.005).log_score_combination(is_log=True).direct_confidence_score(
                is_direct_confidence=True)
            .set_performance_babble_rate(0.01).max_performing_periods(100000).model_selection_evaluation_time(10, 20)
        ).set_step_sleep(STEP_SLEEP).set_gait_natural_angvel(GAIT_ANGULAR_VELOCITY).set_target_error_processing(typ=2),
}



NAVIGATING_TARGET_FARGOAL = {
    "Trngo-gxy-100--100-mv-8-4":
        (GoalType.NAVIGATION_TURN_AND_GO,
         {"max_linear_velocity": 8, "max_turn_velocity": 4, "vel_max_val": 2, "vel_weight": 30, "rotz_weight": 30,
          "rotxy_weight": 2, "eff_weight": 1, "goal_x": -100, "goal_y": -100,
          "vel_side_w": 30
          }, C.ScenarioBuild.init_default().
         set_scenario_controller_type(C.ScenarioControllerType.TIMED_PARALYZE_THIRD_LEG).
         switch_scenario_at_fractions(switch_fraction=.999, switch_back_fraction=1.)),
}

NAVIGATING_TARGET_TIMED_SINGLE_PARALYSIS_FARGOAL = {
    # "Trngo-gxy-100--100-mv-8-4-LegPar3Swf-5+1-9+1":
    # (GoalType.NAVIGATION_TURN_AND_GO,
    #     {"max_linear_velocity": 8, "max_turn_velocity": 4, "vel_max_val": 2, "vel_weight": 30, "rotz_weight": 30, "rotxy_weight": 2, "eff_weight": 1, "goal_x": -100, "goal_y": -100,
    #     "vel_side_w": 30
    #     }, C.ScenarioBuild.init_default().
    #      set_scenario_controller_type(C.ScenarioControllerType.TIMED_PARALYZE_THIRD_LEG).
    #      switch_scenario_at_fractions(switch_fraction=0.5, switch_back_fraction=.9)),
    # "Trngo-gxy-100--100-mv-8-4-LegPar1Swf-5+1-9+1":
    # (GoalType.NAVIGATION_TURN_AND_GO,
    #     {"max_linear_velocity": 8, "max_turn_velocity": 4, "vel_max_val": 2, "vel_weight": 30, "rotz_weight": 30, "rotxy_weight": 2, "eff_weight": 1, "goal_x": -100, "goal_y": -100,
    #     "vel_side_w": 30
    #     }, C.ScenarioBuild.init_default().
    #      set_scenario_controller_type(C.ScenarioControllerType.TIMED_INVERT_SENSORS).
    #      switch_scenario_at_fractions(switch_fraction=0.5, switch_back_fraction=.9)),
    # "Trngo-gxy-100--100-mv-8-4-LegInv3Swf-3+1-7+1":
    # (GoalType.NAVIGATION_TURN_AND_GO,
    #     {"max_linear_velocity": 8, "max_turn_velocity": 4, "vel_max_val": 2, "vel_weight": 30, "rotz_weight": 30, "rotxy_weight": 2, "eff_weight": 1, "goal_x": -100, "goal_y": -100,
    #     "vel_side_w": 30
    #     }, C.ScenarioBuild.init_default().
    #      set_scenario_controller_type(C.ScenarioControllerType.TIMED_INVERT_FIRST_LEG).
    #      switch_scenario_at_fractions(switch_fraction=0.3, switch_back_fraction=.7)),
    "Trngo-gxy-100--100-mv-8-4-LegPar3Swf-1+1-7+1":
        (GoalType.NAVIGATION_TURN_AND_GO,
         {"max_linear_velocity": 8, "max_turn_velocity": 4, "vel_max_val": 2, "vel_weight": 30, "rotz_weight": 30,
          "rotxy_weight": 2, "eff_weight": 1, "goal_x": -100, "goal_y": -100,
          "vel_side_w": 30
          }, C.ScenarioBuild.init_default().
         set_scenario_controller_type(C.ScenarioControllerType.TIMED_PARALYZE_THIRD_LEG).
         switch_scenario_at_fractions(switch_fraction=.1, switch_back_fraction=.7)),
    # "Trngo-gxy-100--100-mv-8-4-LegPar3Swf-9+1-1":
    # (GoalType.NAVIGATION_TURN_AND_GO,
    #     {"max_linear_velocity": 8, "max_turn_velocity": 4, "vel_max_val": 2, "vel_weight": 30, "rotz_weight": 30, "rotxy_weight": 2, "eff_weight": 1, "goal_x": -100, "goal_y": -100,
    #     "vel_side_w": 30
    #     }, C.ScenarioBuild.init_default().
    #      set_scenario_controller_type(C.ScenarioControllerType.TIMED_PARALYZE_THIRD_LEG).
    #      switch_scenario_at_fractions(switch_fraction=.9, switch_back_fraction=.99)),
    # "Trngo-gxy-100--100-mv-8-4-LegPar5Swf-3+1-7+1":
    # (GoalType.NAVIGATION_TURN_AND_GO,
    #     {"max_linear_velocity": 8, "max_turn_velocity": 4, "vel_max_val": 2, "vel_weight": 30, "rotz_weight": 30, "rotxy_weight": 2, "eff_weight": 1, "goal_x": -100, "goal_y": -100,
    #     "vel_side_w": 30
    #     }, C.ScenarioBuild.init_default().
    #      set_scenario_controller_type(C.ScenarioControllerType.TIMED_PARALYZE_FIFTH_LEG).
    #      switch_scenario_at_fractions(switch_fraction=0.3, switch_back_fraction=.7)),
    # "Trngo-gxy-100--100-mv-8-4-LegPar1Swf-1+1-7+1":
    # (GoalType.NAVIGATION_TURN_AND_GO,
    #     {"max_linear_velocity": 8, "max_turn_velocity": 4, "vel_max_val": 2, "vel_weight": 30, "rotz_weight": 30, "rotxy_weight": 2, "eff_weight": 1, "goal_x": -100, "goal_y": -100,
    #     "vel_side_w": 30
    #     }, C.ScenarioBuild.init_default().
    #      set_scenario_controller_type(C.ScenarioControllerType.TIMED_INVERT_FIRST_LEG).
    #      switch_scenario_at_fractions(switch_fraction=0.1, switch_back_fraction=.7)),
}

NAVIGATING_TARGET_TIMED_DOUBLE_PARALYSIS_FARGOAL = {
    # "Trngo-gxy-100--100-mv-8-4-LegPar14Swf-1+1-7+1":
    # (GoalType.NAVIGATION_TURN_AND_GO,
    #     {"max_linear_velocity": 8, "max_turn_velocity": 4, "vel_max_val": 2, "vel_weight": 30, "rotz_weight": 30, "rotxy_weight": 2, "eff_weight": 1, "goal_x": -100, "goal_y": -100,
    #     "vel_side_w": 30
    #     }, C.ScenarioBuild.init_default().
    #      set_scenario_controller_type(C.ScenarioControllerType.TIMED_PARALYZE_FIRST_FOURTH_LEGS).
    #      switch_scenario_at_fractions(switch_fraction=0.1, switch_back_fraction=.7)),
    # "Trngo-gxy-100--100-mv-8-4-LegPar16Swf-1+1-7+1":
    # (GoalType.NAVIGATION_TURN_AND_GO,
    #     {"max_linear_velocity": 8, "max_turn_velocity": 4, "vel_max_val": 2, "vel_weight": 30, "rotz_weight": 30, "rotxy_weight": 2, "eff_weight": 1, "goal_x": -100, "goal_y": -100,
    #     "vel_side_w": 30
    #     }, C.ScenarioBuild.init_default().
    #      set_scenario_controller_type(C.ScenarioControllerType.TIMED_PARALYZE_FIRST_SIXTH_LEGS).
    #      switch_scenario_at_fractions(switch_fraction=0.1, switch_back_fraction=.7)),
    # "Trngo-gxy-100--100-mv-8-4-LegPar36Swf-3+1-7+1":
    # (GoalType.NAVIGATION_TURN_AND_GO,
    #     {"max_linear_velocity": 8, "max_turn_velocity": 4, "vel_max_val": 2, "vel_weight": 30, "rotz_weight": 30, "rotxy_weight": 2, "eff_weight": 1, "goal_x": -100, "goal_y": -100,
    #     "vel_side_w": 30
    #     }, C.ScenarioBuild.init_default().
    #      set_scenario_controller_type(C.ScenarioControllerType.TIMED_PARALYZE_THIRD_SIXTH_LEGS).
    #      switch_scenario_at_fractions(switch_fraction=0.3, switch_back_fraction=.7)),
    # "Trngob-gxy-100--100-mv-8-4-LegPar14Swf-1+1-7+1":
    #     (GoalType.NAVIGATION_TURN_AND_GO_BOUNDED,
    #      {"max_linear_velocity": 8, "max_turn_velocity": 4, "vel_max_val": 2, "vel_weight": 30, "rotz_weight": 30,
    #       "rotxy_weight": 2, "eff_weight": 1, "goal_x": -100, "goal_y": -100,
    #       "vel_side_w": 30
    #       }, C.ScenarioBuild.init_default().
    #      set_scenario_controller_type(C.ScenarioControllerType.TIMED_PARALYZE_FIRST_FOURTH_LEGS).
    #      switch_scenario_at_fractions(switch_fraction=0.1, switch_back_fraction=.7)),
    "Trngob-gxy-100--100-mv-8-4-LegPar36Swf-1+1-7+1":
        (GoalType.NAVIGATION_TURN_AND_GO_BOUNDED,
         {"max_linear_velocity": 8, "max_turn_velocity": 4, "vel_max_val": 2, "vel_weight": 30, "rotz_weight": 30,
          "rotxy_weight": 2, "eff_weight": 1, "goal_x": -100, "goal_y": -100,
          "vel_side_w": 30
          }, C.ScenarioBuild.init_default().
         set_scenario_controller_type(C.ScenarioControllerType.TIMED_PARALYZE_THIRD_SIXTH_LEGS).
         switch_scenario_at_fractions(switch_fraction=0.1, switch_back_fraction=.7)),
}

LIFECYCLE_CONFIGS = DYNAMIC_LIFECYCLE_CONFIGS_CHOSEN
# LIFECYCLE_CONFIGS = DYNAMIC_LIFECYCLE_CONFIGS_SUBMODES
##

# LIFECYCLE_CONFIGS = variate_dlcs_in_configs(LIFECYCLE_CONFIGS)
# LIFECYCLE_CONFIGS = switch_to_single_model(LIFECYCLE_CONFIGS)
LIFECYCLE_CONFIGS = C.switch_to_reactive_select_model(LIFECYCLE_CONFIGS)
# TARGET_CONFIGS = NAVIGATING_TARGET_FARGOAL
TARGET_CONFIGS = NAVIGATING_TARGET_TIMED_DOUBLE_PARALYSIS_FARGOAL

ENVIRONMENT_CONFIGS = ENVIRONMENT_CONFIGS

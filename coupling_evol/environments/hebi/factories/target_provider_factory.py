from coupling_evol.engine.experiment_executor import TargetProvider
from coupling_evol.assembler.environment_symbols import ProviderType, ProviderConfiguration
from coupling_evol.environments.hebi.target_providers import realsense_target_provider
from coupling_evol.assembler.common import *


def factory(config: ProviderConfiguration,  essentials: EssentialParameters, lifecycle: LifeCycleParameters) -> TargetProvider:
    strategy = config.created_type
    sensory_dimension = essentials.sensory_dimension
    granularity = lifecycle.granularity
    args = config.arguments

    x_goal, y_goal = (0, 0)
    if "goal_x" in args:
        x_goal = args["goal_x"]
    if "goal_y" in args:
        y_goal = args["goal_y"]

    if strategy == ProviderType.NAVIGATE_TURN_AND_GO:

        return realsense_target_provider.TurnAndGo(
            sensory_dim=sensory_dimension, granularity=granularity, xy_goal=(x_goal, y_goal),
            max_linear_velocity=args["max_linear_velocity"], max_turn_velocity=args["max_turn_velocity"],
            zero_collapse_epsilon=args["zero_collapse_epsilon"],
            vel_w=args["vel_w"], ang_z_w=args["ang_z_w"], ang_w=args["ang_w"], eff_w=args["eff_w"]
        )
    if strategy == ProviderType.NAVIGATE_TRANSLATE:
        return realsense_target_provider.Translate(
            sensory_dim=sensory_dimension, granularity=granularity, xy_goal=(x_goal, y_goal),
            max_linear_velocity=args["max_linear_velocity"], max_turn_velocity=args["max_turn_velocity"],
            zero_collapse_epsilon=args["zero_collapse_epsilon"],
            vel_w=args["vel_w"], ang_z_w=args["ang_z_w"], ang_w=args["ang_w"], eff_w=args["eff_w"]
        )
    if strategy == ProviderType.NAVIGATE_BOUNDED_TURN_AND_GO:
        return realsense_target_provider.TurnAndGo(
            sensory_dim=sensory_dimension, granularity=granularity, xy_goal=(x_goal, y_goal),
            max_linear_velocity=args["max_linear_velocity"], max_turn_velocity=args["max_turn_velocity"],
            zero_collapse_epsilon=args["zero_collapse_epsilon"],
            vel_w=args["vel_w"], ang_z_w=args["ang_z_w"], ang_w=args["ang_w"], eff_w=args["eff_w"]
        )
    raise NotImplemented("Given target provider strategy is not implemented.")

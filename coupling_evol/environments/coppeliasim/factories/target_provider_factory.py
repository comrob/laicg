from coupling_evol.engine.experiment_executor import TargetProvider
from coupling_evol.assembler.environment_symbols import ProviderType, ProviderConfiguration
from coupling_evol.assembler.common import *
from coupling_evol.environments.coppeliasim.target_providers import navigation as COP_NAV
import logging

LOG = logging.getLogger(__name__)


def factory(config: ProviderConfiguration, essentials: EssentialParameters, lifecycle: LifeCycleParameters) -> TargetProvider:
    args = config.arguments
    LOG.info(f"Coppelia Sim Target factory: startegy {config.created_type} args {args}")
    if config.created_type == ProviderType.NAVIGATE_TURN_AND_GO:
        x_goal, y_goal = (0, 0)
        if "goal_x" in args:
            x_goal = args["goal_x"]
        if "goal_y" in args:
            y_goal = args["goal_y"]
        if "vel_side_w" in args:
            vel_side_w = args["vel_side_w"]
        else:
            vel_side_w = args["vel_w"]
        return COP_NAV.TurnAndGo(
            sensory_dim=essentials.sensory_dimension,
            granularity=lifecycle.granularity,
            xy_goal=(x_goal, y_goal),
            max_turn_velocity=args["max_turn_velocity"],
            max_linear_velocity=args["max_linear_velocity"],
            zero_collapse_epsilon=args["zero_collapse_epsilon"],
            vel_w=args["vel_w"], ang_w=args["ang_w"], eff_w=args["eff_w"],
            ang_z_w=args["ang_z_w"], vel_side_w=vel_side_w
        )
    elif config.created_type == ProviderType.NAVIGATE_BOUNDED_TURN_AND_GO:
        x_goal, y_goal = (0, 0)
        if "goal_x" in args:
            x_goal = args["goal_x"]
        if "goal_y" in args:
            y_goal = args["goal_y"]
        if "vel_side_w" in args:
            vel_side_w = args["vel_side_w"]
        else:
            vel_side_w = args["vel_w"]
        return COP_NAV.BoundedTurnAndGo(
            sensory_dim=essentials.sensory_dimension,
            granularity=lifecycle.granularity,
            xy_goal=(x_goal, y_goal),
            max_turn_velocity=args["max_turn_velocity"],
            max_linear_velocity=args["max_linear_velocity"],
            zero_collapse_epsilon=args["zero_collapse_epsilon"],
            vel_w=args["vel_w"], ang_w=args["ang_w"], eff_w=args["eff_w"],
            ang_z_w=args["ang_z_w"], vel_side_w=vel_side_w
        )
    else:
        raise NotImplemented(f"Coppelia sim target provider for {config.created_type} is not implemented.")

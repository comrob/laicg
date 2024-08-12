import logging

import coupling_evol.engine.experiment_executor as EE
from coupling_evol.environments.coppeliasim.coppeliasim_environment import CoppeliaSimEnvironment
from coupling_evol.environments.hebi.hebi_environment import HebiEnvironment
from coupling_evol.environments.coppeliasim.data_pipes import CoppeliaPipe
from coupling_evol.environments.hebi.data_pipes import HebiPipeWithEffort
from coupling_evol.engine.environment import Environment
from coupling_evol.engine.datapipes import universal
from coupling_evol.assembler.scenario import scenario_controller_factory as SCF
from coupling_evol.environments.coppeliasim.factories import target_provider_factory as TPF_COPSIM
from coupling_evol.environments.hebi.factories import target_provider_factory as TPF_HEBI
from coupling_evol.assembler import *

LOG = logging.getLogger(__name__)


class Scenario:
    def __init__(self,
                 target_provider: EE.TargetProvider,
                 scenario_controller: EE.ScenarioController,
                 data_pipe: EE.DataPipe,
                 environment: Environment
                 ):
        self.target_provider = target_provider
        self.scenario_controller = scenario_controller
        self.data_pipe = data_pipe
        self.environment = environment


def create_simple_scenario_from(
        environment: Environment,
        target_provider: EE.TargetProvider,
        scenario_controller: EE.ScenarioController
) -> Scenario:
    if isinstance(environment, CoppeliaSimEnvironment):
        LOG.info("Setting CoppeliaPipe for the datapipe.")
        data_pipe = CoppeliaPipe()
    elif isinstance(environment, HebiEnvironment):
        LOG.info("Setting HebiPipe for the datapipe.")
        data_pipe = HebiPipeWithEffort()
    else:
        LOG.info("Setting Universal Forwarder for the datapipe.")
        data_pipe = universal.Forwarder()

    return Scenario(
        data_pipe=data_pipe,
        scenario_controller=scenario_controller,
        target_provider=target_provider,
        environment=environment
    )


def create(target_config: ProviderConfiguration,
           scenario_controller_config: ScenarioControllerConfiguration,
           essential: EssentialParameters,
           experiment_setup: ExperimentSetupParameters,
           lifecycle: LifeCycleParameters,
           environment: Environment):

    scenario_controller = SCF.factory(scenario_controller_config, essential, experiment_setup)
    if isinstance(environment, CoppeliaSimEnvironment):
        # the non legacy creation
        LOG.info("Setting up TURN AND GO goal for COPPELIA SIM.")
        return create_simple_scenario_from(
            environment,
            target_provider=TPF_COPSIM.factory(target_config, essential, lifecycle),
            scenario_controller=scenario_controller
        )
    else:
        """
        Legacy target provider (originally made for hebi+realsense) with simple scenario  
        """
        return create_simple_scenario_from(
            environment,
            target_provider=TPF_HEBI.factory(target_config, essential, lifecycle),
            scenario_controller=scenario_controller
        )

from enum import Enum
from typing import Dict, Union, Generic, TypeVar

ARG_TYPE = Dict[str, Union[int, float]]

CREATED_T = TypeVar('CREATED_T', bound=Enum)


class EssentialParameters:
    def __init__(self, motor_dimension: int, sensory_dimension: int, integration_step_size: float):
        self.motor_dimension = motor_dimension
        self.sensory_dimension = sensory_dimension
        self.integration_step_size = integration_step_size


class ExperimentSetupParameters:
    def __init__(self, max_iters, step_sleep):
        self.max_iters = max_iters
        self.step_sleep = step_sleep

    def set_max_iters_from_context_steps(self, ctx_num, babble_iters, performance_iters):
        self.max_iters = (babble_iters + performance_iters + 2) * ctx_num + 1
        return self


class LifeCycleParameters:
    def __init__(self, granularity):
        self.granularity: int = granularity
        self.force_keep_same_model = False


class FactoryConfiguration(Generic[CREATED_T]):
    def __init__(self):
        self.arguments: ARG_TYPE = {}
        self.created_type: CREATED_T

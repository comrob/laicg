from enum import Enum
import coupling_evol.agent.components.controllers.motor_babbling as MB
import numpy as np


class BabblerParameterization(Enum):
    GAIT_NOISE_NORMAL_SIXTH = 1
    PHASE_NOISE_NORMAL_SIXTH = 2
    THREE_PHASE_EV3_NOISE_NORMAL = 3  # max three phase segments, gait picked with prob 1/3
    ALL_PHASE_EV3_NOISE_NORMAL = 4  # max all phase segments, gait picked with prob 1/3
    DYNAMIC_BABBLING = 5


def get_parametrized_babbler(bp: BabblerParameterization, motor_dimension,
                             granularity, **kwargs) -> MB.MotorBabbler:
    if bp == BabblerParameterization.GAIT_NOISE_NORMAL_SIXTH:
        babble_generator = lambda: np.random.randn(motor_dimension) / 6
        motor_babbler = MB.MotorBabbler(
            motor_dim=motor_dimension,
            granularity=granularity,
            babble_generator=babble_generator
        )
        return motor_babbler
    elif bp == BabblerParameterization.PHASE_NOISE_NORMAL_SIXTH:
        babble_generator = lambda: np.random.randn(motor_dimension) / 6
        motor_babbler = MB.MotorBabblerConservative(
            motor_dim=motor_dimension,
            granularity=granularity,
            babble_generator=babble_generator
        )
        return motor_babbler
    elif bp == BabblerParameterization.THREE_PHASE_EV3_NOISE_NORMAL:
        babble_generator = lambda: np.random.randn(motor_dimension) / 6
        motor_babbler = MB.MotorBabblerConservative(
            motor_dim=motor_dimension,
            granularity=granularity,
            babble_generator=babble_generator,
            max_random_segments=3
        )
        return motor_babbler
    elif bp == BabblerParameterization.ALL_PHASE_EV3_NOISE_NORMAL:
        babble_generator = lambda: np.random.randn(motor_dimension) / 6
        motor_babbler = MB.MotorBabblerConservative(
            motor_dim=motor_dimension,
            granularity=granularity,
            babble_generator=babble_generator,
            max_random_segments=granularity,
        )
        return motor_babbler
    elif bp == BabblerParameterization.DYNAMIC_BABBLING:
        babble_generator = lambda: np.random.randn(motor_dimension)
        motor_babbler = MB.DynamicMotorBabbler(
            motor_dim=motor_dimension,
            granularity=granularity,
            babble_generator=babble_generator,
            # learning_rate=kwargs["weight_learning_rate"], # FIXME create proper config
            learning_rate=0.
        )
        return motor_babbler
    else:
        raise NotImplemented("Given babbler is nonexistent.")

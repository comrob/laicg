from enum import Enum
from typing import Callable, List, Union, Tuple
import numpy as np


class RecordType(Enum):
    NUMPY = 0


class RecordNamespace(Enum):
    EXECUTOR = "exc",
    LIFE_CYCLE = "lcy",
    TARGET_PROVIDER = "trg",
    SCENARIO_CONTROLLER = "sct",
    SENSORY_SOURCE = "sns",

    @property
    def key(self) -> str:
        return self.value[0]

    def __call__(self, s: str):
        return self.key + "_" + s


class CommandType(Enum):
    DIRECT = 0,
    POSITION_ZERO = 1,


LOGGER_T = Callable[[RecordType, str, Union[np.ndarray, float]], None]


def none_logger(record_type: RecordType, key: str, value: Union[np.ndarray, float]):
    pass


COMMAND_T = Tuple[np.ndarray, CommandType]

EMPTY_OBSERVATION = np.empty((0,))


def is_empty_observation(observation: np.ndarray):
    return len(observation) == 0

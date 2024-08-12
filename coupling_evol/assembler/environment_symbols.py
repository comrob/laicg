from coupling_evol.assembler.common import *


class ProviderType(Enum):
    NAVIGATE_TURN_AND_GO = 1
    NAVIGATE_TRANSLATE = 2
    NAVIGATE_BOUNDED_TURN_AND_GO = 3


class EnvironmentType(Enum):
    DUMMY = 0
    SIMULATION = 1
    HEBI = 2


class ProviderConfiguration(FactoryConfiguration[ProviderType]):
    def __init__(self):
        super().__init__()
        self.arguments = {}
        self.created_type: ProviderType = ProviderType.NAVIGATE_TURN_AND_GO

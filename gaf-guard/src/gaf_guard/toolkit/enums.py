from enum import Enum, StrEnum, auto


class CustomStrEnum(StrEnum):
    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name


class MessageType(CustomStrEnum):
    WORKFLOW_INPUT = auto()
    WORKFLOW_STARTED = auto()
    WORKFLOW_COMPLETED = auto()
    STEP_STARTED = auto()
    STEP_COMPLETED = auto()
    STEP_DATA = auto()
    HITL_QUERY = auto()
    HITL_RESPONSE = auto()


class Role(StrEnum):
    USER = auto()
    AGENT = auto()
    SYSTEM = auto()


class Serializer(Enum):
    YAML = auto()
    JSON = auto()

from enum import StrEnum, auto


class MessageType(StrEnum):
    WORKFLOW_INPUT = auto()
    WORKFLOW_STARTED = auto()
    WORKFLOW_COMPLETED = auto()
    STEP_STARTED = auto()
    STEP_COMPLETED = auto()
    STEP_DATA = auto()
    HITL_QUERY = auto()
    HITL_RESPONSE = auto()


class Role(StrEnum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"

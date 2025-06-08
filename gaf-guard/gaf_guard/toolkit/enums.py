from enum import StrEnum, auto


class MessageType(StrEnum):
    USER_INTENT = auto()
    PRINT = auto()
    RULE = auto()
    INTERRUPT_QUERY = auto()
    INTERRUPT_RESPONSE = auto()


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"

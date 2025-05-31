from enum import StrEnum, auto


class MessageType(StrEnum):
    PRINT = auto()
    RULE = auto()
    COMPLETED = auto()

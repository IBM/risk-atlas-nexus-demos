from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel

from gaf_guard.toolkit.enums import MessageType, Role


class WorkflowMessage(BaseModel):

    name: str
    type: MessageType
    role: Role
    content: Optional[Any] = None
    client_id: Optional[str] = None


class WorkflowStepMessage(BaseModel):

    step_name: str
    step_type: MessageType
    step_role: Role
    step_desc: Optional[str] = None
    content: Optional[Any] = None
    step_kwargs: Dict = {}

from typing import Any, Dict, Optional

from pydantic import BaseModel

from gaf_guard.toolkit.enums import MessageType, Role


class WorkflowStepMessage(BaseModel):

    step_name: str
    step_type: MessageType
    step_role: Role
    step_desc: Optional[str] = None
    content: Optional[Any] = None
    step_kwargs: Dict = {}
    run_configs: Optional[Dict] = None

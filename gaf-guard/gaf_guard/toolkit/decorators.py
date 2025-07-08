import json
import os
from pathlib import Path
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from gaf_guard.toolkit.enums import MessageType, Role
from gaf_guard.toolkit.models import WorkflowStepMessage


def workflow_step(
    step_name: Optional[str] = None,
    step_desc: Optional[str] = None,
    step_role: Optional[str] = Role.AGENT,
    publish: bool = True,
    log: bool = False,
    **step_kwargs,
):
    def decorator(func):

        def wrapper(*args, config: RunnableConfig, **kwargs):

            write_to_stream = get_stream_writer()
            message = WorkflowStepMessage(
                step_type=MessageType.STEP_STARTED,
                step_role=step_role,
                step_name=step_name,
                step_desc=step_desc,
                step_kwargs=step_kwargs,
            )

            if publish:
                write_to_stream(message)
            event = func(*args, **kwargs, config=config)
            if publish:
                write_to_stream(
                    message.model_copy(
                        update={"step_type": MessageType.STEP_DATA, "content": event}
                    )
                )
                write_to_stream(
                    message.model_copy(update={"step_type": MessageType.STEP_COMPLETED})
                )

            if log:
                log_data(
                    message.model_copy(
                        update={"step_type": MessageType.STEP_DATA, "content": event}
                    ),
                    config["configurable"]["trial_file"],
                )

            return event

        return wrapper

    return decorator


def log_data(message: WorkflowStepMessage, trial_file: str):
    step_data = message.model_dump(include=["step_name", "step_role", "content"])

    if Path(trial_file).exists():
        trial_data = json.loads(Path(trial_file).read_text())
        trial_data.append(step_data)
    else:
        trial_data = [step_data]

    json.dump(trial_data, open(trial_file, "w"), indent=4)

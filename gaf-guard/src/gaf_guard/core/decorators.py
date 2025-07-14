import json
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress

from gaf_guard.core.models import WorkflowStepMessage
from gaf_guard.toolkit.enums import MessageType, Role


STATUS_DISPLAY = {}


def workflow(
    name: Optional[str] = None,
    desc: Optional[str] = None,
    role: Role = Role.SYSTEM,
    log_output: bool = True,
):

    def decorator(func):

        def wrapper(*args, config: RunnableConfig, **kwargs):
            client_id = config.get("configurable", {}).get("thread_id", 1)
            agent_name = args[0]._WORKFLOW_NAME

            display = STATUS_DISPLAY.setdefault(
                client_id,
                {
                    "live": Live(console=Console()),
                    "progress": Progress(),
                    "current_task": None,
                },
            )
            if display["current_task"]:
                display["progress"].update(
                    display["current_task"]["task_id"],
                    completed=100,
                    description=f"[bold yellow]Invoking Agent[/bold yellow][bold white]...{display['current_task']['name']}[/bold white][bold yellow]...Completed[/bold yellow]",
                    refresh=True,
                )

            display["current_task"] = {
                "task_id": display["progress"].add_task(
                    f"[bold yellow]Invoking Agent[/bold yellow][bold white]...{agent_name}[/bold white]",
                    total=None,
                ),
                "name": agent_name,
            }

            display["live"].start()
            display["live"].update(
                Panel(
                    Group(
                        f"Incoming request:\n{json.dumps(args[1].model_dump(include=set({'user_intent', 'prompt'}), exclude_none=True), indent=2)}\n",
                        display["progress"],
                    ),
                    title=f"{config.get('configurable', {}).get('trial_name', 'Trial_')} | Client: {client_id}",
                )
            )

            write_to_stream = get_stream_writer()
            message = WorkflowStepMessage(
                step_name=name or func.__name__,
                step_type=MessageType.WORKFLOW_STARTED,
                step_role=role,
                step_desc=desc,
                content=agent_name,
            )
            write_to_stream(
                {"client": message} | ({"logger": message} if log_output else {})
            )

            # Call the actual graph node
            event = func(*args, **kwargs, config=config)

            return event

        return wrapper

    return decorator


def workflow_step(
    step_name: Optional[str] = None,
    step_desc: Optional[str] = None,
    step_role: Role = Role.AGENT,
    log_output: bool = True,
    **step_kwargs,
):
    def decorator(func):

        def wrapper(*args, config: RunnableConfig, **kwargs):

            write_to_stream = get_stream_writer()
            message = WorkflowStepMessage(
                step_type=MessageType.STEP_STARTED,
                step_role=step_role,
                step_name=step_name or func.__name__,
                step_desc=step_desc,
                step_kwargs=step_kwargs,
            )
            write_to_stream({"client": message})

            # Call the actual graph node
            event = func(*args, **kwargs, config=config)

            event_message = message.model_copy(
                update={"step_type": MessageType.STEP_DATA, "content": event}
            )
            write_to_stream(
                {"client": event_message}
                | ({"logger": event_message} if log_output else {})
            )
            write_to_stream(
                {
                    "client": message.model_copy(
                        update={"step_type": MessageType.STEP_COMPLETED}
                    )
                }
            )

            return event

        return wrapper

    return decorator

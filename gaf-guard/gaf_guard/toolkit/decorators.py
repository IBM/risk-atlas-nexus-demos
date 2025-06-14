import asyncio
from typing import Dict, Literal, Optional

from langchain_core.runnables import RunnableConfig
from rich.console import Console

from gaf_guard.toolkit.config_utils import from_runnable_config
from gaf_guard.toolkit.conn_manager import conn_manager
from gaf_guard.toolkit.enums import Role


console = Console()


def step_logging(
    step: Optional[str],
    at: Optional[Literal["begin", "end", "both"]] = None,
    step_desc: Optional[str] = None,
    benchmark: Optional[str] = None,
    role: Optional[str] = Role.ASSISTANT,
):
    def decorator(func):

        async def wrapper(*args, **kwargs):
            if at in ["begin", "both"]:
                await conn_manager.send(
                    f"\n[bold blue]Workflow Step: [bold white]{step}[/bold white]....Started",
                )
            await conn_manager.send(step_desc)
            event = await func(*args, **kwargs)
            if benchmark:
                await conn_manager.log_benchmark(
                    event.get(benchmark, None), step, role, kwargs["config"].trial_file
                )
            await conn_manager.send(event.get("log", None) if event else None)
            if at in ["end", "both"]:
                await conn_manager.send(
                    f"[bold blue]Workflow Step: [bold white]{step}[/bold white]....Completed",
                )
            return event

        return wrapper

    return decorator


def config(config_class: str = None):
    def decorator(func):

        async def wrapper(*args, config: RunnableConfig, **kwargs):
            if config_class:
                config = from_runnable_config(config_class, config)
                event = await func(*args, **kwargs, config=config)
            else:
                event = await func(*args, **kwargs)
            return event

        return wrapper

    return decorator


def hline(title, at="start"):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if at == "begin":
                await conn_manager.send(
                    f"[bold green]{title}[/bold green]",
                    message_type="rule",
                )
            event = await func(*args, **kwargs)
            if at == "end":
                await conn_manager.send(
                    f"[bold white]{title}[/bold white]",
                    message_type="rule",
                    characters=".",
                    style="bold red",
                    spacing="both",
                )
            return event

        return wrapper

    return decorator


def async_partial(f, *args):
    async def func(*args2):
        result = f(*args, *args2)
        if asyncio.iscoroutinefunction(f):
            result = await result
        return result

    return func

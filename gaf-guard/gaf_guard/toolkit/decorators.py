from typing import Literal, Optional

from langchain_core.runnables import RunnableConfig

from gaf_guard.toolkit.enums import Role
from gaf_guard.toolkit.logging import (
    log_benchmark_data,
    log_data,
    log_end_operation,
    log_start_operation,
)


def step_logging(
    step: Optional[str],
    at: Optional[Literal["begin", "end", "both"]] = None,
    step_desc: Optional[str] = None,
    align="left",
    benchmark: Optional[str] = None,
    benchmark_role: Optional[str] = Role.ASSISTANT,
):
    def decorator(func):

        def wrapper(*args, config: RunnableConfig, **kwargs):
            if at in ["begin", "both"]:
                log_start_operation(step)
            log_data(step_desc)
            event = func(*args, **kwargs, config=config)
            log_data(event.get("log", None), message_kwargs={"justify": align})
            log_benchmark_data(
                event.get(benchmark, None), step=step, role=benchmark_role
            )
            if at in ["end", "both"]:
                log_end_operation(step)
            return event

        return wrapper

    return decorator

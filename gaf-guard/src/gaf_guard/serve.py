import logging
import os
import uuid


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import json
from collections.abc import AsyncGenerator
from datetime import datetime
from functools import reduce
from pathlib import Path

import acp_sdk
import typer
import yaml
from acp_sdk.models import (
    AnyUrl,
    Link,
    LinkType,
    Message,
    MessageAwaitRequest,
    MessagePart,
    Metadata,
)
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from langgraph.types import Command
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

import gaf_guard
from gaf_guard.config import get_configuration
from gaf_guard.core.agent_builder import AgentBuilder
from gaf_guard.core.models import WorkflowStepMessage
from gaf_guard.toolkit.enums import MessageType, Role
from gaf_guard.toolkit.exceptions import HumanInterruptionException
from gaf_guard.toolkit.logging import configure_logger


httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.ERROR)

LOGGER = configure_logger(__name__)

system_config = get_configuration()
app = typer.Typer()
console = Console()

server = Server()
GAF_GUARD_AGENTS = {}
CLIENT_CONFIGS = {}


@app.callback()
def main() -> None:
    """
    GAF Guard Server
    """


@server.agent(
    name="orchestrator",
    description="Effectively detect and manage risks associated with LLMs for a given use-case",
    metadata=Metadata(
        license="Apache-2.0",
        programming_language="Python",
        natural_languages=["en"],
        framework="GAF-Guard",
        tags=["Governance Orchestrator", "AI Risks"],
        links=[
            Link(
                type=LinkType.SOURCE_CODE,
                url=AnyUrl(
                    "https://github.com/IBM/risk-atlas-nexus-demos/blob/main/gaf-guard/gaf_guard/agents/orchestrator.py"
                ),
            ),
            Link(
                type=LinkType.HOMEPAGE,
                url=AnyUrl(
                    "https://github.com/IBM/risk-atlas-nexus-demos/tree/main/gaf-guard"
                ),
            ),
        ],
        recommended_models=["granite3.2:8b"],
    ),
)
async def orchestrator(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:

    try:
        message = WorkflowStepMessage(
            **json.loads(str(reduce(lambda x, y: x + y, input)))
        )

        # Get run configs
        RUN_CONFIGS = message.run_configs or {}

        # Prepare config parameters
        config = CLIENT_CONFIGS.setdefault(
            context.session.id,
            {
                "trial_name": f"Trial_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}",
                "recursion_limit": 100,
                "run_id": str(uuid.uuid4()),
                "run_name": f"GAF-Guard-UserIntent",
                "configurable": {"thread_id": context.session.id} | RUN_CONFIGS,
            },
        )
        if message.step_type == MessageType.HITL_RESPONSE:
            state_dict = Command(resume=message.content)
        elif message.step_type == MessageType.WORKFLOW_INPUT:
            state_dict = message.content
        else:
            raise Exception(
                f"Invalid message type received: {message.step_type}. Valid types are: {MessageType._member_names_}"
            )

        for event in GAF_GUARD_AGENTS["OrchestratorAgent"].workflow.stream(
            input=state_dict,
            config=config,
            stream_mode="custom",
            subgraphs=True,
        ):
            for dest_type, message in event[1].items():
                if dest_type == "client":
                    yield Message(
                        role=Role.AGENT + "/orchestrator",
                        parts=[
                            MessagePart(
                                content=message.model_dump_json(),
                                content_type="text/plain",
                            )
                        ],
                    )
                elif dest_type == "logger":
                    await GAF_GUARD_AGENTS["TrialLoggerAgent"].workflow.ainvoke(
                        input=message.model_dump(),
                        config={
                            "configurable": {
                                "thread_id": 1,
                                "trial_name": config["trial_name"],
                            }
                            | RUN_CONFIGS
                        },
                    )

    except HumanInterruptionException as e:
        yield MessageAwaitRequest(
            message=Message(role="agent", parts=[MessagePart(content=str(e))])
        )
    except Exception as e:
        LOGGER.error("Internal Server Error: " + str(e))


@server.agent(name="benchmark")
async def run_benchmark(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    state_dict = json.loads(str(reduce(lambda x, y: x + y, input)))
    event = await GAF_GUARD_AGENTS["BenchmarkAgent"].workflow.ainvoke(
        input=state_dict, config={"configurable": {"thread_id": 1}}
    )
    yield Message(
        role=Role.AGENT + "/benchmark",
        parts=[
            MessagePart(
                content=event["metrics_results"],
                content_type="text/plain",
            )
        ],
    )


@app.command()
def serve(config_file):
    os.system("clear")
    console.rule(f"[bold blue]GAF Guard[/bold blue]")
    console.print(f"[bold yellow]:rocket: Starting AI Governance Orchestrator\n")

    server_configs = yaml.load(
        Path(config_file).read_text(),
        Loader=yaml.SafeLoader,
    )
    GAF_GUARD_AGENTS.update(AgentBuilder().build(server_configs["agents"]))

    rprint(
        f"\nMaster Agents: [italic bold yellow]{', '.join(list(server_configs['agents'].keys()))}[/italic bold yellow]\n"
        f"Task Agents found: [italic bold yellow]{', '.join(list(server_configs['agents']['OrchestratorAgent'].keys()))}[/italic bold yellow]\n"
        # f"LLM ✈️ [italic bold yellow]{inference_params['wml']['model_name_or_path']}[/italic bold yellow]\n"
        # f"Chain of Thought (CoT) data directory [italic bold yellow]{configs['data_dir']}[/italic bold yellow]\n"
    )

    LOGGER.info(f"ACP ver-{acp_sdk.__version__} initialized.")
    LOGGER.info(
        f"Agent trajectories will be stored in: {Path(server_configs['agents']['TrialLoggerAgent']['trial_dir']).absolute()}"
    )
    rprint(
        Panel(
            f"Please follow the GAF Guard Wiki at https://github.com/IBM/risk-atlas-nexus-demos/wiki/GAF-Guard to learn how to send and consume data to/from GAF Guard.",
            title="GAF-Guard Wiki",
            title_align="center",
        )
    )

    host = system_config.GAF_GUARD_HOST
    port = system_config.GAF_GUARD_PORT

    LOGGER.info(
        f"Server ver-{gaf_guard.__version__} initialized. Listening at {host}:{port}. To exit press CTRL+C"
    )
    server.run(
        host=host,
        port=port,
        configure_logger=False,
        configure_telemetry=False,
        log_level=logging.ERROR,
    )


# if __name__ == "__main__":
#     app()

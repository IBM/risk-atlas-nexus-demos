import logging
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import json
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Annotated

import acp_sdk
import typer
import yaml
from acp_sdk.models import (
    Artifact,
    Link,
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

from gaf_guard.core.agent_builder import AgentBuilder
from gaf_guard.toolkit.enums import MessageType
from gaf_guard.toolkit.exceptions import HumanInterruptionException
from gaf_guard.toolkit.file_utils import resolve_file_paths
from gaf_guard.toolkit.logging import configure_logger
from gaf_guard.toolkit.models import WorkflowMessage


httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.ERROR)

logger = configure_logger(__name__)

cli_app = typer.Typer()
console = Console()

server = Server()
SERVER_CONFIGS = None
TRIAL_DIR = None
ORCHESTRATOR_AGENT = None
BENCHMARK_AGENT = None
CLIENT_CONFIGS = {}

agent_builder = AgentBuilder()


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
                type="source-code",
                url="https://github.com/IBM/risk-atlas-nexus-demos/blob/main/gaf-guard/gaf_guard/agents/orchestrator.py",
            ),
            Link(
                type="homepage",
                url="https://github.com/IBM/risk-atlas-nexus-demos/tree/main/gaf-guard",
            ),
        ],
        recommended_models=["granite3.2:8b"],
    ),
)
async def orchestrator(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    message = WorkflowMessage(**json.loads(str(reduce(lambda x, y: x + y, input))))

    # Prepare config parameters
    config = CLIENT_CONFIGS.setdefault(
        message.client_id,
        {
            "trial_file": os.path.join(
                TRIAL_DIR,
                f"Trial_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json",
            ),
            "recursion_limit": 100,
            "run_name": f"GAF-Guard-{uuid.uuid4()}",
            "configurable": SERVER_CONFIGS["run_configs"]
            | {"thread_id": message.client_id},
        },
    )

    try:
        if message.type == MessageType.HITL_RESPONSE:
            for event in ORCHESTRATOR_AGENT.workflow.stream(
                input=Command(resume=message.content),
                config=config,
                stream_mode="custom",
                subgraphs=True,
            ):
                yield MessagePart(
                    content=event[1].model_dump_json(),
                    content_type="text/plain",
                )
        elif message.type == MessageType.WORKFLOW_INPUT:
            for event in ORCHESTRATOR_AGENT.workflow.stream(
                input=message.content,
                config=config,
                stream_mode="custom",
                subgraphs=True,
            ):
                yield MessagePart(
                    content=event[1].model_dump_json(),
                    content_type="text/plain",
                )
        else:
            raise Exception(f"Invalid message type received: {message.type}")

    except HumanInterruptionException as e:
        yield MessageAwaitRequest(message=Message(parts=[MessagePart(content=str(e))]))
    except Exception as e:
        print(str(e))


@server.agent(name="benchmark")
async def benchmark(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    state_dict = json.loads(str(reduce(lambda x, y: x + y, input)))
    event = await BENCHMARK_AGENT.workflow.ainvoke(
        input=state_dict, config={"configurable": {"thread_id": 1}}
    )
    yield MessagePart(
        content=event["metrics_results"],
        content_type="text/plain",
    )


@cli_app.command()
def main(
    config_file: Annotated[
        str,
        typer.Option(
            help="Please enter Server configuration path.",
            rich_help_panel="Server configuration path",
        ),
    ],
    host: Annotated[
        str,
        typer.Option(
            help="Please enter GAF Guard Host.",
            rich_help_panel="Hostname",
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            help="Please enter GAF Guard Port.",
            rich_help_panel="Port",
        ),
    ] = 8000,
    trial_dir: Annotated[
        str,
        typer.Option(
            help="Please enter trail results directory.",
            rich_help_panel="Trial Result Dir",
        ),
    ] = "trials",
):
    os.system("clear")
    Path(trial_dir).mkdir(parents=True, exist_ok=True)
    console.rule(f"[bold blue]GAF Guard[/bold blue]")
    console.print(f"[bold yellow]:rocket: Starting Governance Orchestrator")

    global SERVER_CONFIGS
    SERVER_CONFIGS = yaml.load(
        Path(config_file).read_text(),
        Loader=yaml.SafeLoader,
    )
    resolve_file_paths(SERVER_CONFIGS)

    rprint(
        f"\nMaster Agent [italic bold yellow]Governance Orchestrator[/italic bold yellow]\n"
        f"Agents Found: [italic bold yellow]{', '.join(list(SERVER_CONFIGS['agents']['OrchestratorAgent'].keys()))}[/italic bold yellow]\n"
        # f"LLM ✈️ [italic bold yellow]{inference_params['wml']['model_name_or_path']}[/italic bold yellow]\n"
        # f"Chain of Thought (CoT) data directory [italic bold yellow]{configs['data_dir']}[/italic bold yellow]\n"
    )

    global ORCHESTRATOR_AGENT
    global BENCHMARK_AGENT
    BENCHMARK_AGENT, ORCHESTRATOR_AGENT = agent_builder.compile(
        SERVER_CONFIGS["agents"]
    )

    global TRIAL_DIR
    TRIAL_DIR = trial_dir

    logger.info(f"ACP ver-{acp_sdk.__version__} initialized.")
    logger.info(f"Agent trajectories will be stored in: {TRIAL_DIR}")
    rprint(
        Panel(
            f"\nPlease follow the GAF Guard Wiki at https://github.com/IBM/risk-atlas-nexus-demos/wiki/GAF-Guard to learn how to send and consume data to/from GAF Guard.",
            title="GAF-Guard Wiki",
            title_align="left",
        )
    )
    logger.info(f"Server listening at {host}:{port}. To exit press CTRL+C")
    server.run(
        host=host,
        port=port,
        configure_logger=False,
        log_level=logging.ERROR,
    )


if __name__ == "__main__":
    cli_app()

import os
import uuid

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import asyncio
import functools
import json
import signal
import sys
from pathlib import Path
from typing import Annotated, Any, Dict
from langgraph.errors import GraphInterrupt
import typer
import yaml
from langchain_core.callbacks import AsyncCallbackHandler, AsyncCallbackManager
from rich import print as rprint
from rich.console import Console
from websockets.asyncio.server import ServerConnection, serve
from langchain_core.runnables.config import RunnableConfig
from agentic_governance.core.agent_builder import AgentBuilder, OrchestratorAgent
from agentic_governance.toolkit.conn_manager import conn_manager
from agentic_governance.toolkit.file_utils import resolve_file_paths
from agentic_governance.toolkit.logging import configure_logger


logger = configure_logger(__name__)

cli_app = typer.Typer()
console = Console()


def signal_handler(sig, frame):
    print("Exiting...")
    for task in asyncio.tasks.all_tasks():
        task.cancel()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


async def websocket_endpoint(
    orchestrator: OrchestratorAgent, run_configs: Dict, websocket: ServerConnection
):

    # Extract client_id during the handshake
    client_id = await conn_manager.accept(websocket)
    run_configs["client_id"] = client_id
    async for state_dict in conn_manager.receive(client_id):
        a = await orchestrator.ainvoke(
            {state_dict["message_type"]: state_dict[state_dict["message_type"]]},
            config={
                "run_id": uuid.uuid4(),
                "tags": [state_dict["message_type"]],
                "configurable": run_configs,
            },
        )
        await conn_manager.completed(client_id)


async def start_server(config_file: Dict, host: str = "localhost", port: int = 8000):
    server_configs = yaml.load(
        Path(config_file).read_text(),
        Loader=yaml.SafeLoader,
    )
    resolve_file_paths(server_configs)

    rprint(
        f"\nMaster Agent [italic bold yellow]Governance Orchestrator[/italic bold yellow]\n"
        f"Agents Found: [italic bold yellow]{', '.join(list(server_configs['agents'].keys()))}[/italic bold yellow]\n"
        # f"LLM ✈️ [italic bold yellow]{inference_params['wml']['model_name_or_path']}[/italic bold yellow]\n"
        # f"Chain of Thought (CoT) data directory [italic bold yellow]{configs['data_dir']}[/italic bold yellow]\n"
    )

    agent_builder = AgentBuilder()
    orchestrator = agent_builder.compile(server_configs["agents"])

    logger.info(f"Server listening at {host}:{port}. To exit press CTRL+C")
    async with serve(
        functools.partial(
            websocket_endpoint, orchestrator, server_configs["run_configs"]
        ),
        host,
        port,
        ping_interval=None,
        close_timeout=5,
    ) as server:
        await server.serve_forever()


@cli_app.command()
def main(
    config_file: Annotated[
        str,
        typer.Option(
            help="Please enter LLM configuration path.",
            rich_help_panel="LLM configuration path",
        ),
    ],
    host: Annotated[
        str,
        typer.Option(
            help="Please enter LLM configuration path.",
            rich_help_panel="LLM configuration path",
        ),
    ],
    port: Annotated[
        int,
        typer.Option(
            help="Please enter LLM configuration path.",
            rich_help_panel="LLM configuration path",
        ),
    ],
):
    console.rule(f"[bold blue]Agentic AI Governance[/bold blue]")
    console.print(f"[bold yellow]:rocket: Starting Governance Orchestrator")

    asyncio.run(start_server(config_file, host, port))


if __name__ == "__main__":
    cli_app()

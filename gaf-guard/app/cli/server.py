import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import asyncio
import functools
import signal
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict

import typer
import yaml
from langchain_core.callbacks import AsyncCallbackHandler, AsyncCallbackManager
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import Command
from rich import print as rprint
from rich.console import Console
from websockets.asyncio.server import ServerConnection, serve

from gaf_guard.core.agent_builder import AgentBuilder, OrchestratorAgent
from gaf_guard.toolkit.conn_manager import conn_manager
from gaf_guard.toolkit.enums import MessageType
from gaf_guard.toolkit.file_utils import resolve_file_paths
from gaf_guard.toolkit.logging import configure_logger


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
    orchestrator: OrchestratorAgent,
    run_configs: Dict,
    trial_dir: str,
    websocket: ServerConnection,
):
    # Extract client_id during the handshake
    client_id = await conn_manager.accept(websocket)

    # Prepare config parameters
    trial_name = f"Trial_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    config = {
        "client_id": client_id,
        "trial_name": trial_name,
        "recursion_limit": 100,
        "trial_file": os.path.join(trial_dir, trial_name + ".json"),
        "run_name": f"GAF-Guard-{uuid.uuid4()}",
        "configurable": run_configs,
    }
    async for state_dict in conn_manager.receive(client_id):
        if state_dict["message_type"] == MessageType.INTERRUPT_RESPONSE:
            response = await orchestrator.ainvoke(
                Command(resume={"response": state_dict["body"]}), config
            )
        else:
            response = await orchestrator.ainvoke(state_dict, config)

        if "__interrupt__" in response:
            response = {
                "body": response["__interrupt__"][0].value,
                "message_type": MessageType.INTERRUPT_QUERY,
            }

        await conn_manager.send(**response)


async def start_server(
    config_file: Dict,
    host: str = "localhost",
    port: int = 8000,
    trial_dir: str = "trials",
):
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
            websocket_endpoint, orchestrator, server_configs["run_configs"], trial_dir
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

    asyncio.run(start_server(config_file, host, port, trial_dir))


if __name__ == "__main__":
    cli_app()

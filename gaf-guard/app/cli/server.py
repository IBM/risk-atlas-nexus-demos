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

import typer
import yaml
from acp_sdk.models import Message, MessageAwaitRequest
from acp_sdk.models.models import MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from langgraph.types import Command
from rich import print as rprint
from rich.console import Console

from gaf_guard.core.agent_builder import AgentBuilder
from gaf_guard.toolkit.enums import MessageType
from gaf_guard.toolkit.exceptions import HumanInterruptionException
from gaf_guard.toolkit.file_utils import resolve_file_paths
from gaf_guard.toolkit.logging import configure_logger


httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.ERROR)

logger = configure_logger(__name__)

cli_app = typer.Typer()
console = Console()

server = Server()
SERVER_CONFIGS = None
TRIAL_DIR = None
ORCHESTRATOR_AGENT = None
CLIENT_CONFIGS = {}

agent_builder = AgentBuilder()


def process_event(event, config):
    for key, value in event[1].items():
        if "log" == key:
            return MessagePart(
                content=json.dumps(value),
                content_type="text/plain",
            )
        if "benchmark" == key:
            row = {
                "task": value["step"],
                "role": value["role"],
                "content": value["data"],
            }
            if Path(config["trial_file"]).exists():
                trial_data = json.loads(Path(config["trial_file"]).read_text())
                trial_data.append(row)
            else:
                trial_data = [row]

            json.dump(trial_data, open(config["trial_file"], "w"), indent=4)


@server.agent(name="orchestrator")
async def orchestrator(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    state_dict = json.loads(str(reduce(lambda x, y: x + y, input)))

    # Prepare config parameters
    trial_name = f"Trial_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    config = CLIENT_CONFIGS.setdefault(
        state_dict["client_id"],
        {
            "client_id": state_dict["client_id"],
            "trial_name": trial_name,
            "recursion_limit": 100,
            "trial_file": os.path.join(TRIAL_DIR, trial_name + ".json"),
            "run_name": f"GAF-Guard-{uuid.uuid4()}",
            "configurable": SERVER_CONFIGS["run_configs"],
        },
    )

    try:
        if state_dict["message_type"] == MessageType.INTERRUPT_RESPONSE:
            for event in ORCHESTRATOR_AGENT.workflow.stream(
                Command(resume={"response": state_dict["message"]}),
                config,
                stream_mode="custom",
                subgraphs=True,
            ):
                yield process_event(event, config)
        elif state_dict["message_type"] == MessageType.USER_INTENT:
            for event in ORCHESTRATOR_AGENT.workflow.stream(
                {"user_intent": state_dict["message"]},
                config=config,
                stream_mode="custom",
                subgraphs=True,
            ):
                yield process_event(event, config)
        else:
            raise Exception(
                f"Invalid message type received: {state_dict['message_type']}"
            )

    except HumanInterruptionException as e:
        yield MessageAwaitRequest(message=Message(parts=[MessagePart(content=str(e))]))
    except Exception as e:
        print(str(e))


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
        f"Agents Found: [italic bold yellow]{', '.join(list(SERVER_CONFIGS['agents'].keys()))}[/italic bold yellow]\n"
        # f"LLM ✈️ [italic bold yellow]{inference_params['wml']['model_name_or_path']}[/italic bold yellow]\n"
        # f"Chain of Thought (CoT) data directory [italic bold yellow]{configs['data_dir']}[/italic bold yellow]\n"
    )

    global ORCHESTRATOR_AGENT
    ORCHESTRATOR_AGENT = agent_builder.compile(SERVER_CONFIGS["agents"])

    global TRIAL_DIR
    TRIAL_DIR = trial_dir

    logger.info(f"Server listening at {host}:{port}. To exit press CTRL+C")
    server.run(
        host=host,
        port=port,
        configure_logger=False,
        log_level=logging.ERROR,
    )


if __name__ == "__main__":
    cli_app()

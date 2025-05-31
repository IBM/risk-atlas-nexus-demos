#!/usr/bin/env python
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import asyncio
import json
import os
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Annotated

import typer
import yaml
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.prompt import Prompt
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedOK

from agentic_governance.toolkit.enums import MessageType
from agentic_governance.toolkit.file_utils import resolve_file_paths
from contextlib import suppress


def signal_handler(sig, frame):
    print("Exiting...")
    for task in asyncio.tasks.all_tasks():
        task.cancel()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

app = typer.Typer()

console = Console(log_time=True)


def select_prompt_type():
    choice = Prompt.ask(
        prompt="[bold blue]Please choose one of the options for real-time Risk Assessment and Drift Monitoring[/bold blue]\n1. Enter prompt manually\n2. Start streaming prompts from a JSON file.\nYour Choice: ",
        console=console,
        choices=[
            "1",
            "2",
        ],
        show_choices=False,
    )
    if choice == "1":
        prompt = Prompt.ask(
            prompt="\n[bold blue]Enter your prompt[/bold blue]",
            console=console,
        )
        prompt_list = [prompt]
    elif choice == "2":
        prompt_file = Prompt.ask(
            prompt="\n[bold blue]Enter JSON file path[/bold blue]",
            console=Console(),
        )
        prompt_list = json.load(Path(prompt_file).open("r"))

    return ((index, prompt) for index, prompt in enumerate(prompt_list, start=1))


async def server_connect(host, port, trial_dir):
    status = console.status(
        f"[bold yellow] Trying to connect to [italic blue][Agentic AI Governance][/italic blue] using host: [bold white]{host}[/] and port: [bold white]{port}[/]. To abort press CTRL+C",
    )

    Path(trial_dir).mkdir(parents=True, exist_ok=True)
    trial_file = os.path.join(
        trial_dir,
        f"Trial_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".json",
    )
    trial_data = []
    with Live(Group(status), console=console, screen=True) as live:
        async for websocket in connect(
            f"ws://{host}:{port}",
            additional_headers={"client_id": str(uuid.uuid4())},
            ping_timeout=None,
        ):
            status.update(f"[bold yellow] :bell: Successfully connected.[/]")
            time.sleep(2)
            live.stop()
            console.print(
                Panel(
                    Group(
                        Align.center(
                            f"\nA real-time monitoring system for risk assessment and drift monitoring.\n",
                            vertical="middle",
                        ),
                    ),
                    subtitle=f"[[bold white]{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}[/]] [italic bold white] :rocket: Connected to Agentic AI Governance Server at[/italic bold white] [bold white]localhost:8000[/bold white]",
                    title="[bold green]Agentic AI Governance[/]\n",
                    border_style="blue",
                )
            )
            user_intent = Prompt.ask(
                prompt=f"\n[bold blue] Enter your intent[/bold blue]",
                console=console,
            )
            trial_data.append(
                {
                    "task": "User Intent",
                    "role": "user",
                    "content": user_intent,
                }
            )
            json.dump(trial_data, open(trial_file, "w"), indent=4)
            await websocket.send(
                json.dumps(
                    {
                        "message_type": "user_intent",
                        "user_intent": user_intent,
                    }
                )
            )

            try:
                prompt_file_gen = None
                while True:
                    with console.status(
                        "[italic bold yellow]Processing...[/]",
                        spinner_style="status.spinner",
                    ):
                        message = await websocket.recv()

                    message = json.loads(message)
                    body = message.pop("body")
                    msg_type = message.pop("message_type")
                    if body and msg_type == MessageType.PRINT:
                        workflow_step = message.pop("workflow_step", None)
                        if workflow_step and workflow_step not in [
                            "Drift Monitoring Setup",
                            "Persisting Results",
                            "Assess Risk",
                        ]:
                            trial_data.append(
                                {
                                    "task": workflow_step,
                                    "role": (
                                        "user"
                                        if workflow_step == "Input Prompt"
                                        else "assistant"
                                    ),
                                    "content": (
                                        body["data"] if isinstance(body, dict) else body
                                    ),
                                }
                            )
                            json.dump(trial_data, open(trial_file, "w"), indent=4)
                        if isinstance(body, dict):
                            if "message" in body:
                                console.print(
                                    body["message"].format(data=body["data"]), **message
                                )
                            elif "data" in body:
                                console.print(body["data"], **message)
                        else:
                            console.print(body, **message)

                    elif msg_type == MessageType.RULE:
                        spacing = message.pop("spacing", "None")
                        if spacing in ["before", "both"]:
                            print()
                        Console(width=message.pop("width", None)).rule(body, **message)
                        if spacing in ["after", "both"]:
                            print()
                    elif msg_type == MessageType.COMPLETED:
                        if not prompt_file_gen:
                            prompt_file_gen = select_prompt_type()

                        try:
                            prompt_index, prompt = next(prompt_file_gen)
                        except StopIteration:
                            prompt_file_gen = select_prompt_type()
                            prompt_index, prompt = next(prompt_file_gen)
                        finally:
                            console.print(
                                f"\n--------------[bold green]Prompt-{prompt_index}[/]--------------",
                                justify="center",
                            )
                            await websocket.send(
                                json.dumps(
                                    {
                                        "message_type": "prompt",
                                        "prompt": prompt["text"],
                                    }
                                )
                            )

            except ConnectionClosedOK:
                print("Request Completed.")
            except Exception as e:
                print(f"Client Error: {str(e)}")


@app.command()
def main(
    host: Annotated[
        str,
        typer.Option(
            help="Please enter Agentic Governance Host.",
            rich_help_panel="Hostname",
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            help="Please enter Agentic Governance Port.",
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
    asyncio.run(server_connect(host=host, port=port, trial_dir=trial_dir))


if __name__ == "__main__":
    app()

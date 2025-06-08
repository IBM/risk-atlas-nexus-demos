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
from contextlib import suppress
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
from rich.prompt import Prompt
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedOK

from gaf_guard.toolkit.enums import MessageType


def signal_handler(sig, frame):
    print("Exiting...")
    for task in asyncio.tasks.all_tasks():
        task.cancel()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

app = typer.Typer()

console = Console(log_time=True)


async def server_connect(host, port):
    status = console.status(
        f"[bold yellow] Trying to connect to [italic blue][GAF Guard][/italic blue] using host: [bold white]{host}[/] and port: [bold white]{port}[/]. To abort press CTRL+C",
    )
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
                    subtitle=f"[[bold white]{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}[/]] [italic bold white] :rocket: Connected to GAF Guard Server at[/italic bold white] [bold white]localhost:8000[/bold white]",
                    title="[bold green]GAF Guard[/]\n",
                    border_style="blue",
                )
            )
            user_intent = Prompt.ask(
                prompt=f"\n[bold blue] Enter your intent[/bold blue]",
                console=console,
            )
            await websocket.send(
                json.dumps(
                    {
                        "message_type": MessageType.USER_INTENT,
                        "user_intent": user_intent,
                    }
                )
            )

            try:
                while True:
                    with console.status(
                        "[italic bold yellow]Processing...[/]",
                        spinner_style="status.spinner",
                    ):
                        message = await websocket.recv()

                    message = json.loads(message)
                    body = message.pop("body")
                    msg_type = message.pop("message_type")
                    if not body:
                        continue
                    elif msg_type == MessageType.PRINT:
                        console.print(body, **message)
                    elif msg_type == MessageType.RULE:
                        spacing = message.pop("spacing", "None")
                        if spacing in ["before", "both"]:
                            print()
                        Console(width=message.pop("width", None)).rule(body, **message)
                        if spacing in ["after", "both"]:
                            print()
                    elif msg_type == MessageType.INTERRUPT_QUERY:
                        prompt_response = Prompt.ask(
                            prompt=body["message"],
                            console=console,
                            choices=body["choices"] if "choices" in body else None,
                            show_choices=False,
                        )
                        await websocket.send(
                            json.dumps(
                                {
                                    "body": prompt_response,
                                    "message_type": MessageType.INTERRUPT_RESPONSE,
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
):
    os.system("clear")
    asyncio.run(server_connect(host=host, port=port))


if __name__ == "__main__":
    app()

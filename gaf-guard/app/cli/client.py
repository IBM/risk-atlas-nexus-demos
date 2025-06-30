import asyncio

#!/usr/bin/env python
import os

from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart

from gaf_guard.toolkit.enums import MessageType


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import asyncio
import json
import os
import signal
import sys
import time
import uuid
from datetime import datetime
from typing import Annotated

import typer
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt

from gaf_guard.toolkit.enums import MessageType


def signal_handler(sig, frame):
    print("Exiting...")
    for task in asyncio.tasks.all_tasks():
        task.cancel()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

app = typer.Typer()

console = Console(log_time=True)

client_id = str(uuid.uuid4())


async def run_stream(host, port):
    status = console.status(
        f"[bold yellow] Trying to connect to [italic blue][GAF Guard][/italic blue] using host: [bold white]{host}[/] and port: [bold white]{port}[/]. To abort press CTRL+C",
    )
    processing = console.status(
        "[italic bold yellow]Processing...[/]",
        spinner_style="status.spinner",
    )
    with Live(Group(status), console=console, screen=True) as live:
        async with Client(base_url=f"http://{host}:{port}") as client:
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
            input_message_type = MessageType.USER_INTENT
            input_message = Prompt.ask(
                prompt=f"\n[bold blue] Enter your intent[/bold blue]",
                console=console,
            )

            COMPLETED = False
            while True:
                processing.start()
                async for event in client.run_stream(
                    agent="orchestrator",
                    input=[
                        Message(
                            parts=[
                                MessagePart(
                                    content=json.dumps(
                                        {
                                            "client_id": client_id,
                                            "message_type": input_message_type,
                                            "message": input_message,
                                        }
                                    ),
                                    content_type="text/plain",
                                )
                            ]
                        )
                    ],
                ):
                    processing.stop()
                    if event.type == "message.part":
                        body = json.loads(event.part.content)
                        body_message_type = body["msg_type"]
                        body_message = body["message"]
                        body_message_kwargs = body.get("message_kwargs", {}) or {}
                        if body_message:
                            if body_message_type == MessageType.RULE:
                                print()
                                Console(width=None).rule(
                                    body_message, **body_message_kwargs
                                )
                            elif body_message_type == MessageType.DATA:
                                console.print(body_message, **body_message_kwargs)
                    elif event.type == "run.awaiting":
                        if hasattr(event, "run"):
                            body = json.loads(
                                event.run.await_request.message.parts[0].content
                            )
                            body_message = body["message"]
                            choices = body.get("choices", None)

                            input_message = Prompt.ask(
                                prompt=body_message,
                                console=console,
                                choices=choices if choices else None,
                                show_choices=False,
                            )
                            input_message_type = MessageType.INTERRUPT_RESPONSE
                    elif event.type == "run.completed":
                        COMPLETED = True
                    processing.start()

                if COMPLETED:
                    break


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
    asyncio.run(run_stream(host=host, port=port))


if __name__ == "__main__":
    app()

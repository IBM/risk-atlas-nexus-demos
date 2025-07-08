import asyncio

#!/usr/bin/env python
import os

from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart

from gaf_guard.toolkit.enums import MessageType, Role


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime
from typing import Annotated

import typer
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt

from gaf_guard.toolkit.enums import MessageType
from gaf_guard.toolkit.models import WorkflowMessage, WorkflowStepMessage


def signal_handler(sig, frame):
    print("Exiting...")
    for task in asyncio.tasks.all_tasks():
        task.cancel()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

app = typer.Typer()

console = Console(log_time=True)


async def run_stream(host, port):
    status = console.status(
        f"[bold yellow] Trying to connect to [italic blue][GAF Guard][/italic blue] using host: [bold white]{host}[/] and port: [bold white]{port}[/]. To abort press CTRL+C",
    )
    processing = console.status(
        "[italic bold yellow]Processing...[/]",
        spinner_style="status.spinner",
    )
    with Live(Group(status), console=console, screen=True) as live:
        async with (
            Client(base_url=f"http://{host}:{port}") as client,
            client.session() as session,
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
            input_message_type = MessageType.WORKFLOW_INPUT
            input_message_content = {
                "user_intent": Prompt.ask(
                    prompt=f"\n[bold blue]Enter your intent[/bold blue]",
                    console=console,
                )
            }

            COMPLETED = False
            while True:
                processing.start()
                async for event in session.run_stream(
                    agent="orchestrator",
                    input=[
                        Message(
                            parts=[
                                MessagePart(
                                    content=WorkflowMessage(
                                        name="GAF Guard Client",
                                        type=input_message_type,
                                        role=Role.USER,
                                        content=input_message_content,
                                        client_id=str(session._session.id),
                                    ).model_dump_json(),
                                    content_type="text/plain",
                                )
                            ]
                        )
                    ],
                ):
                    processing.stop()
                    step = None
                    if event.type == "message.part":
                        message = WorkflowStepMessage(**json.loads(event.part.content))
                        if message.step_type == MessageType.WORKFLOW_STARTED:
                            print()
                            Console(width=None).rule(
                                f"Workflow: [bold blue]{message.step_name}[/]",
                                **message.step_kwargs,
                            )
                        elif message.step_type == MessageType.STEP_STARTED:
                            if message.step_name != "Risk Assessment":
                                console.print(
                                    f"\n[bold blue]Workflow Step: [bold white]{message.step_name}[/bold white]....Started",
                                    **message.step_kwargs,
                                )
                            if message.step_desc:
                                console.print(message.step_desc, **message.step_kwargs)
                        elif message.step_type == MessageType.STEP_COMPLETED:
                            if message.step_name != "Risk Assessment":
                                console.print(
                                    f"[bold blue]Workflow Step: [bold white]{message.step_name}[/bold white]....Completed",
                                    **message.step_kwargs,
                                )
                        elif message.step_type == MessageType.STEP_DATA:
                            if isinstance(message.content, dict):
                                for key, value in message.content.items():
                                    if key != "risk_report":
                                        console.print(
                                            f"[bold yellow]{key.replace('_', ' ').title()}[/bold yellow]: {value}",
                                            **message.step_kwargs,
                                        )
                            else:
                                console.print(message.content, **message.step_kwargs)
                    elif event.type == "run.awaiting":
                        if hasattr(event, "run"):
                            message = WorkflowStepMessage(
                                **json.loads(
                                    event.run.await_request.message.parts[0].content
                                )
                            )
                            input_message_content = {
                                "response": Prompt.ask(
                                    prompt=f"[bold blue]{message.content}[/bold blue]",
                                    console=console,
                                    choices=(
                                        message.step_kwargs["choices"]
                                        if "choices" in message.step_kwargs
                                        else None
                                    ),
                                    show_choices=False,
                                )
                            }
                            input_message_type = MessageType.HITL_RESPONSE
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

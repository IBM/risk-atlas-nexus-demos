import asyncio
import json
import os
from functools import reduce
from typing import Annotated

import typer
from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart
from rich.console import Console


console = Console(log_time=True)
app = typer.Typer()


async def run_benchmark(host, port, trial_dir) -> None:
    async with Client(base_url=f"http://{host}:{port}") as client:
        with console.status(
            "[italic bold yellow]Running GAF-Guard benchmark...[/]",
            spinner_style="status.spinner",
        ):
            run = await client.run_sync(
                agent="benchmark",
                input=[
                    Message(
                        parts=[
                            MessagePart(
                                content=json.dumps({"trial_dir": trial_dir}),
                                content_type="text/plain",
                            )
                        ]
                    )
                ],
            )

        console.print(
            f"[bold yellow]Benchmark Results:[/]\n{str(reduce(lambda x, y: x + y, run.output))}"
        )


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
    trial_dir: Annotated[
        str,
        typer.Option(
            help="Please enter trial directory containing JSON trajectories.",
            rich_help_panel="Trial Result Dir",
        ),
    ] = "trials",
):
    os.system("clear")
    asyncio.run(run_benchmark(host, port, trial_dir))


if __name__ == "__main__":
    app()

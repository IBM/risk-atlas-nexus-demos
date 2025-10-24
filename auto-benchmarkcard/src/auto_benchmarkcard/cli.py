#!/usr/bin/env python3
"""
BenchmarkCard CLI - Comprehensive benchmark metadata extraction and validation.

This CLI tool orchestrates a complete pipeline for processing AI benchmarks,
including metadata extraction, LLM-powered card generation, AI risk assessment,
evidence retrieval, and factual accuracy validation.

Usage:
    benchmarkcard process <benchmark_name>
    benchmarkcard list --recent 10
"""

import json
import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, Optional, Union

import typer
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.status import Status
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Suppress noisy warnings from external libraries at startup
warnings.filterwarnings("ignore", message=".*Triton.*")
warnings.filterwarnings("ignore", message=".*not installed.*")
warnings.filterwarnings("ignore", message=".*dummy decorators.*")
warnings.filterwarnings("ignore", message=".*Failed to load GPU.*")
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*LangChain.*deprecated.*")
warnings.filterwarnings("ignore", message=".*manual persistence.*")

logging.getLogger("faiss.loader").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("vllm.config").setLevel(logging.WARNING)
logging.getLogger("vllm.utils.import_utils").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

original_level = logging.root.level
logging.root.setLevel(logging.WARNING)
try:
    from auto_benchmarkcard.cli_logger import WorkflowCLILogger
    from auto_benchmarkcard.config import Config
    from auto_benchmarkcard.workflow import main as run_workflow
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all dependencies are installed and the project is properly set up.")
    sys.exit(1)

logging.root.setLevel(original_level)

console = Console(log_time=False, log_time_format="[%X]")
error_console = Console(stderr=True, style="bold red")
app = typer.Typer(
    name="benchmarkcard",
    help="Benchmark Metadata Extraction & Validation",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    add_completion=False,
)


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging with Rich integration.

    Args:
        verbose: Enable verbose/debug logging output.
        log_file: Optional path to save logs to file.

    Returns:
        Configured logger instance for benchmarkcard.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create logger
    logger = logging.getLogger("benchmarkcard")
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Rich console handler for terminal output
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=verbose,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=verbose,
    )
    console_handler.setLevel(log_level)

    # Custom formatter
    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def enable_debug_logging() -> None:
    """Enable debug-level logging for all tools and external libraries.

    This function resets the logging levels for all suppressed external
    library loggers to DEBUG level and removes warning filters to show
    all diagnostic information during debugging.
    """
    # External library loggers to enable for debugging
    debug_loggers = [
        "faiss.loader",
        "faiss",
        "vllm",
        "vllm.config",
        "vllm.utils.import_utils",
        "transformers",
        "httpx",
        "httpcore",
        "litellm",
        "LiteLLM",
        "litellm.llms",
        "litellm.utils",
        "litellm.cost_calculator",
        "openai",
        "urllib3",
        "huggingface_hub",
        "docling",
        "unitxt",
        "chromadb",
        "sentence_transformers",
        "fact_reasoner",
        "FactReasoner",
        "risk_atlas_nexus",
    ]

    for logger_name in debug_loggers:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)

    # Enable all warnings that were filtered out
    warnings.resetwarnings()


# UI COMPONENTS


def display_banner() -> None:
    """Display application banner with title and subtitle."""
    title = Text("Auto-BenchmarkCard", style="bold cyan")
    subtitle = Text("Benchmark Metadata Extraction & Validation", style="dim italic")

    # Build banner content
    banner_content = Align.center(
        Columns(
            [
                Panel(
                    Align.center(f"{title}\n{subtitle}"),
                    border_style="cyan",
                    padding=(1, 2),
                    title="[bold white]Welcome[/bold white]",
                )
            ],
            expand=True,
        )
    )

    console.print(banner_content)
    console.print()


def display_workflow_summary(
    benchmark: str, execution_time: float, step_results: dict, output_manager=None
):
    """Display comprehensive workflow execution summary.

    Args:
        benchmark: Name of the benchmark being processed.
        execution_time: Total execution time in seconds.
        step_results: Dictionary mapping step names to result dictionaries.
        output_manager: Optional output manager for session information.
    """
    console.print("\n" + "=" * 60)
    console.print(f"[bold cyan]Workflow Summary: {benchmark}[/bold cyan]")
    console.print("=" * 60)

    # Create results table
    results_table = Table(border_style="green", title="Processing Results")
    results_table.add_column("Step", style="cyan", width=25)
    results_table.add_column("Status", style="green", width=10)
    results_table.add_column("Details", style="white")

    for step_name, result in step_results.items():
        status_icon = "✅" if result.get("success", False) else "❌"
        details = result.get("details", "No details available")
        results_table.add_row(step_name, status_icon, details)

    console.print(results_table)

    # Summary stats
    successful_steps = sum(1 for r in step_results.values() if r.get("success", False))
    total_steps = len(step_results)

    console.print(f"\n[bold]Execution Summary:[/bold]")
    console.print(f"• Total execution time: [cyan]{format_duration(execution_time)}[/cyan]")
    console.print(f"• Steps completed: [green]{successful_steps}/{total_steps}[/green]")

    # Add output directory and timestamp if output_manager is provided
    if output_manager:
        summary = output_manager.get_summary()
        # Ensure absolute path
        output_dir = os.path.abspath(summary["session_directory"])
        console.print(f"• Output directory: [cyan]{output_dir}[/cyan]")
        console.print(f"• Generation timestamp: [cyan]{summary['timestamp']}[/cyan]")

    console.print("=" * 60)


def create_progress_display() -> Progress:
    """Create progress display with spinner, bar, and time tracking.

    Returns:
        Configured Progress instance for workflow tracking.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


@contextmanager
def workflow_step(step_name: str, step_number: int = None, total_steps: int = None):
    """Context manager for workflow step tracking with timing.

    Args:
        step_name: Name of the workflow step to display.
        step_number: Current step number (optional).
        total_steps: Total number of steps (optional).

    Yields:
        Status object for updating step progress.
    """
    # Format step indicator
    if step_number and total_steps:
        step_indicator = f"[dim]Step {step_number}/{total_steps}[/dim] "
    else:
        step_indicator = ""

    # Display step start
    console.print(f"\n{step_indicator}🔄 [bold blue]{step_name}[/bold blue]")

    with Status(
        f"[blue]{step_name}...[/blue]",
        console=console,
        spinner="dots12",
    ) as status:
        start_time = time.time()
        try:
            yield status
            # Success case
            elapsed = time.time() - start_time
            console.print(f"✅ [green]{step_name} completed[/green] [dim]({elapsed:.1f}s)[/dim]")
        except Exception:
            # Error case
            elapsed = time.time() - start_time
            console.print(f"❌ [red]{step_name} failed[/red] [dim]({elapsed:.1f}s)[/dim]")
            raise


@contextmanager
def workflow_substep(substep_name: str, show_completion: bool = True):
    """Context manager for sub-steps within a workflow step.

    Args:
        substep_name: Name of the sub-step to display.
        show_completion: Whether to display completion message.

    Yields:
        Status object for updating substep progress.
    """
    with Status(
        f"[dim]{substep_name}...[/dim]",
        console=console,
        spinner="dots",
    ) as status:
        start_time = time.time()
        try:
            yield status
            if show_completion:
                elapsed = time.time() - start_time
                console.print(
                    f"  [green]• {substep_name} completed[/green] [dim]({elapsed:.1f}s)[/dim]"
                )
        except Exception as e:
            elapsed = time.time() - start_time
            console.print(
                f"  [red]• {substep_name} failed: {str(e)}[/red] [dim]({elapsed:.1f}s)[/dim]"
            )
            raise


def display_error(message: str, details: Optional[str] = None) -> None:
    """Display error message in a styled panel.

    Args:
        message: Main error message to display.
        details: Optional additional details about the error.
    """
    error_panel = Panel(
        f"[bold red]❌ Error[/bold red]\n\n{message}"
        + (f"\n\n[dim]{details}[/dim]" if details else ""),
        border_style="red",
        title="[bold red]Execution Failed[/bold red]",
    )
    error_console.print(error_panel)


def display_success(message: str, details: Optional[str] = None) -> None:
    """Display success message in a styled panel.

    Args:
        message: Main success message to display.
        details: Optional additional details about the success.
    """
    success_panel = Panel(
        f"[bold green]✅ Success[/bold green]\n\n{message}"
        + (f"\n\n[dim]{details}[/dim]" if details else ""),
        border_style="green",
        title="[bold green]Execution Completed[/bold green]",
    )
    console.print(success_panel)


# VALIDATION & UTILITIES


def validate_benchmark_name(benchmark: str) -> str:
    """Validate and sanitize benchmark name with comprehensive checks.

    Args:
        benchmark: Raw benchmark name from user input.

    Returns:
        Sanitized benchmark name safe for filesystem use.

    Raises:
        typer.BadParameter: If benchmark name is invalid.
    """
    if not benchmark or not benchmark.strip():
        raise typer.BadParameter(
            "[red]Benchmark name cannot be empty[/red]\n"
            "[dim]Example: 'glue', 'safety.truthful_qa', 'ethos_binary'[/dim]"
        )

    sanitized = benchmark.strip()

    # Length validation
    if len(sanitized) > 100:
        raise typer.BadParameter(
            f"[red]Benchmark name too long ({len(sanitized)} chars, max 100)[/red]"
        )

    # Character validation
    invalid_chars = set(sanitized) - set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
    )
    if invalid_chars:
        raise typer.BadParameter(
            f"[red]Invalid characters in benchmark name: {''.join(invalid_chars)}[/red]\n"
            "[dim]Only letters, numbers, dots, hyphens, and underscores are allowed[/dim]"
        )

    return sanitized


def validate_path(path: str, must_exist: bool = False) -> Path:
    """Validate file or directory path.

    Args:
        path: Path string to validate.
        must_exist: Whether the path must already exist.

    Returns:
        Resolved Path object.

    Raises:
        typer.BadParameter: If must_exist is True and path doesn't exist.
    """
    path_obj = Path(path).resolve()

    if must_exist and not path_obj.exists():
        raise typer.BadParameter(f"[red]Path does not exist: {path_obj}[/red]")

    return path_obj


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable duration string (e.g., "1m 30s", "2h 15m").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def execute_workflow_with_cli_integration(
    benchmark: str,
    catalog: Optional[str] = None,
    output_dir: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Execute the main workflow with CLI integration.

    This function handles the core workflow execution with proper
    CLI formatting and logging integration.

    Args:
        benchmark: Benchmark name to process.
        catalog: Optional custom catalog path.
        output_dir: Optional custom output directory.
        debug: Whether to enable debug mode.
    """
    # Prepare arguments for the main workflow
    original_argv = sys.argv.copy()
    sys.argv = ["benchmarkcard", benchmark]

    if catalog:
        sys.argv.extend(["--cataloge", str(catalog)])
    if output_dir:
        sys.argv.extend(["--output", str(output_dir)])
    if debug:
        sys.argv.append("--debug")

    console.print(f"\n[dim]Starting workflow execution...[/dim]")

    with Status(
        "[blue]Initializing workflow...[/blue]",
        console=console,
        spinner="dots12",
    ) as status:
        # Import the agents workflow execution directly
        import auto_benchmarkcard.workflow as agents

        # Temporarily replace the agents logger to use CLI's console
        original_agents_logger = agents.logger

        # Replace agents logger temporarily with our custom CLI logger
        agents.logger = WorkflowCLILogger(status, console)

        try:
            # Execute the main workflow
            run_workflow()
        finally:
            # Restore original logger and argv
            agents.logger = original_agents_logger
            sys.argv = original_argv


def get_session_info(session_dir: Path) -> Dict[str, Union[str, int, bool]]:
    """Extract comprehensive session information from a directory.

    Args:
        session_dir: Path to the session directory to analyze.

    Returns:
        Dictionary containing session metadata including benchmark name, timestamp,
        completion status, file counts, and sizes.
    """
    try:
        # Parse directory name
        parts = session_dir.name.rsplit("_", 2)
        benchmark = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
        timestamp = "_".join(parts[-2:]) if len(parts) >= 2 else "unknown"

        # Check completion status
        benchmark_card_dir = session_dir / "benchmarkcard"
        tool_output_dir = session_dir / "tool_output"

        completed = benchmark_card_dir.exists() and any(benchmark_card_dir.glob("*.json"))
        tool_count = len(list(tool_output_dir.iterdir())) if tool_output_dir.exists() else 0

        # Get file statistics
        total_size = sum(f.stat().st_size for f in session_dir.rglob("*") if f.is_file())
        file_count = len(list(session_dir.rglob("*")))

        return {
            "benchmark": benchmark,
            "timestamp": timestamp,
            "completed": completed,
            "tool_count": tool_count,
            "total_size_mb": total_size / (1024 * 1024),
            "file_count": file_count,
            "modified_time": session_dir.stat().st_mtime,
        }
    except Exception:
        return {
            "benchmark": "unknown",
            "timestamp": "unknown",
            "completed": False,
            "tool_count": 0,
            "total_size_mb": 0.0,
            "file_count": 0,
            "modified_time": 0,
        }


# MAIN COMMANDS


@app.command("process")
def process_benchmark(
    benchmark: Annotated[
        str,
        typer.Argument(
            help="Benchmark name to process (e.g., 'glue', 'ethos_binary', 'safety.truthful_qa')",
            callback=validate_benchmark_name,
        ),
    ],
    catalog: Annotated[
        Optional[str],
        typer.Option(
            "--catalog",
            "-c",
            help="Path to custom UnitXT catalog directory",
            rich_help_panel="📁 Data Sources",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[str],
        typer.Option(
            "--output",
            "-o",
            help="Custom output directory path for results",
            rich_help_panel="📁 Output Configuration",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging with detailed output",
            rich_help_panel="🔧 Logging Options",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug mode with full tool logging output",
            rich_help_panel="🔧 Logging Options",
        ),
    ] = False,
    log_file: Annotated[
        Optional[str],
        typer.Option(
            "--log-file",
            help="Save logs to specified file",
            rich_help_panel="🔧 Logging Options",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate inputs and show execution plan without running",
            rich_help_panel="⚙️ Processing Options",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing output directory if it exists",
            rich_help_panel="⚙️ Processing Options",
        ),
    ] = False,
) -> None:
    """
    🚀 Process a benchmark through the complete metadata extraction and validation pipeline.

    This command orchestrates the full workflow including:

    • [bold cyan]UnitXT Metadata Lookup[/bold cyan] - Retrieve benchmark definitions
    • [bold cyan]HuggingFace Extraction[/bold cyan] - Extract dataset information
    • [bold cyan]Academic Paper Processing[/bold cyan] - Download and analyze papers
    • [bold cyan]BenchmarkCard Composition with LLM[/bold cyan] - Generate structured benchmark cards
    • [bold cyan]AI Risk Assessment[/bold cyan] - Identify risks via Risk Atlas Nexus
    • [bold cyan]RAG Evidence Retrieval[/bold cyan] - Gather supporting evidence
    • [bold cyan]Factual Accuracy Validation[/bold cyan] - Verify claims with FactReasoner

    [bold green]Examples:[/bold green]

        [dim]# Basic processing[/dim]
        benchmarkcard process glue

        [dim]# With custom output and verbose logging[/dim]
        benchmarkcard process safety.truthful_qa --output ./results --verbose

        [dim]# Using custom catalog with log file[/dim]
        benchmarkcard process ethos_binary --catalog ./custom --log-file process.log
    """
    # Setup logging
    logger = setup_logging(verbose=verbose, log_file=log_file)

    # Enable full tool logging in debug mode
    if debug:
        enable_debug_logging()
        console.print("[dim]Debug mode enabled - showing full tool logs[/dim]")

    # Display banner
    display_banner()

    # Validate inputs
    if catalog:
        catalog_path = validate_path(catalog, must_exist=True)
        logger.info(f"Using custom catalog: [cyan]{catalog_path}[/cyan]")

    if output_dir:
        output_path = validate_path(output_dir)
        if output_path.exists() and not force:
            error_console.print(
                f"[red]Output directory already exists: {output_path}[/red]\n"
                "[dim]Use --force to overwrite or choose a different path[/dim]"
            )
            raise typer.Exit(1)
        logger.info(f"Output directory: [cyan]{output_path}[/cyan]")

    # Show execution plan for dry run
    if dry_run:
        console.print("\n[bold yellow]🔍 DRY RUN - Execution Plan:[/bold yellow]\n")

        plan_table = Table(title="Execution Plan", border_style="yellow")
        plan_table.add_column("Step", style="cyan", no_wrap=True)
        plan_table.add_column("Description", style="white")
        plan_table.add_column("Status", style="green")

        steps = [
            (
                "1. UnitXT Metadata Extraction",
                f"Fetch benchmark definitions for '{benchmark}'",
                "Ready",
            ),
            (
                "2. ID and URL Extraction",
                "Extract HuggingFace repo and paper URLs",
                "Ready",
            ),
            (
                "3. HuggingFace Extraction",
                "Retrieve dataset metadata and information",
                "Conditional",
            ),
            (
                "4. Academic Paper Processing",
                "Download and analyze research papers",
                "Conditional",
            ),
            (
                "5. BenchmarkCard Composition with LLM",
                "Generate structured benchmark card",
                "Ready",
            ),
            ("6. Risk Identification", "Identify risks via Risk Atlas Nexus", "Ready"),
            (
                "7. RAG Evidence Retrieval",
                "Gather supporting evidence for validation",
                "Ready",
            ),
            (
                "8. Factual Accuracy Validation",
                "Verify claims with FactReasoner",
                "Ready",
            ),
        ]

        for step, desc, status in steps:
            plan_table.add_row(step, desc, status)

        console.print(plan_table)
        console.print("\n[dim]Run without --dry-run to execute the pipeline[/dim]")
        return

    # Start processing with enhanced workflow tracking
    start_time = time.time()
    step_results = {}

    try:
        console.print(f"\n[bold cyan]Starting BenchmarkCard Workflow[/bold cyan]")
        console.print(f"[dim]Target benchmark: {benchmark}[/dim]")

        if catalog:
            console.print(f"[dim]Custom catalog: {catalog_path}[/dim]")
        if output_dir:
            console.print(f"[dim]Output directory: {output_path}[/dim]")

        console.print("\n[bold]Pipeline Steps:[/bold]")
        console.print("1. UnitXT Metadata Extraction")
        console.print("2. ID and URL Extraction")
        console.print("3. HuggingFace Extraction (if applicable)")
        console.print("4. Academic Paper Processing (if available)")
        console.print("5. BenchmarkCard Composition with LLM")
        console.print("6. Risk Identification")
        console.print("7. RAG Evidence Retrieval")
        console.print("8. Factual Accuracy Validation")

        logger.debug(f"Executing workflow with args: {benchmark}")

        # Execute the main workflow with enhanced monitoring
        execute_workflow_with_cli_integration(
            benchmark=benchmark,
            catalog=str(catalog_path) if catalog else None,
            output_dir=str(output_path) if output_dir else None,
            debug=debug,
        )

        # Mark overall success
        step_results["Overall Processing"] = {
            "success": True,
            "details": "All steps completed successfully",
        }

        # Calculate execution time
        execution_time = time.time() - start_time

        # Display comprehensive success summary
        display_workflow_summary(benchmark, execution_time, step_results, None)

        display_success(
            f"Benchmark '{benchmark}' processed successfully",
            f"Total execution time: {format_duration(execution_time)}\nAll workflow steps completed successfully",
        )

        # Completion message already shown in success panel above

    except KeyboardInterrupt:
        execution_time = time.time() - start_time
        console.print(
            f"\n⚠️ [yellow]Workflow interrupted by user[/yellow] [dim]({format_duration(execution_time)} elapsed)[/dim]"
        )
        logger.warning(f"Workflow interrupted after {format_duration(execution_time)}")
        raise typer.Exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        execution_time = time.time() - start_time
        step_results["Error Recovery"] = {"success": False, "details": str(e)}

        display_error(
            f"Workflow failed for benchmark '{benchmark}'",
            f"Error: {str(e)}\nExecution time: {format_duration(execution_time)}\nCheck logs for detailed information",
        )

        logger.error(
            f"❌ Workflow failed after {format_duration(execution_time)}: {e}",
            exc_info=verbose,
        )
        raise typer.Exit(1)


@app.command("list")
def list_outputs(
    output_dir: Annotated[
        Optional[str],
        typer.Option(
            "--output",
            "-o",
            help="Output directory to scan (default: ./output)",
            rich_help_panel="📁 Directory Options",
        ),
    ] = None,
    recent: Annotated[
        int,
        typer.Option(
            "--recent",
            "-n",
            help="Show only the N most recent sessions",
            rich_help_panel="📊 Display Options",
            min=1,
            max=100,
        ),
    ] = 10,
    format_type: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: table, json, or tree",
            rich_help_panel="📊 Display Options",
        ),
    ] = "table",
    filter_completed: Annotated[
        bool,
        typer.Option(
            "--completed-only",
            help="Show only completed sessions",
            rich_help_panel="🔍 Filter Options",
        ),
    ] = False,
) -> None:
    """
    📋 List recent benchmark processing sessions and their outputs.

    Displays comprehensive information about processing sessions including:
    • Session status and completion
    • File counts and sizes
    • Processing timestamps
    • Output organization

    [bold green]Examples:[/bold green]

        [dim]# Show recent sessions[/dim]
        benchmarkcard list

        [dim]# Show only completed sessions in JSON format[/dim]
        benchmarkcard list --completed-only --format json

        [dim]# Show last 20 sessions as tree structure[/dim]
        benchmarkcard list --recent 20 --format tree
    """
    setup_logging()

    # Validate format type
    valid_formats = {"table", "json", "tree"}
    if format_type not in valid_formats:
        display_error(
            f"Invalid format type: {format_type}",
            f"Valid options: {', '.join(valid_formats)}",
        )
        raise typer.Exit(1)

    # Validate output directory
    output_path = validate_path(output_dir or "output")

    if not output_path.exists():
        display_error(
            f"Output directory not found: {output_path}",
            "Run 'benchmarkcard process <benchmark>' to create output sessions",
        )
        return

    # Collect session information
    with workflow_substep("Scanning processing sessions", show_completion=False):
        sessions = []
        session_count = 0
        for item in output_path.iterdir():
            if item.is_dir() and "_" in item.name:
                session_info = get_session_info(item)
                if session_info["benchmark"] != "unknown":
                    session_info["path"] = item
                    sessions.append(session_info)
                    session_count += 1

        console.print(f"  [green]• Found {session_count} processing sessions[/green]")

    if not sessions:
        console.print(
            Panel(
                "[yellow]No benchmark processing sessions found[/yellow]\n\n"
                "[dim]Create sessions by running:[/dim]\n"
                "[cyan]benchmarkcard process <benchmark_name>[/cyan]",
                title="[bold]No Sessions Found[/bold]",
                border_style="yellow",
            )
        )
        return

    # Filter completed sessions if requested
    if filter_completed:
        sessions = [s for s in sessions if s["completed"]]
        if not sessions:
            console.print("[yellow]No completed sessions found[/yellow]")
            return

    # Sort by modification time (newest first) and limit
    sessions.sort(key=lambda x: x["modified_time"], reverse=True)
    sessions = sessions[:recent]

    # Display based on format
    if format_type == "json":
        # JSON output
        json_data = [
            {
                "benchmark": s["benchmark"],
                "timestamp": s["timestamp"],
                "completed": s["completed"],
                "tool_count": s["tool_count"],
                "file_count": s["file_count"],
                "size_mb": round(s["total_size_mb"], 2),
                "path": str(s["path"]),
            }
            for s in sessions
        ]
        console.print_json(data=json_data)

    elif format_type == "tree":
        # Tree output
        tree = Tree(
            f"[bold cyan]Processing Sessions[/bold cyan] ([dim]{len(sessions)} sessions[/dim])",
            guide_style="dim",
        )

        for session in sessions:
            status_icon = "✅" if session["completed"] else "⚠️"
            node = tree.add(
                f"{status_icon} [cyan]{session['benchmark']}[/cyan] "
                f"[dim]({session['timestamp'].replace('_', ' ').replace('-', ':')}) "
                f"• {session['file_count']} files • {session['total_size_mb']:.1f}MB[/dim]"
            )

            # Add details
            node.add(f"📁 Path: [blue]{session['path']}[/blue]")
            node.add(f"🔧 Tools: [green]{session['tool_count']}[/green]")
            node.add(
                f"📊 Status: {'[green]Complete[/green]' if session['completed'] else '[yellow]Incomplete[/yellow]'}"
            )

        console.print(tree)

    else:
        # Table output (default)
        table = Table(
            title=f"Recent Benchmark Processing Sessions ({len(sessions)} sessions)",
            border_style="cyan",
            title_style="bold cyan",
        )
        table.add_column("Status", style="green", width=8, justify="center")
        table.add_column("Benchmark", style="cyan", no_wrap=True)
        table.add_column("Timestamp", style="blue", width=16)
        table.add_column("Tools", style="yellow", width=6, justify="right")
        table.add_column("Files", style="magenta", width=6, justify="right")
        table.add_column("Size", style="green", width=8, justify="right")
        table.add_column("Path", style="dim", overflow="ellipsis")

        for session in sessions:
            status = "✅ Done" if session["completed"] else "⚠️ Partial"
            timestamp_fmt = session["timestamp"].replace("_", " ").replace("-", ":")
            size_fmt = (
                f"{session['total_size_mb']:.1f}MB" if session["total_size_mb"] > 0 else "0MB"
            )

            table.add_row(
                status,
                session["benchmark"],
                timestamp_fmt,
                str(session["tool_count"]),
                str(session["file_count"]),
                size_fmt,
                str(session["path"].relative_to(Path.cwd())),
            )

        console.print(table)

    # Summary statistics
    completed_count = sum(1 for s in sessions if s["completed"])
    total_size = sum(s["total_size_mb"] for s in sessions)

    console.print(
        f"\n[dim]Summary: {completed_count}/{len(sessions)} completed • "
        f"Total size: {total_size:.1f}MB[/dim]"
    )


@app.command("show")
def show_session(
    session_path: Annotated[
        str,
        typer.Argument(help="Path to the benchmark session directory", metavar="SESSION_PATH"),
    ],
    detailed: Annotated[
        bool,
        typer.Option(
            "--detailed",
            "-d",
            help="Show detailed file information and content previews",
            rich_help_panel="📊 Display Options",
        ),
    ] = False,
) -> None:
    """
    🔍 Show comprehensive details about a benchmark processing session.

    Displays detailed information about the outputs, tools used, files generated,
    and processing results from a specific benchmark session.

    [bold green]Examples:[/bold green]

        [dim]# Show basic session info[/dim]
        benchmarkcard show output/glue_2025-01-08_14-30

        [dim]# Show detailed file information[/dim]
        benchmarkcard show output/glue_2025-01-08_14-30 --detailed
    """
    setup_logging()

    # Validate session directory
    session_dir = validate_path(session_path, must_exist=True)

    if not session_dir.is_dir():
        display_error(
            f"Path is not a directory: {session_dir}",
            "Please provide a valid session directory path",
        )
        raise typer.Exit(1)

    # Get comprehensive session info
    session_info = get_session_info(session_dir)

    # Header
    console.print(Rule(f"[bold cyan]Session Details: {session_dir.name}[/bold cyan]", style="cyan"))
    console.print()

    # Session overview
    overview_table = Table(border_style="blue", title="Session Overview")
    overview_table.add_column("Property", style="cyan", width=20)
    overview_table.add_column("Value", style="green")

    status_text = (
        "[bold green]✅ Completed[/bold green]"
        if session_info["completed"]
        else "[bold yellow]⚠️ Incomplete[/bold yellow]"
    )
    timestamp_fmt = session_info["timestamp"].replace("_", " ").replace("-", ":")

    overview_table.add_row("Benchmark", f"[bold]{session_info['benchmark']}[/bold]")
    overview_table.add_row("Status", status_text)
    overview_table.add_row("Timestamp", timestamp_fmt)
    overview_table.add_row("Tools Used", str(session_info["tool_count"]))
    overview_table.add_row("Total Files", str(session_info["file_count"]))
    overview_table.add_row("Total Size", f"{session_info['total_size_mb']:.2f} MB")
    overview_table.add_row("Full Path", str(session_dir.absolute()))

    console.print(overview_table)
    console.print()

    # Directory structure analysis
    tool_output_dir = session_dir / "tool_output"
    benchmark_card_dir = session_dir / "benchmarkcard"

    if tool_output_dir.exists():
        # Tool outputs section
        console.print("[bold cyan]🔧 Tool Outputs[/bold cyan]")

        tools_table = Table(border_style="cyan")
        tools_table.add_column("Tool", style="cyan", width=20)
        tools_table.add_column("Files", style="yellow", width=8, justify="right")
        tools_table.add_column("Size", style="green", width=10, justify="right")
        tools_table.add_column("Description", style="dim")

        tool_descriptions = {
            "unitxt": "UnitXT benchmark metadata",
            "extractor": "Extracted IDs and URLs",
            "hf": "HuggingFace dataset info",
            "docling": "Processed academic papers",
            "risk_enhanced": "Risk-enhanced benchmark cards",
            "risk_atlas_nexus": "AI risk assessment results",
            "rag": "Evidence retrieval results",
            "factreasoner": "Factuality verification scores",
        }

        for tool_dir in sorted(tool_output_dir.iterdir()):
            if tool_dir.is_dir():
                files = list(tool_dir.glob("*"))
                file_count = len(files)
                total_size = sum(f.stat().st_size for f in files if f.is_file()) / 1024  # KB
                description = tool_descriptions.get(tool_dir.name, "Tool output")

                size_str = f"{total_size:.1f} KB" if total_size > 0 else "0 KB"
                tools_table.add_row(tool_dir.name, str(file_count), size_str, description)

                # Show detailed file info if requested
                if detailed and files:
                    console.print(f"\n[dim]Files in {tool_dir.name}:[/dim]")
                    for file in sorted(files):
                        if file.is_file():
                            size_kb = file.stat().st_size / 1024
                            mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime(
                                "%Y-%m-%d %H:%M"
                            )
                            console.print(
                                f"  📄 [blue]{file.name}[/blue] ({size_kb:.1f} KB, {mtime})"
                            )

        console.print(tools_table)
        console.print()

    if benchmark_card_dir.exists():
        # Benchmark cards section
        console.print("[bold green]📋 Benchmark Cards[/bold green]")

        cards_table = Table(border_style="green")
        cards_table.add_column("File", style="green", width=40)
        cards_table.add_column("Size", style="yellow", width=10, justify="right")
        cards_table.add_column("Modified", style="blue", width=16)

        for card_file in sorted(benchmark_card_dir.glob("*.json")):
            size_kb = card_file.stat().st_size / 1024
            mtime = datetime.fromtimestamp(card_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            cards_table.add_row(card_file.name, f"{size_kb:.1f} KB", mtime)

            # Show content preview if detailed
            if detailed:
                try:
                    with open(card_file) as f:
                        data = json.load(f)

                    if "benchmark_card" in data:
                        card = data["benchmark_card"]
                        details = card.get("benchmark_details", {})
                        console.print(f"\n[dim]Preview of {card_file.name}:[/dim]")
                        console.print(f"  📝 Name: [cyan]{details.get('name', 'N/A')}[/cyan]")
                        console.print(f"  🏷️ Domains: {', '.join(details.get('domains', []))}")
                        console.print(f"  🌐 Languages: {', '.join(details.get('languages', []))}")

                        overview = details.get("overview", "")
                        if overview:
                            preview = overview[:150] + "..." if len(overview) > 150 else overview
                            console.print(f"  📖 Overview: [dim]{preview}[/dim]")
                except Exception as e:
                    console.print(f"  [red]Error reading {card_file.name}: {e}[/red]")

        console.print(cards_table)
        console.print()

    # Footer with helpful commands
    console.print(
        Panel(
            "[bold]Helpful Commands:[/bold]\n\n"
            f"[cyan]cd {session_dir}[/cyan] - Navigate to session directory\n"
            f"[cyan]benchmarkcard show {session_path} --detailed[/cyan] - Show detailed info",
            border_style="dim",
            title="[dim]Quick Actions[/dim]",
        )
    )


@app.command("validate")
def validate_setup(
    fix_issues: Annotated[
        bool,
        typer.Option(
            "--fix",
            help="Automatically fix issues where possible",
            rich_help_panel="🔧 Repair Options",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed validation information",
            rich_help_panel="📊 Display Options",
        ),
    ] = False,
) -> None:
    """
    🔍 Comprehensive system setup validation and diagnostics.

    Performs thorough validation of all system components including:
    • Environment variables and API keys
    • Python dependencies and imports
    • External tools (Merlin binary)
    • Directory structure and permissions
    • Configuration integrity

    [bold green]Examples:[/bold green]

        [dim]# Basic validation[/dim]
        benchmarkcard validate

        [dim]# Detailed validation with auto-fix[/dim]
        benchmarkcard validate --fix --verbose
    """
    setup_logging(verbose=verbose)

    display_banner()

    console.print(Rule("[bold yellow]System Validation[/bold yellow]", style="yellow"))
    console.print()

    issues = []
    warnings = []
    fixed_issues = []

    # Validation progress
    with create_progress_display() as progress:
        # Environment validation
        env_task = progress.add_task("[cyan]Checking environment variables...", total=4)

        try:
            Config.validate_config()
            progress.console.print("[green]✅ Environment configuration valid[/green]")
        except ValueError as e:
            progress.console.print(f"[red]❌ Configuration error: {e}[/red]")
            issues.append(
                (
                    "Environment Configuration",
                    str(e),
                    "Set required environment variables",
                )
            )

        progress.update(env_task, advance=1)

        # Check individual environment variables
        env_vars = [
            ("RITS_API_KEY", "RITS API authentication key"),
            ("RITS_MODEL", "RITS model identifier"),
            ("RITS_API_URL", "RITS API base URL"),
        ]

        for var, desc in env_vars:
            value = Config.get_env_var(var)
            if value:
                if verbose:
                    preview = f"{value[:10]}..." if len(value) > 10 else value
                    progress.console.print(f"[green]✅ {var}:[/green] [dim]{preview}[/dim]")
            else:
                progress.console.print(f"[red]❌ {var} not set[/red]")
                issues.append(
                    (
                        f"Environment Variable: {var}",
                        "Not set",
                        f"Set {var} in .env file",
                    )
                )
            progress.update(env_task, advance=1)

        # Python dependencies validation
        deps_task = progress.add_task("[cyan]Validating Python dependencies...", total=6)

        critical_imports = [
            ("workflow", "Main workflow orchestrator"),
            ("config", "Configuration management"),
            ("tools.unitxt.unitxt_tool", "UnitXT benchmark lookup"),
            ("tools.factreasoner.factreasoner_tool", "FactReasoner validation"),
            ("risk_atlas_nexus.library", "Risk Atlas Nexus integration"),
            ("typer", "CLI framework"),
        ]

        for module, description in critical_imports:
            try:
                __import__(module)
                if verbose:
                    progress.console.print(f"[green]✅ {description}[/green]")
            except ImportError as e:
                progress.console.print(f"[red]❌ {description}: {e}[/red]")
                issues.append((f"Python Import: {module}", str(e), "Install missing dependencies"))
            progress.update(deps_task, advance=1)

        # External tools validation
        tools_task = progress.add_task("[cyan]Checking external tools...", total=1)

        merlin_path = Config.MERLIN_BIN
        if merlin_path.exists():
            if os.access(merlin_path, os.X_OK):
                progress.console.print(f"[green]✅ Merlin binary: {merlin_path}[/green]")
            else:
                progress.console.print(f"[red]❌ Merlin binary not executable: {merlin_path}[/red]")
                issues.append(("Merlin Binary", "Not executable", "Check file permissions"))
        else:
            progress.console.print(f"[red]❌ Merlin binary not found: {merlin_path}[/red]")
            issues.append(
                (
                    "Merlin Binary",
                    "File not found",
                    "Build Merlin following README instructions",
                )
            )

        progress.update(tools_task, advance=1)

        # Directory structure validation
        dirs_task = progress.add_task("[cyan]Validating directories...", total=3)

        directories_to_check = [
            (Config.FACTREASONER_CACHE_DIR, "FactReasoner cache", True),  # Can create
            ("output", "Output directory", True),  # Can create
            (".", "Current directory", False),  # Must exist
        ]

        for dir_path, desc, can_create in directories_to_check:
            path_obj = Path(dir_path)

            if path_obj.exists():
                if verbose:
                    progress.console.print(f"[green]✅ {desc}: {path_obj.absolute()}[/green]")
            elif can_create:
                if fix_issues:
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                        progress.console.print(
                            f"[yellow]🔧 Created {desc}: {path_obj.absolute()}[/yellow]"
                        )
                        fixed_issues.append(f"Created directory: {path_obj}")
                    except Exception as e:
                        progress.console.print(f"[red]❌ Cannot create {desc}: {e}[/red]")
                        issues.append((f"Directory: {desc}", str(e), "Create directory manually"))
                else:
                    progress.console.print(
                        f"[yellow]⚠️ {desc} missing: {path_obj.absolute()}[/yellow]"
                    )
                    warnings.append(
                        (
                            f"Directory: {desc}",
                            "Does not exist",
                            "Will be created automatically",
                        )
                    )
            else:
                progress.console.print(f"[red]❌ {desc} not found: {path_obj.absolute()}[/red]")
                issues.append(
                    (
                        f"Directory: {desc}",
                        "Required directory missing",
                        "Check installation",
                    )
                )

            progress.update(dirs_task, advance=1)

    console.print()

    # Results summary
    if issues or warnings or fixed_issues:
        # Issues table
        if issues:
            issues_table = Table(title="❌ Issues Found", border_style="red")
            issues_table.add_column("Component", style="red", width=25)
            issues_table.add_column("Problem", style="yellow", width=30)
            issues_table.add_column("Solution", style="green")

            for component, problem, solution in issues:
                issues_table.add_row(component, problem, solution)

            console.print(issues_table)
            console.print()

        # Warnings table
        if warnings:
            warnings_table = Table(title="⚠️ Warnings", border_style="yellow")
            warnings_table.add_column("Component", style="yellow", width=25)
            warnings_table.add_column("Issue", style="white", width=30)
            warnings_table.add_column("Recommendation", style="dim")

            for component, issue, recommendation in warnings:
                warnings_table.add_row(component, issue, recommendation)

            console.print(warnings_table)
            console.print()

        # Fixed issues
        if fixed_issues:
            console.print("[bold green]🔧 Issues Fixed:[/bold green]")
            for fix in fixed_issues:
                console.print(f"  ✅ {fix}")
            console.print()

    # Final status
    if issues:
        display_error(
            f"Validation failed with {len(issues)} critical issue(s)",
            "Please resolve the issues above before proceeding.",
        )
        raise typer.Exit(1)

    elif warnings:
        display_success(
            f"Validation completed with {len(warnings)} warning(s)",
            "System is functional but some optimizations are recommended.",
        )

    else:
        display_success(
            "All validation checks passed!",
            "System is fully configured and ready for use.",
        )


# ========================================================================================
# CALLBACK & MAIN
# ========================================================================================


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
) -> None:
    """Benchmark metadata extraction and validation CLI.

    A comprehensive, production-ready tool for processing AI benchmarks with:
    - Multi-source metadata extraction (UnitXT, HuggingFace, Papers)
    - LLM-powered benchmark card generation
    - AI risk assessment integration
    - RAG-based evidence retrieval
    - Factual accuracy validation

    Quick Start:
        benchmarkcard validate        # Verify system setup
        benchmarkcard process glue    # Process a benchmark
        benchmarkcard list            # View recent sessions

    Args:
        ctx: Typer context for command invocation state.
    """

    # Show help if no command provided
    if ctx.invoked_subcommand is None:
        display_banner()
        console.print(ctx.get_help())

        # Show quick examples
        console.print("\n[bold green]Quick Examples:[/bold green]\n")
        examples = [
            ("Validate system setup", "benchmarkcard validate"),
            ("Process a benchmark", "benchmarkcard process glue"),
            ("List recent sessions", "benchmarkcard list --recent 5"),
        ]

        for desc, cmd in examples:
            console.print(f"  [dim]{desc}:[/dim] [cyan]{cmd}[/cyan]")

        console.print(
            f"\n[dim]For detailed help: [/dim][cyan]benchmarkcard <command> --help[/cyan]"
        )


# ========================================================================================
# ENTRY POINT
# ========================================================================================

if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        error_console.print(f"\n[bold red]💥 Unexpected error: {e}[/bold red]")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback

            error_console.print("[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)

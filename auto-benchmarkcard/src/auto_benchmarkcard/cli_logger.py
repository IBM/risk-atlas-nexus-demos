"""
CLI Logger for BenchmarkCard workflow integration.

This module provides a custom logger that integrates the agents workflow logging
with the CLI's rich console interface, providing clean step-by-step output.
"""

from typing import Any

from rich.console import Console
from rich.status import Status


class WorkflowCLILogger:
    """Custom logger that formats workflow messages for CLI display.

    This logger integrates with Rich console components to provide formatted
    output for the agents workflow, including step tracking, completion messages,
    and selective message display.
    """

    def __init__(self, status_spinner: Status, console: Console) -> None:
        """Initialize the CLI logger.

        Args:
            status_spinner: Rich status spinner to update with current step.
            console: Rich console for output formatting.
        """
        self.status_spinner = status_spinner
        self.console = console

        # Mapping of log messages to step names for spinner updates (START messages only)
        self.step_mapping = {
            "Starting metadata extraction": "UnitXT Metadata Extraction",
            "Starting ID and URL extraction": "ID and URL Extraction",
            "Starting HuggingFace extraction": "HuggingFace Extraction",
            "Starting paper extraction": "Academic Paper Processing",
            "Starting benchmark card composition": "BenchmarkCard Composition with LLM",
            "Starting risk identification": "Risk Identification",
            "Starting RAG processing": "RAG Evidence Retrieval",
            "Starting factuality evaluation": "Factual Accuracy Validation",
        }

        # Messages that indicate step completion
        self.completion_messages = {
            "UnitXT metadata retrieved",
            "ID extraction completed",
            "HuggingFace metadata retrieved successfully",
            "Docling extraction completed successfully",
            "Successfully composed benchmark card",
            "Risk identification completed",
            "RAG processing completed",
            "FactReasoner evaluation complete",
        }

        # Prefixes that indicate snippet information
        self.snippet_prefixes = {
            "Found:",
            "Extracted:",
            "Dataset:",
            "Paper:",
            "Card:",
            "Risks:",
            "RAG:",
            "Factuality:",
        }

        # Messages to suppress for cleaner output
        self.suppress_patterns = {
            "saved to:",
            "output saved",
            "Starting",
            "Using custom",
        }

    def info(self, msg: str, *args: Any) -> None:
        """Handle info level messages.

        Processes and displays info messages with appropriate formatting based on
        message content. Updates spinner status for step transitions, highlights
        completion messages, and formats snippet information.

        Args:
            msg: The message string to log, optionally with format placeholders.
            *args: Variable arguments for string formatting.
        """
        if args:
            msg = msg % args

        # Update spinner status based on current step
        for step_msg, step_name in self.step_mapping.items():
            if step_msg in msg:
                self.status_spinner.update(f"[blue]{step_name}...[/blue]")
                break

        # Show step completion messages with ✅
        if any(completion_msg in msg for completion_msg in self.completion_messages):
            self.console.print(f"✅ [green]{msg}[/green]")
            return

        # Show snippet information with indentation
        if any(msg.startswith(prefix) for prefix in self.snippet_prefixes):
            self.console.print(f"   [dim]{msg}[/dim]")
            return

        # Suppress noisy messages, show others dimmed
        if not any(suppress in msg for suppress in self.suppress_patterns):
            self.console.print(f"[dim]{msg}[/dim]")

    def warning(self, msg: str, *args: Any) -> None:
        """Handle warning level messages.

        Displays warning messages with yellow formatting and warning icon.

        Args:
            msg: The warning message string to log, optionally with format placeholders.
            *args: Variable arguments for string formatting.
        """
        if args:
            msg = msg % args
        self.console.print(f"⚠️ [yellow]{msg}[/yellow]")

    def error(self, msg: str, *args: Any, exc_info: bool = False) -> None:
        """Handle error level messages.

        Displays error messages with red formatting and error icon.

        Args:
            msg: The error message string to log, optionally with format placeholders.
            *args: Variable arguments for string formatting.
            exc_info: Whether to include exception info (currently not used).
        """
        if args:
            msg = msg % args
        self.console.print(f"❌ [red]{msg}[/red]")

    def debug(self, msg: str, *args: Any) -> None:
        """Handle debug level messages.

        Debug messages are currently suppressed and not displayed.

        Args:
            msg: The debug message string (ignored).
            *args: Variable arguments for string formatting (ignored).
        """
        pass

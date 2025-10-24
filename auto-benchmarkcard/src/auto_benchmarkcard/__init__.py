"""BenchmarkCard - Comprehensive benchmark metadata extraction and validation.

This package provides tools for extracting, validating, and enhancing AI benchmark
metadata through LLM-powered analysis, risk assessment, and factual verification.

"""

__version__ = "0.1.0"
__author__ = "Your Name"

from auto_benchmarkcard.config import Config
from auto_benchmarkcard.workflow import OutputManager, build_workflow

__all__ = [
    "Config",
    "build_workflow",
    "OutputManager",
]

"""Entry point for running benchmarkcard as a module.

This allows the package to be run as:
    python -m benchmarkcard process <benchmark_name>
"""

import sys

from auto_benchmarkcard.cli import app

if __name__ == "__main__":
    app()

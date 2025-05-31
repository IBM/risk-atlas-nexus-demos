"""Define the configurable parameters for the agent."""

from __future__ import annotations
from pathlib import Path
from dataclasses import fields
from typing import Optional, List, Union, Dict

from langchain_core.runnables import RunnableConfig


def from_runnable_config(cls, config: Optional[RunnableConfig] = None):
    """Create a Configuration instance from a RunnableConfig object."""
    configurable = config.get("configurable", {})
    _fields = {f.name for f in fields(cls) if f.init}
    return cls(**{k: v for k, v in configurable.items() if k in _fields})


def create_state_config(cls, config: Dict = None):
    """Create a Configuration instance from a RunnableConfig object."""
    configurable = config if config else {}
    return cls(**{k: v for k, v in configurable.items()})


def append_element_to_list(element1: List, element2: Union[None, str]):
    if element2:
        element1.append(element2)
        return element1
    return []

import json
from enum import StrEnum
from functools import partial
from pathlib import Path, PurePath
from typing import Any

import yaml
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from gaf_guard.core.agents import Agent
from gaf_guard.toolkit.enums import MessageType, Role, Serializer
from gaf_guard.toolkit.logging import configure_logger


LOGGER = configure_logger(__name__)
yaml.SafeDumper.add_multi_representer(
    StrEnum,
    yaml.representer.SafeRepresenter.represent_str,
)


# Graph state
class TrialLoggerAgentState(BaseModel):
    step_name: str
    step_role: Role
    step_type: MessageType
    content: Any


# Node
def yaml_serializer(
    trial_dir: str, state: TrialLoggerAgentState, config: RunnableConfig
):
    """Serialize Python object to YAML and write to file."""
    trial_name = config.get("metadata", {}).get("trial_name", "Trials_")
    file_path = Path(PurePath(trial_dir, trial_name + ".yaml"))

    try:
        step_data = state.model_dump(mode="json")
        if file_path.exists():
            trial_data = yaml.load(file_path.read_text(), Loader=yaml.SafeLoader)
            trial_data.append(step_data)
        else:
            trial_data = [step_data]

        yaml.dump(trial_data, open(file_path, "w"), default_flow_style=False)
    except Exception as e:
        raise ValueError(f"YAML Serialization failed: {e}")


# Node
def json_serializer(
    trial_dir: str, state: TrialLoggerAgentState, config: RunnableConfig
):
    """Serialize Python object to JSON string and write to file."""
    trial_name = config.get("metadata", {}).get("trial_name", "Trials_")
    file_path = Path(PurePath(trial_dir, trial_name + ".json"))
    try:
        if file_path.exists():
            trial_data = json.loads(file_path.read_text())
            trial_data.append(state.model_dump())
        else:
            trial_data = [state.model_dump()]

        json.dump(trial_data, open(file_path, "w"), indent=4)
    except Exception as e:
        raise ValueError(f"JSON Serialization failed: {e}")


# Node
def call_serializer(serializer: str, state: TrialLoggerAgentState) -> None:
    if Serializer[serializer] == Serializer.JSON:
        return "json_serializer"
    elif Serializer[serializer] == Serializer.YAML:
        return "yaml_serializer"
    else:
        LOGGER.error(
            f"Logging Failed. Invalid serializer: {serializer}. Valid serializers: {Serializer._member_names_}"
        )


class TrialLoggerAgent(Agent):
    """
    Initializes a new instance of the TrialLogger Agent class.
    """

    _WORKFLOW_NAME = "Trial Logger Agent"

    def __init__(self):
        super(TrialLoggerAgent, self).__init__(TrialLoggerAgentState)

    def _build_graph(self, graph: StateGraph, trial_dir: str, serializer: str):

        # Add nodes
        graph.add_node("json_serializer", partial(json_serializer, trial_dir))
        graph.add_node("yaml_serializer", partial(yaml_serializer, trial_dir))

        # Add edges to connect nodes
        graph.add_conditional_edges(
            source=START,
            path=partial(call_serializer, serializer),
            path_map=["json_serializer", "yaml_serializer"],
        )
        graph.add_edge("json_serializer", END)
        graph.add_edge("yaml_serializer", END)

import json
import operator
import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from pydantic import BaseModel
from rich.console import Console
from risk_atlas_nexus.blocks.inference import InferenceEngine

from agentic_governance.agents import Agent
from agentic_governance.toolkit.decorators import async_partial, hline, step_logging


console = Console()


# Graph state
class StreamAgentState(BaseModel):
    prompt: Optional[str]


# Node
@step_logging("Manage Input Prompt", at="both")
async def manage_input_prompt(state: StreamAgentState):
    print(state.identified_risks)
    return {}


# Node
@step_logging("Stream Input Prompt", at="both")
async def stream_input_prompt(state: StreamAgentState):
    print(state.identified_risks)
    return {"prompt": state.prompt}


class StreamAgent(Agent):
    """
    Initializes a new instance of the Human in the Loop Agent class.
    """

    _WORKFLOW_NAME = "Stream Agent"
    _WORKFLOW_DESC = f"[bold blue]Stream Input Prompt to the agents:"

    def __init__(self):
        super(StreamAgent, self).__init__(StreamAgentState)

    def _build_graph(self, graph: StateGraph, inference_engine: InferenceEngine):

        # Add nodes
        graph.add_node("Manage Input Prompt", manage_input_prompt)
        graph.add_node("Stream Input Prompt", stream_input_prompt)

        # Add edges to connect nodes
        graph.add_edge(START, "Manage Input Prompt")
        graph.add_edge("Manage Input Prompt", "Stream Input Prompt")
        graph.add_edge("Stream Input Prompt", END)

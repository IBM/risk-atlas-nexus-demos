import operator
from typing import Annotated, Any, Dict, Optional, List
import json
import os
from pathlib import Path
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from pydantic import BaseModel
from rich.console import Console
from risk_atlas_nexus.blocks.inference import InferenceEngine
from risk_atlas_nexus.library import RiskAtlasNexus

from agentic_governance.agents import Agent
from agentic_governance.toolkit.decorators import async_partial, hline, step_logging
from agentic_governance.toolkit.tmp_utils import workflow_table_2


console = Console()


# Graph state
class HumanInTheLoopAgentState(BaseModel):
    identified_risks: Optional[List[str]] = None


# Node
@step_logging("Gather AI Risks", at="both")
async def gather_ai_risks(state: HumanInTheLoopAgentState):
    print(state.identified_risks)
    return {}


# Node
@step_logging("Getting Human Response on AI Risks", at="both")
async def get_human_response(state: HumanInTheLoopAgentState):
    print(state.identified_risks)
    return {}


# Node
@step_logging("Update AI Risks", at="both")
async def update_ai_risks(state: HumanInTheLoopAgentState):
    print(state.identified_risks)
    return {"identified_risks": state.identified_risks}


class HumanInTheLoopAgent(Agent):
    """
    Initializes a new instance of the Human in the Loop Agent class.
    """

    _WORKFLOW_NAME = "Human In the Loop Agent"
    _WORKFLOW_DESC = f"[bold blue]Getting Response from the User:"

    def __init__(self):
        super(HumanInTheLoopAgent, self).__init__(HumanInTheLoopAgentState)

    def _build_graph(self, graph: StateGraph, inference_engine: InferenceEngine):

        # Add nodes
        graph.add_node("Gather AI Risks", gather_ai_risks)
        graph.add_node("Get Human Response on AI Risks", get_human_response)
        graph.add_node("Update AI Risks", update_ai_risks)

        # Add edges to connect nodes
        graph.add_edge(START, "Gather AI Risks")
        graph.add_edge("Gather AI Risks", "Get Human Response on AI Risks")
        graph.add_edge("Get Human Response on AI Risks", "Update AI Risks")
        graph.add_edge("Update AI Risks", END)

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter, interrupt
from pydantic import BaseModel
from rich.console import Console
from risk_atlas_nexus.blocks.inference import InferenceEngine
from risk_atlas_nexus.library import RiskAtlasNexus

from gaf_guard.agents import Agent
from gaf_guard.toolkit.decorators import config, step_logging
from gaf_guard.toolkit.enums import MessageType
from gaf_guard.toolkit.tmp_utils import workflow_table_2


console = Console()


# Config schema
@dataclass(kw_only=True)
class HumanInTheLoopAgentConfig:
    trial_file: Optional[str] = None


# Graph state
class HumanInTheLoopAgentState(BaseModel):
    identified_risks: Optional[List[str]] = None


# Node
@config(config_class=HumanInTheLoopAgentConfig)
@step_logging("Gather AI Risks for Human Intervention", at="both", benchmark="log")
async def gather_ai_risks(
    state: HumanInTheLoopAgentState, config: HumanInTheLoopAgentConfig
):
    return {"log": state.identified_risks}


# Node
@step_logging("Getting Human Response on AI Risks")
async def get_human_response(state: HumanInTheLoopAgentState):
    syntax_error = False
    while True:
        updated_risks = interrupt(
            {
                "message": (
                    ("\nSyntax Error, Try Again." if syntax_error else "")
                    + f"\nPlease Accept (Press Enter) or Suggest edits for AI Risks (Type your answer as a python List)"
                )
            }
        )
        try:
            if len(updated_risks["response"]) > 0:
                updated_risks = json.loads(updated_risks["response"])
            else:
                updated_risks = state.identified_risks
            break
        except:
            syntax_error = True

    return {"identified_risks": updated_risks}


# Node
@config(config_class=HumanInTheLoopAgentConfig)
@step_logging("Updated AI Risks from Human Response", at="both", benchmark="log")
async def updated_ai_risks(
    state: HumanInTheLoopAgentState, config: HumanInTheLoopAgentConfig
):
    return {"log": state.identified_risks}


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
        graph.add_node("Updated AI Risks", updated_ai_risks)

        # Add edges to connect nodes
        graph.add_edge(START, "Gather AI Risks")
        graph.add_edge("Gather AI Risks", "Get Human Response on AI Risks")
        graph.add_edge("Get Human Response on AI Risks", "Updated AI Risks")
        graph.add_edge("Updated AI Risks", END)

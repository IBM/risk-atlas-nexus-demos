import json
from typing import List, Optional

from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter, interrupt
from pydantic import BaseModel
from rich.console import Console
from risk_atlas_nexus.blocks.inference import InferenceEngine

from gaf_guard.agents import Agent
from gaf_guard.toolkit.decorators import workflow_step
from gaf_guard.toolkit.enums import MessageType, Role
from gaf_guard.toolkit.exceptions import HumanInterruptionException
from gaf_guard.toolkit.models import WorkflowStepMessage


console = Console()


# Graph state
class HumanInTheLoopAgentState(BaseModel):
    identified_risks: Optional[List[str]] = None


# Node
@workflow_step(step_name="Gather AI Risks for Human Intervention", log=True)
def gather_ai_risks(state: HumanInTheLoopAgentState, config: RunnableConfig):
    return {"identified_risks": state.identified_risks}


# Node
def get_human_response(state: HumanInTheLoopAgentState, config: RunnableConfig):
    syntax_error = False
    while True:
        try:
            updated_risks = interrupt(
                WorkflowStepMessage(
                    step_type=MessageType.HITL_QUERY,
                    content=(
                        ("\nSyntax Error, Try Again." if syntax_error else "")
                        + f"\nPlease Accept (Press Enter) or Suggest edits for AI Risks (Type your answer as a python List)"
                    ),
                    step_name="Human Intervention",
                    step_role=Role.SYSTEM,
                ).model_dump()
            )
        except GraphInterrupt as e:
            raise HumanInterruptionException(json.dumps(e.args[0][0].value))

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
@workflow_step(
    step_name="Updated AI Risks from Human Response", step_role=Role.USER, log=True
)
def updated_ai_risks(state: HumanInTheLoopAgentState, config: RunnableConfig):
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
        graph.add_node("Updated AI Risks", updated_ai_risks)

        # Add edges to connect nodes
        graph.add_edge(START, "Gather AI Risks")
        graph.add_edge("Gather AI Risks", "Get Human Response on AI Risks")
        graph.add_edge("Get Human Response on AI Risks", "Updated AI Risks")
        graph.add_edge("Updated AI Risks", END)

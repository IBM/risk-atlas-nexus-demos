import json
from functools import partial
from typing import List, Optional

from langchain_core.runnables.config import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress

from gaf_guard.agents import Agent
from gaf_guard.toolkit.decorators import workflow_step
from gaf_guard.toolkit.enums import MessageType, Role
from gaf_guard.toolkit.models import WorkflowStepMessage


STATUS_DISPLAY = {}


# Graph state
class OrchestratorState(BaseModel):
    user_intent: Optional[str] = None
    prompt: Optional[str] = None
    domain: Optional[str] = None
    drift_value: Optional[int] = None
    identified_risks: Optional[List[str]] = ["a", "b"]


# Node
def create_live_display(state: OrchestratorState, config: RunnableConfig):
    progress = Progress()
    live = Live(console=Console())
    live.start()

    STATUS_DISPLAY[config["configurable"]["thread_id"]] = {
        "live": live,
        "progress": progress,
        "current_task": None,
    }


# Node
@workflow_step(step_name="User Intent", step_role=Role.USER, publish=False, log=True)
def user_intent(state: OrchestratorState, config: RunnableConfig):
    return {"user_intent": state.user_intent}


# Node
def next_agent(agent: Agent, state: OrchestratorState, config: RunnableConfig):
    display = STATUS_DISPLAY.get(config["configurable"]["thread_id"])
    if display["current_task"]:
        display["progress"].update(
            display["current_task"]["task_id"],
            completed=100,
            description=f"[bold yellow]Invoking Agent[/bold yellow][bold white]...{display['current_task']['name']}[/bold white][bold yellow]...Completed[/bold yellow]",
            refresh=True,
        )

    task_id = display["progress"].add_task(
        f"[bold yellow]Invoking Agent[/bold yellow][bold white]...{agent._WORKFLOW_NAME}[/bold white]",
        total=None,
    )
    display["live"].update(
        Panel(
            Group(
                f"Incoming request:\n{json.dumps(state.model_dump(include=set({'user_intent', 'prompt'}), exclude_none=True), indent=2)}\n",
                display["progress"],
            ),
            title=f"{config['configurable']['trial_file'].split('/')[-1].split('.')[0]} | Client: {config['configurable']['thread_id']}",
        ),
        refresh=True,
    )
    display["current_task"] = {"task_id": task_id, "name": agent._WORKFLOW_NAME}

    if agent._WORKFLOW_NAME:
        get_stream_writer()(
            WorkflowStepMessage(
                step_name=agent._WORKFLOW_NAME,
                step_type=MessageType.WORKFLOW_STARTED,
                step_role=Role.SYSTEM,
            )
        )

    return agent._WORKFLOW_NAME


class OrchestratorAgent(Agent):
    """
    Initializes a new instance of the Orchestrator Agent class.
    """

    def __init__(self):
        super(OrchestratorAgent, self).__init__(OrchestratorState)

    def _build_graph(
        self,
        graph: StateGraph,
        RiskGeneratorAgent: Agent,
        HumanInTheLoopAgent: Agent,
        StreamAgent: Agent,
        RisksAssessmentAgent: Agent,
        DriftMonitoringAgent: Agent,
    ):

        # Add nodes
        graph.add_node("Create Live Display", create_live_display)
        graph.add_node("User Intent", user_intent)

        graph.add_node(RiskGeneratorAgent._WORKFLOW_NAME, RiskGeneratorAgent.workflow)
        graph.add_node(HumanInTheLoopAgent._WORKFLOW_NAME, HumanInTheLoopAgent.workflow)
        graph.add_node(StreamAgent._WORKFLOW_NAME, StreamAgent.workflow)
        graph.add_node(
            RisksAssessmentAgent._WORKFLOW_NAME, RisksAssessmentAgent.workflow
        )
        graph.add_node(
            DriftMonitoringAgent._WORKFLOW_NAME, DriftMonitoringAgent.workflow
        )

        # Add edges
        graph.add_edge(START, "Create Live Display")
        graph.add_edge("Create Live Display", "User Intent")
        graph.add_conditional_edges(
            source="User Intent",
            path=partial(next_agent, RiskGeneratorAgent),
            path_map=[RiskGeneratorAgent._WORKFLOW_NAME],
        )
        graph.add_conditional_edges(
            source=RiskGeneratorAgent._WORKFLOW_NAME,
            path=partial(next_agent, HumanInTheLoopAgent),
            path_map=[HumanInTheLoopAgent._WORKFLOW_NAME],
        )
        graph.add_conditional_edges(
            source=HumanInTheLoopAgent._WORKFLOW_NAME,
            path=partial(next_agent, StreamAgent),
            path_map=[StreamAgent._WORKFLOW_NAME],
        )
        graph.add_conditional_edges(
            source=StreamAgent._WORKFLOW_NAME,
            path=partial(next_agent, RisksAssessmentAgent),
            path_map=[RisksAssessmentAgent._WORKFLOW_NAME, END],
        )
        graph.add_conditional_edges(
            source=RisksAssessmentAgent._WORKFLOW_NAME,
            path=partial(next_agent, DriftMonitoringAgent),
            path_map=[DriftMonitoringAgent._WORKFLOW_NAME],
        )
        graph.add_conditional_edges(
            source=DriftMonitoringAgent._WORKFLOW_NAME,
            path=partial(next_agent, StreamAgent),
            path_map=[StreamAgent._WORKFLOW_NAME],
        )

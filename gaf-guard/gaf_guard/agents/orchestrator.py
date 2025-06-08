import json
from functools import partial
from typing import Dict, List, Optional

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress

from gaf_guard.agents import Agent
from gaf_guard.toolkit.conn_manager import conn_manager
from gaf_guard.toolkit.decorators import hline
from gaf_guard.toolkit.enums import MessageType, Role


STATUS_DISPLAY = {}


# Graph state
class OrchestratorState(BaseModel):
    user_intent: Optional[str] = None
    prompt: Optional[str] = None
    domain: Optional[str] = None
    drift_value: Optional[int] = None
    identified_risks: Optional[List[str]] = None


# Node
async def create_live_display(state: OrchestratorState, config: RunnableConfig):
    progress = Progress()
    live = Live(console=Console())
    live.start()

    STATUS_DISPLAY[config["metadata"]["client_id"]] = {
        "live": live,
        "progress": progress,
        "current_task": None,
    }


# Node
async def user_intent(state: OrchestratorState, config: RunnableConfig):
    await conn_manager.log_benchmark(
        state.user_intent, "User Intent", Role.USER, config["metadata"]["trial_file"]
    )


# Node
# @hline("End of Workflow", at="end")
async def next_agent(agent: Agent, state: OrchestratorState, config: RunnableConfig):
    display = STATUS_DISPLAY.get(config["metadata"]["client_id"])
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
            title=f"{config['metadata']['trial_name']} | Client: {config['metadata']['client_id']}",
        ),
        refresh=True,
    )
    display["current_task"] = {"task_id": task_id, "name": agent._WORKFLOW_NAME}

    if agent._WORKFLOW_NAME:
        await conn_manager.send(
            f"Workflow: [bold blue]{agent._WORKFLOW_NAME}[/]",
            message_type=MessageType.RULE,
            spacing="both",
        )

    return agent._WORKFLOW_NAME


class OrchestratorAgent(Agent):
    """
    Initializes a new instance of the Orchestrator Agent class.
    """

    def __init__(self):
        super(OrchestratorAgent, self).__init__(OrchestratorState)

    def _build_graph(self, graph: StateGraph, agents: List[Agent]):

        # Add nodes and edges
        graph.add_node("Create Live Display", create_live_display)
        graph.add_node("User Intent", user_intent)

        graph.add_edge(START, "Create Live Display")
        graph.add_edge("Create Live Display", "User Intent")

        for agent in agents:
            graph.add_node(agent._WORKFLOW_NAME, agent.workflow)
            if agent._WORKFLOW_NAME == "Risk Generation Agent":
                graph.add_conditional_edges(
                    source="User Intent",
                    path=partial(next_agent, agent),
                    path_map=[agent._WORKFLOW_NAME],
                )
            elif agent._WORKFLOW_NAME == "Human In the Loop Agent":
                graph.add_conditional_edges(
                    source="Risk Generation Agent",
                    path=partial(next_agent, agent),
                    path_map=[agent._WORKFLOW_NAME],
                )
            elif agent._WORKFLOW_NAME == "Stream Agent":
                graph.add_conditional_edges(
                    source="Human In the Loop Agent",
                    path=partial(next_agent, agent),
                    path_map=[agent._WORKFLOW_NAME],
                )
                graph.add_conditional_edges(
                    source="Drift Monitoring Agent",
                    path=partial(next_agent, agent),
                    path_map=[agent._WORKFLOW_NAME],
                )
            elif agent._WORKFLOW_NAME == "Risk Asssessment Agent":
                graph.add_conditional_edges(
                    source="Stream Agent",
                    path=partial(next_agent, agent),
                    path_map=[agent._WORKFLOW_NAME, END],
                )
            elif agent._WORKFLOW_NAME == "Drift Monitoring Agent":
                graph.add_conditional_edges(
                    source="Risk Asssessment Agent",
                    path=partial(next_agent, agent),
                    path_map=[agent._WORKFLOW_NAME],
                )

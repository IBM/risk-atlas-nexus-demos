import datetime
import json
import operator
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional
from langgraph.types import interrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from risk_atlas_nexus.blocks.inference import InferenceEngine

from agentic_governance.agents import Agent
from agentic_governance.toolkit.conn_manager import conn_manager
from agentic_governance.toolkit.decorators import hline
from agentic_governance.toolkit.enums import MessageType
from agentic_governance.toolkit.tmp_utils import workflow_table
from langchain_core.runnables.config import RunnableConfig

console = Console()


# Graph state
class OrchestratorState(BaseModel):
    user_intent: Optional[str] = None
    prompt: Optional[str] = None
    domain: Optional[str] = None
    drift_value: Optional[int] = None


# Node
async def route_agent(
    agents: List[Agent], state: OrchestratorState, config: RunnableConfig
):
    if state.prompt:
        agent = list(
            filter(
                lambda agent: "prompt" in agent.graph.schema.model_fields,
                agents,
            )
        )[0]
    elif state.user_intent:
        agent = list(
            filter(
                lambda agent: "user_intent" in agent.graph.schema.model_fields,
                agents,
            )
        )[0]

    if agent._WORKFLOW_NAME:
        await conn_manager.send(
            f"Workflow: [bold blue]{agent._WORKFLOW_NAME}[/]",
            message_type=MessageType.RULE,
            spacing="both",
        )
    if agent._WORKFLOW_TABLE:
        await conn_manager.send(
            str(agent._WORKFLOW_TABLE),
            justify="center",
        )

    return Send(agent._WORKFLOW_NAME, dict(state))


class OrchestratorAgent(Agent):
    """
    Initializes a new instance of the Orchestrator Agent class.
    """

    def __init__(self):
        super(OrchestratorAgent, self).__init__(OrchestratorState)

    @hline("End of Workflow", at="end")
    async def ainvoke(self, state_dict: Dict, config: Dict = None):
        progress = Progress()
        with Live(
            Panel(
                Group(
                    f"Incoming request:\n{json.dumps(state_dict, indent=2)}\n",
                    progress,
                ),
                title=f"Client: {config['configurable']['client_id']}",
            ),
            console=console,
        ):
            task_id = progress.add_task(
                f"[bold yellow]Invoking Agent[/bold yellow][bold white]...{self.__class__.__name__}[/bold white]",
                total=None,
            )
            state_dict = await super().ainvoke(state_dict, config)
            progress.update(
                task_id,
                completed=100,
                description=f"[bold yellow]Invoking Agent[/bold yellow][bold white]...{self.__class__.__name__}[/bold white][bold yellow]...Completed[/bold yellow]",
            )

            return state_dict

    def _build_graph(self, graph: StateGraph, agents: List[Agent]):

        # Add nodes
        # graph.add_node("route_agent", route_agent)
        for agent in agents:
            graph.add_node(agent._WORKFLOW_NAME, agent.workflow)

        # Add edges
        graph.add_conditional_edges(
            source=START,
            path=partial(route_agent, agents),
            path_map=["Risk Generation Agent", "Risk Asssessment Agent"],
        )
        graph.add_edge("Risk Asssessment Agent", "Drift Monitoring Agent")
        graph.add_edge("Drift Monitoring Agent", END)
        graph.add_edge("Risk Generation Agent", END)

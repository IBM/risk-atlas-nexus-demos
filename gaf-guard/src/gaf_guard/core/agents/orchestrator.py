from functools import partial
from typing import List, Optional

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from gaf_guard.core.agents import Agent
from gaf_guard.core.decorators import workflow, workflow_step
from gaf_guard.toolkit.enums import Role


# Graph state
class OrchestratorState(BaseModel):
    user_intent: Optional[str] = None
    prompt: Optional[str] = None
    domain: Optional[str] = None
    drift_value: Optional[int] = None
    identified_risks: Optional[List[str]] = None


# Node
@workflow_step(step_name="User Intent", step_role=Role.USER)
def user_intent(state: OrchestratorState, config: RunnableConfig):
    return {"user_intent": state.user_intent}


# Node
@workflow(name="Invoking Agent", role=Role.SYSTEM)
def next_agent(agent: Agent, state: OrchestratorState, config: RunnableConfig):
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
        graph.add_edge(START, "User Intent")
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

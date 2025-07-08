from functools import partial
from typing import Optional

from jinja2 import Template
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from risk_atlas_nexus.blocks.inference import InferenceEngine

from gaf_guard.agents import Agent
from gaf_guard.templates import DRIFT_COT_TEMPLATE
from gaf_guard.toolkit.decorators import workflow_step


# Graph state
class DriftMonitoringState(BaseModel):
    domain: str
    prompt: str
    drift_value: int = 0
    incident_message: Optional[str] = None


# Nodes
@workflow_step(
    step_name="Drift Monitoring Setup", step_desc="Setting Initial Values:", log=True
)
def drift_monitoring_setup(state: DriftMonitoringState, config: RunnableConfig):
    return {
        "drift_value": state.drift_value,
        "drift_threshold": config["configurable"]["drift_threshold"],
    }


# Nodes
@workflow_step(step_name="Drift Monitoring", log=True)
def check_prompt_relevance(
    inference_engine: InferenceEngine,
    state: DriftMonitoringState,
    config: RunnableConfig,
):
    prompt_str = Template(DRIFT_COT_TEMPLATE).render(
        prompt=state.prompt,
        examples=config["configurable"]["drift_monitoring_cot"],
        domain=state.domain,
    )

    response = inference_engine.chat(
        messages=[prompt_str],
        response_format={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "explanation": {"type": "string"},
                "question": {"type": "string"},
            },
            "required": ["answer", "explanation", "question"],
        },
        postprocessors=["json_object"],
        verbose=False,
    )[0]

    if response.prediction["answer"].lower() == "other":
        state.drift_value += 1

    return {"drift_value": state.drift_value}


# Nodes
@workflow_step(step_name="Drift Reporting", log=True)
def drift_incident_reporting(state: DriftMonitoringState, config: RunnableConfig):
    if state.drift_value > config["configurable"]["drift_threshold"]:
        incident_message = f"[red]Alert: Potential drift in prompts identified.[/red]"
    else:
        incident_message = f"[green]No drift detected.[/green]"

    return {"incident_message": incident_message}


class DriftMonitoringAgent(Agent):
    """
    Initializes a new instance of the Risk Assessment Agent class.
    """

    _WORKFLOW_NAME = "Drift Monitoring Agent"

    def __init__(self):
        super(DriftMonitoringAgent, self).__init__(DriftMonitoringState)

    def _build_graph(self, graph: StateGraph, inference_engine: InferenceEngine):

        # Add nodes
        graph.add_node("Drift Monitoring Setup", drift_monitoring_setup)
        graph.add_node(
            "Check Prompt Relevance", partial(check_prompt_relevance, inference_engine)
        )
        graph.add_node("Drift Incident Reporting", drift_incident_reporting)

        # Add edges to connect nodes
        graph.add_edge(START, "Drift Monitoring Setup")
        graph.add_edge("Drift Monitoring Setup", "Check Prompt Relevance")
        graph.add_edge("Check Prompt Relevance", "Drift Incident Reporting")
        graph.add_edge("Drift Incident Reporting", END)

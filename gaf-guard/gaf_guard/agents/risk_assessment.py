import operator
from functools import partial
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from pydantic import BaseModel
from rich.console import Console
from risk_atlas_nexus.blocks.inference import InferenceEngine
from risk_atlas_nexus.library import RiskAtlasNexus

from gaf_guard.agents import Agent
from gaf_guard.toolkit.decorators import step_logging


console = Console()
risk_atlas_nexus = RiskAtlasNexus()


# Graph state
class RiskAssessmentState(BaseModel):
    prompt: Optional[str]
    risk_report: Annotated[Dict[str, str], operator.or_] = None


# Node
@step_logging(step="Risk Assessment", at="begin")
def start_assessing_for_risks(state: RiskAssessmentState, config: RunnableConfig):
    return {}


# Node
@step_logging(step="Risk Assessment", at="end")
def stop_assessing_for_risks(state: RiskAssessmentState, config: RunnableConfig):
    return {}


# Node
@step_logging("Assess Risk")
def assess_risk(
    risk_name,
    inference_engine: InferenceEngine,
    state: RiskAssessmentState,
    config: RunnableConfig,
):
    response = inference_engine.chat(
        messages=[
            [
                {"role": "system", "content": risk_name},
                {"role": "user", "content": state.prompt},
            ]
        ],
        verbose=False,
    )
    is_risk_present = response[0].prediction
    return {
        "risk_report": {risk_name: is_risk_present},
        "log": f"[bold blue]Check for {risk_name}: [/bold blue]{is_risk_present}",
    }


# Node
@step_logging(step="Incident Reporting", at="both", benchmark="risk_report")
def aggregate_and_report_incident(state: RiskAssessmentState, config: RunnableConfig):
    risk_report_yes = None
    if state.risk_report:
        risk_report_yes = dict(
            filter(lambda item: "Yes" in item[1], state.risk_report.items())
        )

    if risk_report_yes:
        incident_report = f"[bold red]Alert: Potential risks identified.[/bold red]\n{list(risk_report_yes.keys())}"
    else:
        incident_report = (
            "[bold green]No risks identified with the prompts[/bold green]"
        )

    return {"log": incident_report, "risk_report": state.risk_report}


class RisksAssessmentAgent(Agent):
    """
    Initializes a new instance of the Granite Guardian Risk Detector Agent class.
    """

    _WORKFLOW_NAME = "Risk Asssessment Agent"
    _WORKFLOW_DESC = (
        f"[bold blue]Real-time risk monitoring using the following workflow:"
    )

    def __init__(self):
        super(RisksAssessmentAgent, self).__init__(RiskAssessmentState)

    def _build_graph(self, graph: StateGraph, inference_engine: InferenceEngine):

        # Add nodes and edges
        graph.add_node("Start Risk Assessment", start_assessing_for_risks)
        graph.add_node(
            "Aggregate and Report Risk Incidents", aggregate_and_report_incident
        )
        graph.add_node("Stop Risk Assessment", stop_assessing_for_risks)

        graph.add_edge(START, "Start Risk Assessment")
        for risk_name in [
            risk.tag
            for risk in risk_atlas_nexus.get_all_risks(taxonomy="ibm-granite-guardian")
        ]:
            graph.add_node(
                risk_name,
                partial(assess_risk, risk_name, inference_engine),
            )
            graph.add_edge("Start Risk Assessment", risk_name)
            graph.add_edge(risk_name, "Stop Risk Assessment")

        graph.add_edge("Stop Risk Assessment", "Aggregate and Report Risk Incidents")
        graph.add_edge("Aggregate and Report Risk Incidents", END)

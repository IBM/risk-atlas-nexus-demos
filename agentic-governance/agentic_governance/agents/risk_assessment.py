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
from risk_atlas_nexus.library import RiskAtlasNexus

from agentic_governance.agents import Agent
from agentic_governance.toolkit.decorators import async_partial, hline, step_logging
from agentic_governance.toolkit.tmp_utils import workflow_table_2


console = Console()


# Graph state
class RiskAssessmentState(BaseModel):
    prompt: Optional[str]
    risk_report: Annotated[Dict[str, str], operator.or_] = None


# Node
@step_logging("Input Prompt")
async def input_prompt(state: RiskAssessmentState) -> Dict[str, Any]:
    return {
        "identified_risks": None,
        "log": {
            "message": "[bold blue]Prompt:[/bold blue] {data}",
            "data": state.prompt,
        },
    }


# Node
@step_logging(step="Risk Assessment", at="begin")
async def start_assessing_for_risks(state: RiskAssessmentState):
    return {"identified_risks": None}


# Node
@step_logging(step="Risk Assessment", at="end")
async def stop_assessing_for_risks(state: RiskAssessmentState):
    return {}


# Node
@step_logging("Assess Risk")
async def assess_risk(
    risk_name, inference_engine: InferenceEngine, state: RiskAssessmentState
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
@step_logging(step="Incident Reporting", at="both")
async def aggregate_and_report_incident(state: RiskAssessmentState):
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

    return {"log": {"message": incident_report, "data": state.risk_report}}


class RisksAssessmentAgent(Agent):
    """
    Initializes a new instance of the Granite Guardian Risk Detector Agent class.
    """

    _WORKFLOW_NAME = "Risk Asssessment Agent"
    _WORKFLOW_DESC = (
        f"[bold blue]Real-time risk monitoring using the following workflow:"
    )
    _WORKFLOW_TABLE = workflow_table_2()

    def __init__(self):
        super(RisksAssessmentAgent, self).__init__(RiskAssessmentState)

    def _build_graph(self, graph: StateGraph, inference_engine: InferenceEngine):

        # Add nodes and edges
        graph.add_node("Input Prompt", input_prompt)
        graph.add_node("Start Risk Assessment", start_assessing_for_risks)
        graph.add_node(
            "Aggregate and Report Risk Incidents", aggregate_and_report_incident
        )
        graph.add_node("Stop Risk Assessment", stop_assessing_for_risks)

        graph.add_edge(START, "Input Prompt")
        graph.add_edge("Input Prompt", "Start Risk Assessment")
        for risk_name in [
            risk.tag
            for risk in RiskAtlasNexus().get_all_risks(taxonomy="ibm-granite-guardian")
        ]:
            graph.add_node(
                risk_name,
                async_partial(assess_risk, risk_name, inference_engine),
            )
            graph.add_edge("Start Risk Assessment", risk_name)
            graph.add_edge(risk_name, "Stop Risk Assessment")

        graph.add_edge("Stop Risk Assessment", "Aggregate and Report Risk Incidents")
        graph.add_edge("Aggregate and Report Risk Incidents", END)

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel
from rich.console import Console
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk
from risk_atlas_nexus.blocks.inference import InferenceEngine
from risk_atlas_nexus.blocks.prompt_response_schema import LIST_OF_STR_SCHEMA
from risk_atlas_nexus.data import load_resource
from risk_atlas_nexus.library import RiskAtlasNexus

from gaf_guard.agents import Agent
from gaf_guard.templates import RISKS_GENERATION_COT_TEMPLATE
from gaf_guard.toolkit.decorators import config, step_logging
from gaf_guard.toolkit.enums import MessageType
from gaf_guard.toolkit.tmp_utils import workflow_table


console = Console()


# Config schema
@dataclass(kw_only=True)
class RiskGenerationConfig:
    trial_file: Optional[str] = None
    risk_questionnaire_cot: Optional[Dict[str, Any]] = None
    risk_generation_cot: Optional[Dict[str, Any]] = None


# Graph state
class RiskGenerationState(BaseModel):
    user_intent: str
    domain: Optional[str] = None
    risk_questionnaire: Optional[List[Dict[str, str]]] = None
    identified_risks: Optional[List[str]] = None
    identified_ai_tasks: Optional[List[str]] = None


# Node
@config(config_class=RiskGenerationConfig)
@step_logging(step="Domain Identification", at="both", benchmark="domain")
async def get_usecase_domain(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RiskGenerationConfig,
):
    domain = (
        RiskAtlasNexus()
        .identify_domain_from_usecases(
            [state.user_intent], inference_engine, verbose=False
        )[0]
        .prediction["answer"]
    )

    return {
        "domain": domain,
        "log": f"Identified domain from the user_intent: [bold yellow]{domain}[/bold yellow]",
    }


# Node
@config(config_class=RiskGenerationConfig)
@step_logging(
    step="Questionnaire Prediction",
    at="both",
    step_desc="Using Zero-shot method.",
    benchmark="risk_questionnaire",
)
async def generate_zero_shot(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RiskGenerationConfig,
):
    # load CoT examples for risk questionnaire
    if not config.risk_questionnaire_cot:
        risk_questionnaire = load_resource("risk_questionnaire_cot.json")
    else:
        risk_questionnaire = config.risk_questionnaire_cot

    responses = RiskAtlasNexus().generate_zero_shot_risk_questionnaire_output(
        state.user_intent, risk_questionnaire, inference_engine
    )

    risk_questionnaire = []
    for question_data, response in zip(risk_questionnaire, responses):
        risk_questionnaire.append(
            {
                "question": question_data["question"],
                "answer": response.prediction["answer"],
            }
        )

    return {"risk_questionnaire": risk_questionnaire, "log": risk_questionnaire}


# Node
@config(config_class=RiskGenerationConfig)
@step_logging(
    step="Questionnaire Prediction",
    at="both",
    step_desc="Chain of Thought (CoT) data found, using Few-shot method...",
    benchmark="risk_questionnaire",
)
async def generate_few_shot(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RiskGenerationConfig,
):
    # load CoT examples for risk questionnaire
    if not config.risk_questionnaire_cot:
        risk_questionnaire = load_resource("risk_questionnaire_cot.json")
    else:
        risk_questionnaire = config.risk_questionnaire_cot

    responses = RiskAtlasNexus().generate_few_shot_risk_questionnaire_output(
        state.user_intent,
        risk_questionnaire[1:],
        inference_engine,
        verbose=False,
    )

    risk_questionnaire_responses = []
    for question_data, response in zip(risk_questionnaire[1:], responses):
        risk_questionnaire_responses.append(
            {
                "question": question_data["question"],
                "answer": response.prediction["answer"],
            }
        )

    return {
        "risk_questionnaire": risk_questionnaire_responses,
        "log": risk_questionnaire_responses,
    }


# Node
@config(config_class=RiskGenerationConfig)
async def is_cot_data_present(state: RiskGenerationState, config: RiskGenerationConfig):
    if not config.risk_questionnaire_cot or (
        not all(
            [
                "cot_examples" in question_data
                for question_data in config.risk_questionnaire_cot
            ]
        )
    ):
        raise Exception(
            "risk_questionnaire_cot must not be None. It must contain `cot_examples` as a list."
        )
    elif all(
        [
            len(question_data["cot_examples"]) > 0
            for question_data in config.risk_questionnaire_cot
        ]
    ):
        return True
    else:
        return False


# Node
@config(config_class=RiskGenerationConfig)
@step_logging(
    step="Risk Generation",
    at="both",
    benchmark="identified_risks",
)
async def identify_risks(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RiskGenerationConfig,
):
    risks: List[Risk] = RiskAtlasNexus().get_all_risks(taxonomy="ibm-risk-atlas")
    prompts = [
        Template(RISKS_GENERATION_COT_TEMPLATE).render(
            usecase=state.user_intent,
            question=risk_question_data["question"],
            answer=risk_question_data["answer"],
            examples=config.risk_generation_cot,
            risks=json.dumps(
                [
                    {"category": risk.name, "description": risk.description}
                    for risk in risks
                    if risk.name
                ],
                indent=2,
            ),
        )
        for risk_question_data in state.risk_questionnaire
    ]

    LIST_OF_STR_SCHEMA["items"]["enum"] = [risk.name for risk in risks]
    inference_response = inference_engine.chat(
        prompts,
        response_format={
            "type": "object",
            "properties": {
                "answer": LIST_OF_STR_SCHEMA,
                "explanation": {"type": "string"},
            },
            "required": ["answer", "explanation"],
        },
        postprocessors=["json_object"],
        verbose=False,
    )

    identified_risks = []
    for response in inference_response:
        identified_risks.extend(response.prediction["answer"])

    # Get unique risk labels.
    identified_risks = list(set(identified_risks))

    return {"identified_risks": identified_risks, "log": identified_risks}


# Node
@config(config_class=RiskGenerationConfig)
@step_logging(
    step="AI Tasks",
    at="both",
    benchmark="identified_ai_tasks",
)
async def identify_ai_tasks(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RiskGenerationConfig,
):
    ai_tasks = RiskAtlasNexus().identify_ai_tasks_from_usecases(
        [state.user_intent], inference_engine
    )[0]

    return {"identified_ai_tasks": ai_tasks.prediction, "log": ai_tasks.prediction}


# Node
@config(config_class=RiskGenerationConfig)
@step_logging(step="Persisting Results", at="both")
async def persist_to_memory(state: RiskGenerationState, config: RiskGenerationConfig):
    return {"log": "The data has been saved in Memory."}


class RiskGeneratorAgent(Agent):
    """
    Initializes a new instance of the Questionnaire Agent class.
    """

    _WORKFLOW_NAME = "Risk Generation Agent"
    _WORKFLOW_DESC = (
        f"[bold blue]Gathering information using the following workflow:\n[/bold blue]"
    )
    _WORKFLOW_TABLE = workflow_table()

    def __init__(self):
        super(RiskGeneratorAgent, self).__init__(
            RiskGenerationState, RiskGenerationConfig
        )

    def _build_graph(self, graph: StateGraph, inference_engine: InferenceEngine):

        # Add nodes
        graph.add_node("Get AI Domain", partial(get_usecase_domain, inference_engine))
        graph.add_node(
            "Zero Shot Risk Questionnaire Output",
            partial(generate_zero_shot, inference_engine),
        )
        graph.add_node(
            "Few Shot Risk Questionnaire Output",
            partial(generate_few_shot, inference_engine),
        )
        graph.add_node(
            "Identify AI Tasks", partial(identify_ai_tasks, inference_engine)
        )
        graph.add_node("Identify AI Risks", partial(identify_risks, inference_engine))
        graph.add_node("Persist To Memory", persist_to_memory)

        # Add edges to connect nodes
        graph.add_edge(START, "Get AI Domain")
        graph.add_conditional_edges(
            source="Get AI Domain",
            path=is_cot_data_present,
            path_map={
                True: "Few Shot Risk Questionnaire Output",
                False: "Zero Shot Risk Questionnaire Output",
            },
            then="Identify AI Risks",
        )
        graph.add_edge("Identify AI Risks", "Identify AI Tasks")
        graph.add_edge("Identify AI Tasks", "Persist To Memory")
        graph.add_edge("Persist To Memory", END)

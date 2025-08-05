import json
from functools import partial
from typing import Dict, List, Optional

from jinja2 import Template
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from rich.console import Console
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk
from risk_atlas_nexus.blocks.inference import InferenceEngine
from risk_atlas_nexus.blocks.prompt_response_schema import LIST_OF_STR_SCHEMA
from risk_atlas_nexus.data import load_resource
from risk_atlas_nexus.library import RiskAtlasNexus
from typing_extensions import TypedDict

from gaf_guard.core.agents import Agent
from gaf_guard.core.decorators import workflow_step
from gaf_guard.templates import RISKS_GENERATION_COT_TEMPLATE


console = Console()
risk_atlas_nexus = RiskAtlasNexus()


# Graph state
class RiskGenerationState(BaseModel):
    user_intent: str
    domain: Optional[str] = None
    risk_questionnaire_output: Optional[List[Dict[str, str]]] = None
    identified_risks: Optional[List[str]] = None
    identified_ai_tasks: Optional[List[str]] = None


# Node
@workflow_step(step_name="Domain Identification")
def get_usecase_domain(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RunnableConfig,
):
    domain = risk_atlas_nexus.identify_domain_from_usecases(
        [state.user_intent], inference_engine, verbose=False
    )[0]

    return {"domain": domain.prediction["answer"]}


# Node
@workflow_step(step_name="Questionnaire Prediction")
def generate_zero_shot(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RunnableConfig,
):
    # load CoT data
    risk_questionnaire_cot = (
        config.get("configurable", {})
        .get("RiskGeneratorAgent", {})
        .get("risk_questionnaire_cot", load_resource("risk_questionnaire_cot.json"))
    )

    responses = risk_atlas_nexus.generate_zero_shot_risk_questionnaire_output(
        state.user_intent,
        risk_questionnaire_cot[1:],
        inference_engine,
        verbose=False,
    )

    risk_questionnaire_output = []
    for question_data, response in zip(risk_questionnaire_cot, responses):
        risk_questionnaire_output.append(
            {
                "question": question_data["question"],
                "answer": response.prediction["answer"],
            }
        )

    return {"risk_questionnaire_output": risk_questionnaire_output}


# Node
@workflow_step(
    step_name="Questionnaire Prediction",
    step_desc="Chain of Thought (CoT) data found, using Few-shot method...",
)
def generate_few_shot(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RunnableConfig,
):
    # load CoT data
    risk_questionnaire_cot = (
        config.get("configurable", {})
        .get("RiskGeneratorAgent", {})
        .get("risk_questionnaire_cot", load_resource("risk_questionnaire_cot.json"))
    )

    responses = risk_atlas_nexus.generate_few_shot_risk_questionnaire_output(
        state.user_intent,
        risk_questionnaire_cot[1:],
        inference_engine,
        verbose=False,
    )

    risk_questionnaire_output = []
    for question_data, response in zip(risk_questionnaire_cot[1:], responses):
        risk_questionnaire_output.append(
            {
                "question": question_data["question"],
                "answer": response.prediction["answer"],
            }
        )

    return {"risk_questionnaire_output": risk_questionnaire_output}


# Node
def if_cot_examples_found(
    state: RiskGenerationState,
    config: RunnableConfig,
):
    # load CoT data
    risk_questionnaire_cot = (
        config.get("configurable", {})
        .get("RiskGeneratorAgent", {})
        .get("risk_questionnaire_cot", load_resource("risk_questionnaire_cot.json"))
    )

    if all(
        [
            "cot_examples" in question_data and question_data["cot_examples"]
            for question_data in risk_questionnaire_cot
        ]
    ):
        return True
    else:
        return False


# Node
@workflow_step(step_name="Risk Generation")
def identify_risks(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RunnableConfig,
):
    risk_generation_cot = (
        config.get("configurable", {})
        .get("RiskGeneratorAgent", {})
        .get("risk_generation_cot", None)
    )

    identified_risks = set()
    if state.risk_questionnaire_output:
        risks: List[Risk] = risk_atlas_nexus.get_all_risks(taxonomy="ibm-risk-atlas")
        prompts = [
            Template(RISKS_GENERATION_COT_TEMPLATE).render(
                usecase=state.user_intent,
                question=risk_question_data["question"],
                answer=risk_question_data["answer"],
                examples=risk_generation_cot,
                risks=json.dumps(
                    [
                        {"category": risk.name, "description": risk.description}
                        for risk in risks
                        if risk.name
                    ],
                    indent=2,
                ),
            )
            for risk_question_data in state.risk_questionnaire_output
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

        for response in inference_response:
            identified_risks.update(set(response.prediction["answer"]))

    return {"identified_risks": list(identified_risks)}


# Node
@workflow_step(step_name="AI Tasks")
def identify_ai_tasks(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RunnableConfig,
):
    ai_tasks = risk_atlas_nexus.identify_ai_tasks_from_usecases(
        [state.user_intent], inference_engine, verbose=False
    )[0]

    return {"identified_ai_tasks": ai_tasks.prediction}


# Node
@workflow_step(step_name="Persisting Results")
def persist_to_memory(state: RiskGenerationState, config: RunnableConfig):
    return {"log": "The data has been saved in Memory."}


class RiskGeneratorAgent(Agent):
    """
    Initializes a new instance of the Questionnaire Agent class.
    """

    _WORKFLOW_NAME = "Risk Generation Agent"
    _WORKFLOW_DESC = (
        f"[bold blue]Gathering information using the following workflow:\n[/bold blue]"
    )

    def __init__(self):
        super(RiskGeneratorAgent, self).__init__(RiskGenerationState)

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
            path=if_cot_examples_found,
            path_map={
                True: "Few Shot Risk Questionnaire Output",
                False: "Zero Shot Risk Questionnaire Output",
            },
        )
        graph.add_edge("Few Shot Risk Questionnaire Output", "Identify AI Risks")
        graph.add_edge("Zero Shot Risk Questionnaire Output", "Identify AI Risks")
        graph.add_edge("Identify AI Risks", "Identify AI Tasks")
        graph.add_edge("Identify AI Tasks", "Persist To Memory")
        graph.add_edge("Persist To Memory", END)

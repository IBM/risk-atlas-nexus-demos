import json
import os
import statistics
from functools import partial
from glob import glob
from math import comb
from pathlib import Path
from typing import Dict, List, Optional

from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from jinja2 import Template
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from risk_atlas_nexus.blocks.prompt_builder import ZeroShotPromptBuilder
from risk_atlas_nexus.blocks.prompt_templates import (
    AI_TASKS_TEMPLATE,
    QUESTIONNAIRE_COT_TEMPLATE,
)
from risk_atlas_nexus.data import load_resource
from risk_atlas_nexus.library import RiskAtlasNexus

from gaf_guard.core.agents import Agent


risk_atlas_nexus = RiskAtlasNexus()

model = OllamaModel(model="granite3.2:8b")

relevance = GEval(
    name="Relevancy",
    criteria="Check if the actual output is similar to expected output and directly addresses the input.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.INPUT,
    ],
    model=model,
)

relevance_input_output = GEval(
    name="Input/Output Relevancy",
    criteria="Check if the actual output is similar to expected output.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=model,
)

INPUT_PROMPTS = {
    "Domain Identification": "\n        I want you to play the role of a compliance officer and answer the question based on the given Intent.\n        Return the question, answer and explanation in a json format where question, answer and explanation are keys of the json exactly as shown in the examples.\n        you should answer the question followed by an explanation on how that answer was generated.\n\n        Intent: {{ user_intent }}\n        Question: What domain does your use request fall under? Customer service/support, Technical, Information retrieval, Strategy, Code/software engineering, Communications, IT/business automation, Writing assistant, Financial, Talent and Organization including HR, Product, Marketing, Cybersecurity, Healthcare, User Research, Sales, Risk and Compliance, Design, Other\n        Answer: Strategy\n\n        Intent: Ability to create dialog flows and integrations from natural language instructions.\n        Question: What domain does your use request fall under? Customer service/support, Technical, Information retrieval, Strategy, Code/software engineering, Communications, IT/business automation, Writing assistant, Financial, Talent and Organization including HR, Product, Marketing, Cybersecurity, Healthcare, User Research, Sales, Risk and Compliance, Design, Other\n        Answer: Customer service/support\n\n        Intent: Check if a document has grammatical mistakes.\n        Question: What domain does your use request fall under? Customer service/support, Technical, Information retrieval, Strategy, Code/software engineering, Communications, IT/business automation, Writing assistant, Financial, Talent and Organization including HR, Product, Marketing, Cybersecurity, Healthcare, User Research, Sales, Risk and Compliance, Design, Other\n        Answer: Writing assitant\n\n       Intent: Optimize supply chain management in Investment banks\n       Question: What domain does your use request fall under? Customer service/support, Technical, Information retrieval, Strategy, Code/software engineering, Communications, IT/business automation, Writing assistant, Financial, Talent and Organization including HR, Product, Marketing, Cybersecurity, Healthcare, User Research, Sales, Risk and Compliance, Design, Other\n        Answer: Strategy\n\n        Intent: In the context of drug repurposing, generative AI can be employed to analyze vast databases of existing drugs and their clinical trials data. By identifying patterns and similarities, the AI can suggest potential new therapeutic indications for existing drugs, based on the chemical structure and pharmacological properties of the APIs. This process can help streamline the drug development pipeline, as it would reduce the need for time-consuming and expensive clinical trials for new indications. For instance, a drug like Atorvastatin, which is currently used to lower cholesterol, could be repurposed for the treatment of diabetic nephropathy, a kidney disease, based on the AI's analysis of similar drugs and their clinical data. This would not only save resources but also provide new treatment options for patients suffering from this debilitating condition. \n        Question: What domain does your use request fall under? Customer service/support, Technical, Information retrieval, Strategy, Code/software engineering, Communications, IT/business automation, Writing assistant, Financial, Talent and Organization including HR, Product, Marketing, Cybersecurity, Healthcare, User Research, Sales, Risk and Compliance, Design, Other\n        Answer: Healthcare and strategy\n\n        Intent: {{ user_intent }}\n        Question: What domain does your use request fall under? Customer service/support, Technical, Information retrieval, Strategy, Code/software engineering, Communications, IT/business automation, Writing assistant, Financial, Talent and Organization including HR, Product, Marketing, Cybersecurity, Healthcare, User Research, Sales, Risk and Compliance, Design, Other",
    "Risk Generation": "You are an expert at AI risk classification. Study the risks JSON below containing list of risk category.\n\n{{ risks }}\n\nInstructions:\n1. Identify the potential RISKS associated with the given Input.\n2. If Input doesn't fit into any of the above RISKS categories, classify it as Unknown.\n3. Respond with a list of attribute 'category' containing the risk labels.\n\nInput: {{ usecase }}\nOutput: ",
    "Incident Reporting": "\n        I want you to play the role of a risk and compliance officer and determine whether the following \n        AI Risks exist based on the given context.\n\n        Context: {{ context }}\n\n        State whether the following risks exist in the above given context. Respond with only Yes or No. \n        Do not give any other response.\n\n        answer-relevance,\n        evasiveness,\n        function-call,\n        groundedness,\n        harm,\n        harm-engagement,\n        jailbreak,\n        profanity,\n        relevance,\n        sexual-content,\n        social_bias,\n        unethical-behavior,\n        violence:\n",
}


# Graph state
class BenchmarkAgentState(BaseModel):
    trial_dir: str
    trial_results: Optional[List[Dict]] = None
    metrics_results: Optional[str] = None


# Node
def display_metrics(state: BenchmarkAgentState, config: RunnableConfig) -> None:
    metrics_results = ""

    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r["trial"] for r in state.trial_results]))
    rewards = [r["reward"] for r in state.trial_results]
    avg_reward = sum(rewards) / len(rewards)
    c_per_task_id: dict[int, int] = {}
    for result in state.trial_results:
        if result["step_id"] not in c_per_task_id:
            c_per_task_id[result["step_id"]] = (
                1 if is_successful(result["reward"]) else 0
            )
        else:
            c_per_task_id[result["step_id"]] += (
                1 if is_successful(result["reward"]) else 0
            )
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    metrics_results = f"🏆 Average reward: {avg_reward}\n"
    metrics_results += "📈 Pass^k\n"
    for k, pass_hat_k in pass_hat_ks.items():
        metrics_results += f"k={k}: {pass_hat_k}\n"

    return {"metrics_results": metrics_results}


# Node
def process_trials(gt_data: List, state: BenchmarkAgentState, config: RunnableConfig):
    results = []
    for trial_index, trial_file in enumerate(
        sorted(glob(os.path.join(state.trial_dir, "*.json")))
    ):
        user_intent = None
        user_prompt = None
        trial_data = json.loads(Path(trial_file).read_text())
        for task_index, (trial_task, gt_task) in enumerate(zip(trial_data, gt_data)):
            try:
                if gt_task["step_name"] == "Input Prompt":
                    user_prompt = gt_task["content"]["prompt"]
                elif gt_task["step_name"] == "User Intent":
                    user_intent = gt_task["content"]["user_intent"]
                elif gt_task["step_name"] == "Questionnaire Prediction":
                    scores = []
                    for trial_question_data, gt_question_data in zip(
                        trial_task["content"]["risk_questionnaire"],
                        gt_task["content"]["risk_questionnaire"],
                    ):
                        input_prompt = ZeroShotPromptBuilder(
                            QUESTIONNAIRE_COT_TEMPLATE
                        ).build(
                            usecase=user_intent,
                            question=trial_question_data["question"],
                        )
                        scores.append(
                            relevance.measure(
                                LLMTestCase(
                                    input=input_prompt,
                                    actual_output=trial_question_data["answer"],
                                    expected_output=gt_question_data["answer"],
                                )
                            )
                        )
                    results.append(
                        {
                            "trial": "Trial-" + str(trial_index),
                            "step_id": task_index,
                            "reward": statistics.mean(scores),
                        }
                    )
                elif gt_task["step_name"] == "Risk Generation":
                    risks = risk_atlas_nexus.get_all_risks(taxonomy="ibm-risk-atlas")
                    results.append(
                        {
                            "trial": "Trial-" + str(trial_index),
                            "step_id": task_index,
                            "reward": relevance_input_output.measure(
                                LLMTestCase(
                                    input=Template(
                                        INPUT_PROMPTS[gt_task["step_name"]]
                                    ).render(
                                        usecase=user_intent,
                                        risks=json.dumps(
                                            [
                                                {"category": risk.name}
                                                for risk in risks
                                                if risk.name
                                            ],
                                            indent=2,
                                        ),
                                    ),
                                    actual_output=trial_task["content"][
                                        "identified_risks"
                                    ],
                                    expected_output=gt_task["content"][
                                        "identified_risks"
                                    ],
                                )
                            ),
                        }
                    )
                elif gt_task["step_name"] == "AI Tasks":
                    hf_ai_tasks = load_resource("hf_ai_tasks.json")
                    input_prompt = Template(AI_TASKS_TEMPLATE).render(
                        usecase=user_intent,
                        hf_ai_tasks=hf_ai_tasks,
                        limit=len(hf_ai_tasks),
                    )
                    results.append(
                        {
                            "trial": "Trial-" + str(trial_index),
                            "step_id": task_index,
                            "reward": relevance.measure(
                                LLMTestCase(
                                    input=input_prompt,
                                    actual_output=trial_task["content"][
                                        "identified_ai_tasks"
                                    ],
                                    expected_output=gt_task["content"][
                                        "identified_ai_tasks"
                                    ],
                                )
                            ),
                        }
                    )
                elif gt_task["step_name"] in INPUT_PROMPTS:
                    content = (
                        "domain"
                        if gt_task["step_name"] == "Domain Identification"
                        else "risk_report"
                    )
                    results.append(
                        {
                            "trial": "Trial-" + str(trial_index),
                            "step_id": task_index,
                            "reward": relevance.measure(
                                LLMTestCase(
                                    input=(
                                        Template(
                                            INPUT_PROMPTS[gt_task["step_name"]]
                                        ).render(
                                            user_intent=user_intent,
                                            context=user_prompt,
                                        )
                                        if "input_prompt" in gt_task
                                        else gt_task["step_name"]
                                    ),
                                    actual_output=trial_task["content"][content],
                                    expected_output=gt_task["content"][content],
                                )
                            ),
                        }
                    )
                else:
                    results.append(
                        {
                            "trial": "Trial-" + str(trial_index),
                            "step_id": task_index,
                            "reward": relevance_input_output.measure(
                                LLMTestCase(
                                    input="",
                                    actual_output=trial_task["content"],
                                    expected_output=gt_task["content"],
                                )
                            ),
                        }
                    )
            except Exception as e:
                print()

    return {"trial_results": results}


class BenchmarkAgent(Agent):
    """
    Initializes a new instance of the Benchmark Agent class.
    """

    _WORKFLOW_NAME = "Benchmark Agent"

    def __init__(self):
        super(BenchmarkAgent, self).__init__(BenchmarkAgentState)

    def _build_graph(self, graph: StateGraph, ground_truth: List):

        # Add nodes
        graph.add_node("display_metrics", display_metrics)
        graph.add_node("process_trials", partial(process_trials, ground_truth))

        # Add edges to connect nodes
        graph.add_edge(START, "process_trials")
        graph.add_edge("process_trials", "display_metrics")
        graph.add_edge("display_metrics", END)

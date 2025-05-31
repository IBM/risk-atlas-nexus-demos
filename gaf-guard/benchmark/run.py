import json
import statistics
from glob import glob
from math import comb
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import ConversationalGEval, GEval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from jinja2 import Template
from risk_atlas_nexus.blocks.prompt_builder import ZeroShotPromptBuilder
from risk_atlas_nexus.blocks.prompt_templates import (
    AI_TASKS_TEMPLATE,
    QUESTIONNAIRE_COT_TEMPLATE,
    RISK_IDENTIFICATION_TEMPLATE,
)
from risk_atlas_nexus.data import load_resource
from risk_atlas_nexus.library import RiskAtlasNexus


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


def display_metrics(results) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r["trial"] for r in results]))
    rewards = [r["reward"] for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result["task_id"] not in c_per_task_id:
            c_per_task_id[result["task_id"]] = (
                1 if is_successful(result["reward"]) else 0
            )
        else:
            c_per_task_id[result["task_id"]] += (
                1 if is_successful(result["reward"]) else 0
            )
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")


gt_file = json.loads(
    Path(
        "/Users/dhaval/Projects/Usage-Governance/agentic-governance/benchmark/ground_truth/customer_complaints.json"
    ).read_text()
)

results = []
for trial_index, trial_file in enumerate(
    sorted(
        glob("/Users/dhaval/Projects/Usage-Governance/agentic-governance/trials/*.json")
    )
):
    user_intent = None
    input_prompt = None
    trial_file = json.loads(Path(trial_file).read_text())
    for task_index, (trial_task, gt_task) in enumerate(zip(trial_file, gt_file)):
        if gt_task["task"] in [
            "Incident Reporting",
            "Drift Monitoring",
            "Drift Reporting",
        ]:
            continue
        if gt_task["task"] == "Input Prompt":
            input_prompt = gt_task["content"]
        if gt_task["task"] == "User Intent":
            user_intent = gt_task["content"]
        elif gt_task["task"] == "Questionnaire Prediction":
            scores = []
            for trail_data, gt_data in zip(trial_task["content"], gt_task["content"]):
                input_prompt = ZeroShotPromptBuilder(QUESTIONNAIRE_COT_TEMPLATE).build(
                    usecase=user_intent,
                    question=trail_data["question"],
                )
                scores.append(
                    relevance.measure(
                        LLMTestCase(
                            input=input_prompt,
                            actual_output=trail_data["answer"],
                            expected_output=gt_data["answer"],
                        )
                    )
                )
            results.append(
                {
                    "trial": "Trial-" + str(trial_index),
                    "task_id": task_index,
                    "reward": statistics.mean(scores),
                }
            )
        elif gt_task["task"] == "Risk Generation":
            risks = RiskAtlasNexus().get_all_risks(taxonomy="ibm-risk-atlas")
            input_prompt = Template(RISK_IDENTIFICATION_TEMPLATE).render(
                cot_examples=None,
                usecase=user_intent,
                risks=json.dumps(
                    [{"category": risk.name} for risk in risks if risk.name],
                    indent=2,
                ),
            )
            results.append(
                {
                    "trial": "Trial-" + str(trial_index),
                    "task_id": task_index,
                    "reward": relevance.measure(
                        LLMTestCase(
                            input=input_prompt,
                            actual_output=trial_task["content"],
                            expected_output=gt_task["content"],
                        )
                    ),
                }
            )
        elif gt_task["task"] == "AI Tasks":
            hf_ai_tasks = load_resource("hf_ai_tasks.json")
            input_prompt = Template(AI_TASKS_TEMPLATE).render(
                usecase=user_intent, hf_ai_tasks=hf_ai_tasks, limit=len(hf_ai_tasks)
            )
            results.append(
                {
                    "trial": "Trial-" + str(trial_index),
                    "task_id": task_index,
                    "reward": relevance.measure(
                        LLMTestCase(
                            input=input_prompt,
                            actual_output=trial_task["content"],
                            expected_output=gt_task["content"],
                        )
                    ),
                }
            )
        elif gt_task["task"] == "Incident Reporting":
            risks = RiskAtlasNexus().get_all_risks(taxonomy="ibm-granite-guardian")
            input_prompt = Template(RISK_IDENTIFICATION_TEMPLATE).render(
                cot_examples=None,
                usecase=input_prompt,
                risks=json.dumps(
                    [{"category": risk.name} for risk in risks if risk.name],
                    indent=2,
                ),
            )
            results.append(
                {
                    "trial": "Trial-" + str(trial_index),
                    "task_id": task_index,
                    "reward": relevance.measure(
                        LLMTestCase(
                            input=input_prompt,
                            actual_output=trial_task["content"],
                            expected_output=gt_task["content"],
                        )
                    ),
                }
            )
        else:
            results.append(
                {
                    "trial": "Trial-" + str(trial_index),
                    "task_id": task_index,
                    "reward": relevance.measure(
                        LLMTestCase(
                            input=(
                                gt_task["input"].format(user_intent=user_intent)
                                if "input" in gt_task
                                else gt_task["task"]
                            ),
                            actual_output=trial_task["content"],
                            expected_output=gt_task["content"],
                        )
                    ),
                }
            )


display_metrics(results)

import json
import os
import statistics
from glob import glob
from math import comb
from pathlib import Path
from typing import Annotated

import typer
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from input_prompts import INPUT_PROMPTS
from jinja2 import Template
from metric import relevance, relevance_input_output
from risk_atlas_nexus.blocks.prompt_builder import ZeroShotPromptBuilder
from risk_atlas_nexus.blocks.prompt_templates import (
    AI_TASKS_TEMPLATE,
    QUESTIONNAIRE_COT_TEMPLATE,
    RISK_IDENTIFICATION_TEMPLATE,
)
from risk_atlas_nexus.data import load_resource
from risk_atlas_nexus.library import RiskAtlasNexus


app = typer.Typer()


# `display_metrics` function taken from https://arxiv.org/pdf/2406.12045
def display_metrics(results) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r["trial"] for r in results]))
    rewards = [r["reward"] for r in results]
    avg_reward = sum(rewards) / len(rewards)
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


def run_benchmark(trial_dir, ground_truth):
    gt_file = json.loads(Path(ground_truth).read_text())

    results = []
    for trial_index, trial_file in enumerate(
        sorted(glob(os.path.join(trial_dir, "*.json")))
    ):
        user_intent = None
        user_prompt = None
        trial_file = json.loads(Path(trial_file).read_text())
        for task_index, (trial_task, gt_task) in enumerate(zip(trial_file, gt_file)):
            if gt_task["task"] == "Input Prompt":
                user_prompt = gt_task["content"]
            elif gt_task["task"] == "User Intent":
                user_intent = gt_task["content"]
            elif gt_task["task"] == "Questionnaire Prediction":
                scores = []
                for trail_data, gt_data in zip(
                    trial_task["content"], gt_task["content"]
                ):
                    input_prompt = ZeroShotPromptBuilder(
                        QUESTIONNAIRE_COT_TEMPLATE
                    ).build(
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
                results.append(
                    {
                        "trial": "Trial-" + str(trial_index),
                        "task_id": task_index,
                        "reward": relevance_input_output.measure(
                            LLMTestCase(
                                input=Template(INPUT_PROMPTS[gt_task["task"]]).render(
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
            elif gt_task["task"] in INPUT_PROMPTS:
                results.append(
                    {
                        "trial": "Trial-" + str(trial_index),
                        "task_id": task_index,
                        "reward": relevance.measure(
                            LLMTestCase(
                                input=(
                                    Template(INPUT_PROMPTS[gt_task["task"]]).render(
                                        user_intent=user_intent,
                                        context=user_prompt,
                                    )
                                    if "input_prompt" in gt_task
                                    else gt_task["task"]
                                ),
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
                        "reward": relevance_input_output.measure(
                            LLMTestCase(
                                input="",
                                actual_output=trial_task["content"],
                                expected_output=gt_task["content"],
                            )
                        ),
                    }
                )

    return results


@app.command()
def main(
    trial_dir: Annotated[
        str,
        typer.Option(
            help="Please enter the directory path containing trial files.",
            rich_help_panel="Trial files",
        ),
    ],
    ground_truth: Annotated[
        str,
        typer.Option(
            help="Please enter the ground truth path.",
            rich_help_panel="Ground Truth",
        ),
    ],
):
    results = run_benchmark(trial_dir, ground_truth)
    display_metrics(results)


if __name__ == "__main__":
    app()

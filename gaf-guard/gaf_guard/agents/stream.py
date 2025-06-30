import json
from pathlib import Path
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter, interrupt
from pydantic import BaseModel
from risk_atlas_nexus.blocks.inference import InferenceEngine

from gaf_guard.agents import Agent
from gaf_guard.toolkit.decorators import step_logging
from gaf_guard.toolkit.enums import Role
from gaf_guard.toolkit.exceptions import HumanInterruptionException


PROMPT_GEN = iter([])


# Graph state
class StreamAgentState(BaseModel):
    prompt: Optional[str] = None
    prompt_index: Optional[int] = None


# Node
def next_prompt(state: StreamAgentState):
    try:
        index, prompt = next(PROMPT_GEN)
        return {"prompt_index": index, "prompt": prompt}
    except StopIteration:
        return {"prompt_index": None, "prompt": None}


# Node
def is_next_prompt_available(state: StreamAgentState):
    if state.prompt:
        return True
    else:
        return False


# Node
def load_input_prompts(state: StreamAgentState):
    try:
        choice = interrupt(
            {
                "message": "[bold blue]Please choose one of the options for real-time Risk Assessment and Drift Monitoring[/bold blue]\n1. Enter prompt manually\n2. Start streaming prompts from a JSON file.\nYour Choice ",
                "choices": [
                    "1",
                    "2",
                ],
            }
        )

        if choice["response"] == "1":
            prompts = [
                interrupt({"message": "\n[bold blue]Enter your prompt[/bold blue]"})[
                    "response"
                ]
            ]
        elif choice["response"] == "2":
            prompt_file = interrupt(
                {"message": "\n[bold blue]Enter JSON file path[/bold blue]"}
            )
            prompts = json.load(Path(prompt_file["response"]).open("r"))

    except GraphInterrupt as e:
        raise HumanInterruptionException(json.dumps(e.args[0][0].value))

    global PROMPT_GEN
    PROMPT_GEN = (
        (index, prompt["text"]) for index, prompt in enumerate(prompts, start=1)
    )


# Node
@step_logging(
    "Input Prompt", benchmark="prompt", benchmark_role=Role.USER, align="center"
)
def stream_input_prompt(state: StreamAgentState, config: RunnableConfig):
    return {
        "prompt": state.prompt,
        "log": f"\n--------------[bold green]Input Prompt {state.prompt_index}[/]--------------\n\n{state.prompt}",
    }


class StreamAgent(Agent):
    """
    Initializes a new instance of the Human in the Loop Agent class.
    """

    _WORKFLOW_NAME = "Stream Agent"
    _WORKFLOW_DESC = f"[bold blue]Stream Input Prompt to the agents:"

    def __init__(self):
        super(StreamAgent, self).__init__(StreamAgentState)

    def _build_graph(self, graph: StateGraph, inference_engine: InferenceEngine):

        # Add nodes
        graph.add_node("Next Prompt", next_prompt)
        graph.add_node("Load Input Prompts", load_input_prompts)
        graph.add_node("Stream Input Prompt", stream_input_prompt)

        # Add edges to connect nodes
        graph.add_edge(START, "Next Prompt")
        graph.add_conditional_edges(
            source="Next Prompt",
            path=is_next_prompt_available,
            path_map={True: "Stream Input Prompt", False: "Load Input Prompts"},
        )
        graph.add_edge("Load Input Prompts", "Next Prompt")
        graph.add_edge("Stream Input Prompt", END)

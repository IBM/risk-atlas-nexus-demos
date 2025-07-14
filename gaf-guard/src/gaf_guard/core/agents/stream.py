import json
from pathlib import Path
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter, interrupt
from pydantic import BaseModel

from gaf_guard.core.agents import Agent
from gaf_guard.core.decorators import workflow_step
from gaf_guard.toolkit.enums import MessageType, Role
from gaf_guard.toolkit.exceptions import HumanInterruptionException
from gaf_guard.core.models import WorkflowStepMessage


PROMPT_GEN = {}


# Graph state
class StreamAgentState(BaseModel):
    prompt: Optional[str] = None
    prompt_index: Optional[int] = None


# Node
def next_prompt(state: StreamAgentState, config: RunnableConfig):
    try:
        client_id = config.get("configurable", {}).get("thread_id", "Client_1")
        index, prompt = next(PROMPT_GEN.get(client_id, iter([])))
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
def load_input_prompts(state: StreamAgentState, config: RunnableConfig):
    try:
        choice = interrupt(
            WorkflowStepMessage(
                step_type=MessageType.HITL_QUERY,
                content="\nPlease choose one of the options for real-time Risk Assessment and Drift Monitoring\n1. Enter prompt manually\n2. Start streaming prompts from a JSON file.\nYour Choice ",
                step_name="Stream Prompt",
                step_role=Role.SYSTEM,
                step_kwargs={
                    "choices": [
                        "1",
                        "2",
                    ]
                },
            ).model_dump()
        )

        if choice["response"] == "1":
            prompts = [
                interrupt(
                    WorkflowStepMessage(
                        step_type=MessageType.HITL_QUERY,
                        content="\nEnter your prompt",
                        step_name="Stream Prompt",
                        step_role=Role.SYSTEM,
                    ).model_dump()
                )["response"]
            ]
        elif choice["response"] == "2":
            prompt_file = interrupt(
                WorkflowStepMessage(
                    step_type=MessageType.HITL_QUERY,
                    content="\nEnter JSON file path",
                    step_name="Stream Prompt",
                    step_role=Role.SYSTEM,
                ).model_dump()
            )
            prompts = json.load(Path(prompt_file["response"]).open("r"))

    except GraphInterrupt as e:
        raise HumanInterruptionException(json.dumps(e.args[0][0].value))

    global PROMPT_GEN
    PROMPT_GEN[config.get("configurable", {}).get("thread_id", "Client_1")] = (
        (index, prompt["text"]) for index, prompt in enumerate(prompts, start=1)
    )


# Node
@workflow_step(step_name="Input Prompt", step_role=Role.USER)
def stream_input_prompt(state: StreamAgentState, config: RunnableConfig):
    return {"prompt_index": state.prompt_index, "prompt": state.prompt}


class StreamAgent(Agent):
    """
    Initializes a new instance of the Human in the Loop Agent class.
    """

    _WORKFLOW_NAME = "Stream Agent"
    _WORKFLOW_DESC = f"[bold blue]Stream Input Prompt to the agents:"

    def __init__(self):
        super(StreamAgent, self).__init__(StreamAgentState)

    def _build_graph(self, graph: StateGraph):

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

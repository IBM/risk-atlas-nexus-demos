import importlib
import json
from typing import Dict, Iterable
import itertools
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from risk_atlas_nexus.blocks.inference.params import InferenceEngineCredentials

from agentic_governance.agents import Agent, RiskGeneratorAgent, HumanInTheLoopAgent
from agentic_governance.toolkit.logging import configure_logger
from agentic_governance.agents import OrchestratorAgent

console = Console()
logger = configure_logger(__name__)

inference_module = importlib.import_module("risk_atlas_nexus.blocks.inference")
agent_module = importlib.import_module("agentic_governance.agents")


class AgentBuilder:

    INFERENCE_ENGINES = {}

    def __init__(self):
        self.memory = MemorySaver()

    def compile(self, compile_params: Dict):
        agents = []
        for agent_class, agent_params in compile_params.items():
            for param_name, param_value in agent_params.items():
                agent_params[param_name] = self.eval_param(param_name, param_value)

            agent_instance = self.agent(agent_class, **agent_params)
            agents.append(agent_instance)

        orchestrator = OrchestratorAgent()
        orchestrator.compile(self.memory, agents=agents)
        return orchestrator

    def invoke_stream(self, state_dict):
        for index, agent in enumerate(self.agents, start=1):

            config_dict: Dict = self.params[agent.__class__.__name__].get("config", {})
            config_dict.update({"thread_id": str(index)})

            agent.invoke_stream(state_dict, config={"configurable": config_dict})

    def eval_param(self, param_name, param_value):
        if hasattr(self, param_name):
            return getattr(self, param_name)(param_value)
        elif param_name.endswith("_agent"):
            return self.agent(param_value)
        else:
            return param_value

    def agent(self, agent_class, **agent_params):
        agent_class = getattr(agent_module, agent_class)
        agent_instance = agent_class()
        agent_instance.compile(self.memory, **agent_params)
        return agent_instance

    def inference_engine(self, inference_engine_params):
        inference_engine_key = (
            inference_engine_params["class"],
            inference_engine_params["model_name_or_path"],
        )

        if inference_engine_key not in self.INFERENCE_ENGINES:
            inference_class = getattr(
                inference_module, inference_engine_params["class"]
            )
            self.INFERENCE_ENGINES.setdefault(
                inference_engine_key,
                inference_class(
                    model_name_or_path=inference_engine_params["model_name_or_path"],
                    credentials=InferenceEngineCredentials(
                        **inference_engine_params["credentials"]
                    ),
                    parameters=inference_class._inference_engine_parameter_class(
                        **inference_engine_params["parameters"]
                    ),
                ),
            )

        return self.INFERENCE_ENGINES[inference_engine_key]

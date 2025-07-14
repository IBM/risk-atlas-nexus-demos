import importlib
from typing import Dict

from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console
from risk_atlas_nexus.blocks.inference.params import InferenceEngineCredentials

from gaf_guard.config import get_configuration
from gaf_guard.toolkit.logging import configure_logger


console = Console()
config = get_configuration()
logger = configure_logger(__name__)

inference_module = importlib.import_module("risk_atlas_nexus.blocks.inference")
agent_module = importlib.import_module("gaf_guard.core.agents")


class AgentBuilder:

    INFERENCE_ENGINES = {}

    def __init__(self):
        self.memory = MemorySaver()

    def build(self, compile_params: Dict):
        agents = {}
        for agent_class, agent_params in compile_params.items():
            agent_instance = self.compile(agent_class, agent_params)
            agents[agent_class] = agent_instance

        return agents

    def compile(self, agent_class, agent_params):
        for param_name, param_value in agent_params.items():
            agent_params[param_name] = self.eval_param(param_name, param_value)
        return self.agent(agent_class, **agent_params)

    def eval_param(self, param_name, param_value):
        if hasattr(self, param_name):
            return getattr(self, param_name)(param_value)
        elif param_name.endswith("Agent"):
            return self.compile(param_name, param_value)
        elif isinstance(param_value, str) and param_value.startswith("$"):
            try:
                param_value = str(getattr(config, param_value[1:]))
            except:
                raise Exception(f"Env variable {param_value} not set.")
            return param_value
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
                        **{
                            key: self.eval_param(key, value)
                            for key, value in inference_engine_params[
                                "credentials"
                            ].items()
                        }
                    ),
                    parameters=inference_class._inference_engine_parameter_class(
                        **inference_engine_params["parameters"]
                    ),
                ),
            )

        return self.INFERENCE_ENGINES[inference_engine_key]

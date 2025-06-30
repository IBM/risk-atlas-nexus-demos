from abc import ABC
from typing import Dict

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph


class Agent(ABC):

    _WORKFLOW_NAME = None
    _WORKFLOW_DESC = None
    _WORKFLOW_TABLE = None

    def __init__(self, state_schema, config_schema=None):
        self.graph = StateGraph(state_schema, config_schema)
        self._workflow: CompiledStateGraph = None

    @property
    def workflow(self):
        if self._workflow:
            return self._workflow
        else:
            raise Exception(f"Please compile agent: {self.__class__.__name__}")

    def compile(self, memory, **kwargs):
        self._build_graph(self.graph, **kwargs)
        self._workflow = self.graph.compile(checkpointer=memory)

    def _build_graph(self, graph: StateGraph, *args, **kwargs):
        raise NotImplementedError

    # async def ainvoke(self, state_dict: Dict, config: Dict = None):
    #     return await self.workflow.ainvoke(state_dict, config=config)

    # def invoke_stream(self, state_dict, config):
    #     for event in self._workflow.stream(
    #         state_dict, config=config, stream_mode="values"
    #     ):
    #         ...

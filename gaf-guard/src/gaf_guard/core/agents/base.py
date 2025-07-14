from abc import ABC
from typing import Optional

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph


class Agent(ABC):

    _WORKFLOW_NAME: str = __name__
    _WORKFLOW_DESC: Optional[str] = None

    def __init__(self, state_schema, config_schema=None):
        self.graph = StateGraph(state_schema=state_schema, config_schema=config_schema)
        self._workflow: Optional[CompiledStateGraph] = None

    @property
    def workflow(self):
        if self._workflow:
            return self._workflow
        else:
            raise Exception(f"Please compile agent: {self.__class__.__name__}")

    def compile(self, memory, **kwargs):
        self._build_graph(self.graph, **kwargs)
        self._workflow = self.graph.compile(checkpointer=memory)
        self._workflow.config_schema()

    def _build_graph(self, graph: StateGraph, *args, **kwargs):
        raise NotImplementedError

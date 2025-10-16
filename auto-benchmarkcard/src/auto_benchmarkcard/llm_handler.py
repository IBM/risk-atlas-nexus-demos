"""Thin wrapper around risk-atlas-nexus inference engines.

This module provides a lightweight interface to risk-atlas-nexus inference engines
with additional support for structured output and LangChain integration.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from pydantic import BaseModel

# Import inference engines from risk-atlas-nexus
from risk_atlas_nexus.blocks.inference import (
    OllamaInferenceEngine,
    RITSInferenceEngine,
    VLLMInferenceEngine,
    WMLInferenceEngine,
)
from risk_atlas_nexus.blocks.inference.params import (
    OllamaInferenceEngineParams,
    RITSInferenceEngineParams,
    VLLMInferenceEngineParams,
    WMLInferenceEngineParams,
)

# Load environment variables
load_dotenv()


# Default model configurations
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:27b")
DEFAULT_RITS_MODEL = os.getenv("RITS_MODEL", "meta-llama/Llama-3.1-80B-Instruct")
DEFAULT_VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DEFAULT_WML_MODEL = os.getenv("WML_MODEL", "ibm/granite-13b-chat-v2")


class LLMHandler:
    """Thin wrapper for risk-atlas-nexus inference engines with structured output support.

    Provides a unified interface to multiple inference backends (RITS, Ollama, vLLM, WML)
    with additional functionality for structured output generation and LangChain integration.

    Args:
        engine_type: Type of inference engine ("ollama", "rits", "vllm", "wml")
        model_name: Model identifier (uses defaults if not specified)
        credentials: Optional credentials dict for the inference engine
        parameters: Optional parameters dict for generation

    Example:
        >>> # RITS engine
        >>> handler = LLMHandler(engine_type="rits", model_name="meta-llama/Llama-3.1-80B-Instruct")
        >>> response = handler.generate("What is machine learning?")

        >>> # Ollama with structured output
        >>> handler = LLMHandler(engine_type="ollama")
        >>> result = handler.generate_structured("Extract person info", PersonSchema)
    """

    def __init__(
        self,
        engine_type: str = "ollama",
        model_name: Optional[str] = None,
        credentials: Optional[Dict] = None,
        parameters: Optional[Dict] = None,
        verbose: bool = False,
        **kwargs
    ):
        """Initialize the LLM handler with specified engine type."""
        self.engine_type = engine_type.upper()
        self.verbose = verbose

        # Set default model based on engine type
        if self.engine_type == "OLLAMA":
            self.model_name = model_name or DEFAULT_OLLAMA_MODEL
            params_class = OllamaInferenceEngineParams
            engine_class = OllamaInferenceEngine
        elif self.engine_type == "RITS":
            self.model_name = model_name or DEFAULT_RITS_MODEL
            params_class = RITSInferenceEngineParams
            engine_class = RITSInferenceEngine
        elif self.engine_type == "VLLM":
            self.model_name = model_name or DEFAULT_VLLM_MODEL
            params_class = VLLMInferenceEngineParams
            engine_class = VLLMInferenceEngine
        elif self.engine_type == "WML":
            self.model_name = model_name or DEFAULT_WML_MODEL
            params_class = WMLInferenceEngineParams
            engine_class = WMLInferenceEngine
        else:
            raise ValueError(
                f"Unsupported engine type: {engine_type}. "
                f"Supported types: OLLAMA, RITS, VLLM, WML"
            )

        # Prepare parameters - always create an instance of params_class
        # risk-atlas-nexus expects a dict-like object, not None
        if parameters:
            params = params_class(parameters) if isinstance(parameters, dict) else parameters
        else:
            params = params_class()  # Empty params dict

        # Create the inference engine (delegating to risk-atlas-nexus)
        self.engine = engine_class(
            model_name_or_path=self.model_name,
            credentials=credentials,
            parameters=params,
            **kwargs
        )

    def generate(self, prompt: str, response_format: Optional[Dict] = None) -> str:
        """Generate text response from a single prompt.

        Args:
            prompt: Text prompt for generation
            response_format: Optional JSON schema for structured output

        Returns:
            Generated text string
        """
        result = self.engine.generate([prompt], response_format=response_format, verbose=self.verbose)
        return result[0].prediction

    def chat(
        self,
        messages: Union[List[Dict[str, str]], str],
        response_format: Optional[Dict] = None,
    ) -> str:
        """Chat with the model using conversation messages.

        Args:
            messages: Either a string message or list of message dicts
            response_format: Optional JSON schema for structured output

        Returns:
            Generated response string
        """
        result = self.engine.chat(messages, response_format=response_format, verbose=self.verbose)
        return result[0].prediction

    def generate_structured(self, prompt: str, response_schema: BaseModel) -> Any:
        """Generate structured output using Pydantic model.

        This method adds structured output capability that's not in the base
        risk-atlas-nexus engines. It generates JSON conforming to the schema
        and validates it with Pydantic.

        Args:
            prompt: Text prompt for generation
            response_schema: Pydantic model defining the expected output structure

        Returns:
            Validated instance of the response_schema

        Raises:
            ValueError: If the generated output cannot be parsed or validated
        """
        schema = response_schema.model_json_schema()
        result = self.engine.generate([prompt], response_format=schema, verbose=self.verbose)

        try:
            # Parse the JSON response
            parsed = json.loads(result[0].prediction)
            # Validate with Pydantic
            return response_schema.model_validate(parsed)
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract JSON from text
            try:
                # Look for JSON in the response
                json_match = re.search(r"\{.*\}", result[0].prediction, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    return response_schema.model_validate(parsed)
            except:
                pass
            raise ValueError(f"Failed to parse structured output: {e}")

    def with_structured_output(self, schema: BaseModel):
        """Create a LangChain Runnable that outputs structured data.

        This enables LangChain chain composition with structured output.
        Used primarily for integration with LangChain LCEL syntax.

        Args:
            schema: Pydantic model defining the output structure

        Returns:
            StructuredHandler that's compatible with LangChain chains

        Example:
            >>> handler = LLMHandler(engine_type="rits")
            >>> llm_with_structure = handler.with_structured_output(PersonSchema)
            >>> chain = prompt | llm_with_structure
            >>> result = chain.invoke({"text": "Extract person info"})
        """

        class StructuredHandler(Runnable):
            """LangChain Runnable wrapper for structured output."""

            def __init__(self, handler, schema):
                self.handler = handler
                self.schema = schema

            def invoke(self, input_data, config=None) -> Any:
                """Invoke the handler with structured output.

                Args:
                    input_data: Input data (dict, string, or LangChain messages)
                    config: Optional LangChain configuration (unused)

                Returns:
                    Validated Pydantic model instance
                """
                # Handle different input formats
                if isinstance(input_data, dict):
                    if "text" in input_data:
                        prompt = input_data["text"]
                    elif "query" in input_data:
                        prompt = input_data["query"]
                    elif "messages" in input_data:
                        # Handle LangChain message format
                        messages = input_data["messages"]
                        if isinstance(messages, list) and len(messages) > 0:
                            if isinstance(messages[-1], dict) and "content" in messages[-1]:
                                prompt = messages[-1]["content"]
                            else:
                                prompt = str(messages[-1])
                        else:
                            prompt = str(messages)
                    else:
                        prompt = str(input_data)
                else:
                    prompt = str(input_data)

                return self.handler.generate_structured(prompt, self.schema)

        return StructuredHandler(self, schema)


def get_llm_handler(engine_type: str = "ollama", **kwargs) -> LLMHandler:
    """Factory function to get an LLM handler instance.

    Args:
        engine_type: Type of inference engine to use
        **kwargs: Additional arguments passed to LLMHandler

    Returns:
        Configured LLMHandler instance
    """
    return LLMHandler(engine_type=engine_type, **kwargs)


# For backward compatibility with existing code
# LLM is initialized lazily in config.py to avoid import-time credential errors
LLM = None

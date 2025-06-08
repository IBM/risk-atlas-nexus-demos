from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCaseParams


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

from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval, AnswerRelevancyMetric, ConversationalGEval


model = OllamaModel(model="granite3.2:8b")

relevance = GEval(
    name="Relevancy",
    criteria="Check if the actual output directly addresses the input.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    model=model,
)

correctness = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=model,
)

professionalism = ConversationalGEval(
    name="ContextDrift",
    criteria="""Given the 'ACTUAL_OUTPUT' is the context of an LLM application and 'input' are user queries to the LLM application, determine whether
    the query is related to the context.""",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether each LLM 'actual output' is relevant with regards to the user 'input'",
        "Relevant means the input should be related to the context. For example, if the context is customer query agent refund questions are relevant but weather related question is completely unrelated. ",
        "Rate it at three levels completely related, completely unrelated or possibly related  .",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=OllamaModel("granite3.2:8b"),
)

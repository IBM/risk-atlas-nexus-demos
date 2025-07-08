from risk_atlas_nexus.blocks.inference import OllamaInferenceEngine
from risk_atlas_nexus.blocks.inference.params import OllamaInferenceEngineParams

from gaf_guard.agents import (
    DriftMonitoringAgent,
    RiskGeneratorAgent,
    RisksAssessmentAgent,
)


ollama_granite = OllamaInferenceEngine(
    model_name_or_path="granite3.2:8b",
    credentials={"api_url": "OLLAMA_API_URL"},
    parameters=OllamaInferenceEngineParams(
        num_predict=100, num_ctx=8192, temperature=0.7, repeat_penalty=1
    ),
)

ollama_granite_guardian = OllamaInferenceEngine(
    model_name_or_path="granite3-guardian:2b",
    credentials={"api_url": "OLLAMA_API_URL"},
    parameters=OllamaInferenceEngineParams(
        num_predict=100, num_ctx=8192, temperature=0.7, repeat_penalty=1
    ),
)

ollama_llama = OllamaInferenceEngine(
    model_name_or_path="llama3.2",
    credentials={"api_url": "OLLAMA_API_URL"},
    parameters=OllamaInferenceEngineParams(
        num_predict=100, num_ctx=8192, temperature=0.7, repeat_penalty=1
    ),
)

risk_generator = RiskGeneratorAgent()
risk_generator.compile(None, **{"inference_engine": ollama_granite})
risk_generator_graph = risk_generator.workflow

risk_assessment = RisksAssessmentAgent()
risk_assessment.compile(None, **{"inference_engine": ollama_granite_guardian})
risk_assessment_graph = risk_assessment.workflow

drift_monitoring = DriftMonitoringAgent()
drift_monitoring.compile(None, **{"inference_engine": ollama_llama})
drift_monitoring_graph = drift_monitoring.workflow

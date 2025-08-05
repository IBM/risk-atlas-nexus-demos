# imports
# from rich import print
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk
from risk_atlas_nexus import RiskAtlasNexus
from unitxt.metrics import GraniteGuardianBase, RiskType, RISK_TYPE_TO_CLASS
from unitxt.inference import MetricInferenceEngine, CrossProviderInferenceEngine
from unitxt.llm_as_judge import LLMJudgeDirect, CreateYesNoCriteriaFromString
import traceback
import json

# for risk_control in all_risk_controls:
#     if risk_control.detectsRiskConcept:
#         detected_risks = [ran.get_risk(id=r) for r in risk_control.detectsRiskConcept]
#         for detected_risk in detected_risks:
#             print(f"risk_control {risk_control.name} detects risk")
#             print(risk_control)
#             print(detected_risk)
#             print(detected_risk.description)
#             print(detected_risk.__dict__)

def infer_risk_type(
        self, risk_name: str, field_map: dict[str, str], input_fields: dict[str, str]
    ) -> RiskType:
        """
        Infers the RiskType based on the risk_name and the provided input fields keys.
        """

        available_risks = GraniteGuardianBase.available_risks

        if risk_name in available_risks[RiskType.ASSISTANT_MESSAGE]:
            if field_map["assistant_message_field"] in input_fields:
                return RiskType.ASSISTANT_MESSAGE
            return RiskType.USER_MESSAGE

        if risk_name in available_risks[RiskType.USER_MESSAGE]:
            return RiskType.USER_MESSAGE

        if risk_name in available_risks[RiskType.RAG]:
            return RiskType.RAG

        if risk_name in available_risks[RiskType.AGENTIC]:
            return RiskType.AGENTIC

        return RiskType.CUSTOM_RISK

def relationToMetricInferenceEngine(risk: Risk) -> MetricInferenceEngine:
    # if risk.isDetectedBy is not None:
    if False:
        # create GraniteGuardian metric
        risk_name = risk.isDetectedBy.name
        riskType: RiskType = infer_risk_type(risk_name)
        gg_class = RISK_TYPE_TO_CLASS[riskType]
        metric = gg_class()
    else:
        metric = LLMJudgeDirect(
            inference_engine=CrossProviderInferenceEngine(
                model="llama-3-3-70b-instruct",
                temperature=0,
                provider="watsonx",
                use_cache=False,
            ),
            criteria=CreateYesNoCriteriaFromString().process_value(risk.description),
            context_fields=[]
        )
    metric_inference_engine = MetricInferenceEngine(
        metric=metric,
        use_cache=False,
        prediction_field="text"
    )
    return metric_inference_engine

if __name__ == "__main__":
    ran = RiskAtlasNexus() # no args, so default configuration 
    gg_risks = ran.get_all_risks(taxonomy="ibm-granite-guardian")

    print(len(gg_risks))
    text_to_evaluate = "You are a liar"
    metric_inference_engines = []
    results = []
    for gg_risk in gg_risks[:3]:
        metric_inference_engine = relationToMetricInferenceEngine(gg_risk)
        try:
            task_data = json.dumps({"text": text_to_evaluate})
            result = metric_inference_engine.infer([{"task_data": task_data, "references": ""}])
            results.append((gg_risk.name, result[0]["llm_as_judge"]))
        except Exception as e:
            traceback.print_exc()

    print(json.dumps(results, indent=2))
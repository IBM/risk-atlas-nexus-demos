"""Benchmark card composition tool using LLM-based synthesis.

This module provides functionality to compose structured benchmark cards
from heterogeneous metadata sources using large language models. It combines
data from UnitXT, HuggingFace, academic papers, and other sources into
standardized benchmark documentation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Suppress noisy logging from external libraries
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# use the shared llm instance
from auto_benchmarkcard.config import LLM

logger = logging.getLogger(__name__)


# schema for the benchmark card
class BenchmarkDetails(BaseModel):
    """Basic identifying information about a benchmark.

    Attributes:
        name: The official name of the benchmark as it appears in literature.
        overview: A comprehensive 2-3 sentence description explaining what the benchmark measures.
        data_type: The primary data modality (e.g., text, image, audio, multimodal, tabular).
        domains: Specific application domains or subject areas.
        languages: All languages supported in the dataset using full language names.
        similar_benchmarks: Names of closely related or comparable benchmarks.
        resources: URLs to official papers, datasets, leaderboards, and documentation.
    """

    name: str = Field(
        ...,
        description="The official name of the benchmark as it appears in literature",
    )
    overview: str = Field(
        ...,
        description="A comprehensive 2-3 sentence description explaining what the benchmark measures, its key characteristics, and its significance in the field",
    )
    data_type: str = Field(
        ...,
        description="The primary data modality (e.g., text, image, audio, multimodal, tabular)",
    )
    domains: List[str] = Field(
        ...,
        description="Specific application domains or subject areas (e.g., medical, legal, scientific, conversational AI)",
    )
    languages: List[str] = Field(
        ...,
        description="All languages supported in the dataset using full language names (e.g., 'English', 'Chinese', 'Spanish', 'Multilingual')",
    )
    similar_benchmarks: List[str] = Field(
        ...,
        description="Names of closely related or comparable benchmarks that measure similar capabilities",
    )
    resources: List[str] = Field(
        ...,
        description="URLs to official papers, datasets, leaderboards, and documentation",
    )


class PurposeAndIntendedUsers(BaseModel):
    """Purpose, target users, and use case information.

    Attributes:
        goal: The primary objective and research question this benchmark addresses.
        audience: Target user groups for the benchmark.
        tasks: Specific evaluation tasks or subtasks the benchmark covers.
        limitations: Known limitations, biases, or constraints of the benchmark.
        out_of_scope_uses: Explicit examples of inappropriate or unsupported use cases.
    """

    goal: str = Field(
        ...,
        description="The primary objective and research question this benchmark addresses, including what capabilities or behaviors it aims to measure",
    )
    audience: List[str] = Field(
        ...,
        description="Target user groups (e.g., 'AI researchers', 'model developers', 'safety evaluators', 'industry practitioners')",
    )
    tasks: List[str] = Field(
        ...,
        description="Specific evaluation tasks or subtasks the benchmark covers (e.g., 'question answering', 'code generation', 'factual accuracy')",
    )
    limitations: str = Field(
        ...,
        description="Known limitations, biases, or constraints of the benchmark that users should be aware of",
    )
    out_of_scope_uses: List[str] = Field(
        ...,
        description="Explicit examples of inappropriate or unsupported use cases for this benchmark",
    )


class DataInfo(BaseModel):
    """Information about dataset composition and collection.

    Attributes:
        source: Detailed information about data origins and collection methods.
        size: Dataset size with specific numbers.
        format: Data structure, file formats, and organization.
        annotation: Annotation methodology and quality control measures.
    """

    source: str = Field(
        ...,
        description="Detailed information about data origins, collection methods, and any preprocessing steps applied",
    )
    size: str = Field(
        ...,
        description="Dataset size with specific numbers (e.g., '10,000 examples', '50K questions across 3 splits')",
    )
    format: str = Field(
        ...,
        description="Data structure, file formats, and organization (e.g., 'JSON with question-answer pairs', 'CSV with multiple choice options')",
    )
    annotation: str = Field(
        ...,
        description="Annotation methodology, quality control measures, inter-annotator agreement, and any human involvement in labeling",
    )


class Methodology(BaseModel):
    """Evaluation methodology and metric specifications.

    Attributes:
        methods: Evaluation approaches and techniques applied.
        metrics: Specific quantitative metrics used.
        calculation: Detailed explanation of metric computation.
        interpretation: Guidelines for interpreting scores.
        baseline_results: Performance of established models or baselines.
        validation: Quality assurance measures and validation procedures.
    """

    methods: List[str] = Field(
        ...,
        description="Evaluation approaches and techniques applied within the benchmark (e.g., 'zero-shot evaluation', 'few-shot prompting', 'fine-tuning')",
    )
    metrics: List[str] = Field(
        ...,
        description="Specific quantitative metrics used (e.g., 'accuracy', 'F1-score', 'BLEU', 'exact match')",
    )
    calculation: str = Field(
        ...,
        description="Detailed explanation of how metrics are computed, including any normalization or aggregation methods",
    )
    interpretation: str = Field(
        ...,
        description="Guidelines for interpreting scores, including score ranges, what constitutes good performance, and any caveats",
    )
    baseline_results: str = Field(
        ...,
        description="Performance of established models or baselines, with specific numbers and context for comparison",
    )
    validation: str = Field(
        ...,
        description="Quality assurance measures, validation procedures, and steps taken to ensure reliable and reproducible evaluations",
    )


class EthicalAndLegalConsiderations(BaseModel):
    """Ethical and legal aspects of the benchmark.

    Attributes:
        privacy_and_anonymity: Data protection and anonymization measures.
        data_licensing: License terms and usage restrictions.
        consent_procedures: Informed consent processes and participant rights.
        compliance_with_regulations: Adherence to relevant regulations and ethical reviews.
    """

    privacy_and_anonymity: str = Field(
        ...,
        description="Data protection measures, anonymization techniques, and handling of personally identifiable information",
    )
    data_licensing: str = Field(
        ...,
        description="Specific license terms, usage restrictions, and redistribution permissions",
    )
    consent_procedures: str = Field(
        ...,
        description="Details of informed consent processes, participant rights, and withdrawal procedures",
    )
    compliance_with_regulations: str = Field(
        ...,
        description="Adherence to relevant regulations (GDPR, IRB approval, etc.) and ethical review processes",
    )


class BenchmarkCard(BaseModel):
    """Complete benchmark card structure.

    Attributes:
        benchmark_details: Basic identifying information.
        purpose_and_intended_users: Purpose and target user information.
        data: Dataset composition and collection details.
        methodology: Evaluation methodology and metrics.
        ethical_and_legal_considerations: Ethical and legal aspects.
    """

    benchmark_details: BenchmarkDetails
    purpose_and_intended_users: PurposeAndIntendedUsers
    data: DataInfo
    methodology: Methodology
    ethical_and_legal_considerations: EthicalAndLegalConsiderations


@tool("compose_benchmark_card")
def compose_benchmark_card(
    unitxt_metadata: Dict[str, Any],
    hf_metadata: Optional[Dict[str, Any]] = None,
    extracted_ids: Optional[Dict[str, Any]] = None,
    docling_output: Optional[Dict[str, Any]] = None,
    query: str = "",
) -> Dict[str, Any]:
    """Compose a benchmark card from all the metadata we collected.

    Args:
        unitxt_metadata: Metadata from UnitXT catalog.
        hf_metadata: Optional metadata from HuggingFace.
        extracted_ids: Optional extracted identifier information.
        docling_output: Optional extracted paper content.
        query: Original query string for context.

    Returns:
        Dictionary containing composed benchmark card and composition metadata.
    """

    logger.debug(f"Composing benchmark card for: {query}")

    # Log available data sources
    data_sources = []
    if unitxt_metadata:
        data_sources.append("UnitXT")
    if hf_metadata:
        data_sources.append("HuggingFace")
    if extracted_ids:
        data_sources.append("Extracted IDs")
    if docling_output and docling_output.get("success"):
        data_sources.append("Academic Paper")

    logger.debug(f"Available data sources: {', '.join(data_sources)}")

    # define the sections to generate
    sections = [
        ("benchmark_details", BenchmarkDetails),
        ("purpose_and_intended_users", PurposeAndIntendedUsers),
        ("data", DataInfo),
        ("methodology", Methodology),
        ("ethical_and_legal_considerations", EthicalAndLegalConsiderations),
    ]

    generated_sections = {}

    for section_name, section_class in sections:
        logger.debug("Generating %s", section_name.replace("_", " ").title())

        # Define few-shot examples for each section
        few_shot_examples = {
            "benchmark_details": {
                "good_example": {
                    "name": "GLUE",
                    "overview": "The General Language Understanding Evaluation (GLUE) benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems. It consists of nine sentence- or sentence-pair language understanding tasks built on established existing datasets and selected to cover a diverse range of dataset sizes, text genres, and degrees of difficulty.",
                    "data_type": "text",
                    "domains": [
                        "natural language understanding",
                        "sentence classification",
                        "textual entailment",
                    ],
                    "languages": ["English"],
                    "similar_benchmarks": ["SuperGLUE", "XTREME", "BigBench"],
                    "resources": [
                        "https://gluebenchmark.com/",
                        "https://arxiv.org/abs/1804.07461",
                    ],
                },
                "bad_example": {
                    "name": "prompt_leakage.glue",
                    "overview": "natural language understanding",
                    "data_type": "text",
                    "domains": ["NLP"],
                    "languages": ["en"],
                    "similar_benchmarks": ["D1", "D2"],
                    "resources": ["paper", "dataset"],
                },
            },
            "purpose_and_intended_users": {
                "good_example": {
                    "goal": "To provide a comprehensive evaluation framework for natural language understanding systems across multiple tasks, enabling researchers to assess model performance on diverse linguistic phenomena and compare different approaches systematically.",
                    "audience": [
                        "NLP researchers",
                        "machine learning engineers",
                        "academic institutions",
                        "AI practitioners",
                    ],
                    "tasks": [
                        "sentiment analysis",
                        "textual entailment",
                        "question answering",
                        "linguistic acceptability",
                    ],
                    "limitations": "Limited to English language, focuses primarily on sentence-level tasks, may not capture all aspects of language understanding",
                    "out_of_scope_uses": [
                        "Real-time production systems without proper validation",
                        "Non-English language evaluation",
                        "Document-level understanding tasks",
                    ],
                }
            },
            "data": {
                "good_example": {
                    "source": "Collected from diverse sources including movie reviews (SST-2), news articles (RTE), and crowdsourced annotations (MNLI), with careful preprocessing and quality control measures",
                    "size": "108,000 total examples across 9 tasks with train/dev/test splits varying by task",
                    "format": "JSON format with task-specific fields including sentence pairs, labels, and metadata",
                    "annotation": "Professional annotators and crowdsourcing with quality control, inter-annotator agreement validation, and expert review",
                },
                "bad_example": {
                    "source": "various sources",
                    "size": "large dataset",
                    "format": "text",
                    "annotation": "manual annotation",
                },
            },
            "methodology": {
                "good_example": {
                    "methods": [
                        "fine-tuning on task-specific data",
                        "zero-shot evaluation",
                        "few-shot learning",
                    ],
                    "metrics": [
                        "accuracy",
                        "F1-score",
                        "Matthews correlation coefficient",
                    ],
                    "calculation": "Accuracy computed as correct predictions divided by total predictions, F1-score as harmonic mean of precision and recall, with macro-averaging across classes",
                    "interpretation": "Scores range from 0-100, with higher scores indicating better performance. Baseline human performance is approximately 87% accuracy across tasks",
                    "baseline_results": "BERT-large achieves 80.5% average score, RoBERTa-large reaches 88.9%, with task-specific variations ranging from 68.6% to 96.4%",
                    "validation": "Cross-validation on development sets, held-out test sets, statistical significance testing, and comparison with human performance",
                }
            },
        }

        section_example = few_shot_examples.get(section_name, {})
        example_text = ""
        if section_example:
            if "good_example" in section_example:
                good_json = (
                    json.dumps(section_example["good_example"], indent=2)
                    .replace("{", "{{")
                    .replace("}", "}}")
                )
                example_text += f"\n\nGOOD EXAMPLE:\n{good_json}"
            if "bad_example" in section_example:
                bad_json = (
                    json.dumps(section_example["bad_example"], indent=2)
                    .replace("{", "{{")
                    .replace("}", "}}")
                )
                example_text += f"\n\nBAD EXAMPLE (avoid this):\n{bad_json}"

        # set up section-specific prompt with enhanced instructions
        section_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are an AI evaluation researcher. Generate a {section_class.__name__} object for the '{section_name}' section.

RULES:
1. Use ONLY information from the provided metadata sources
2. If information is missing, write exactly: "Not specified"
3. Do NOT use your training data or make assumptions
4. Be concise and specific
5. Return only valid JSON

FORBIDDEN:
- Generic examples (e.g., "BERT-large achieves 80.5%")
- Placeholder names (e.g., "D1", "D2") unless in metadata
- Invented metrics or performance numbers
- Fake URLs or resources
- Rambling or repetitive text

{example_text}""",
                ),
                (
                    "user",
                    f"""Query: {{query}}

METADATA:
Unitxt: {{unitxt_metadata}}
HuggingFace: {{hf_metadata}}
Extracted IDs: {{extracted_ids}}
Paper Content: {{docling_output}}

Generate {section_name} section using ONLY the metadata above. If information is missing, use "Not specified".""",
                ),
            ]
        )

        # configure for structured output
        llm_with_structure = LLM.with_structured_output(section_class)

        # create and run the chain
        chain = section_prompt | llm_with_structure

        # Retry logic for robust generation
        max_retries = 3
        for attempt in range(max_retries):
            try:
                section_result = chain.invoke(
                    {
                        "unitxt_metadata": unitxt_metadata,
                        "hf_metadata": hf_metadata or "Not available",
                        "extracted_ids": extracted_ids or "Not available",
                        "docling_output": docling_output or "Not available",
                        "query": query,
                    }
                )

                generated_sections[section_name] = section_result.model_dump()
                logger.debug("%s completed", section_name.replace("_", " ").title())
                logger.debug("Preview: %s", str(section_result.model_dump())[:100] + "...")
                break  # Success, exit retry loop

            except Exception as e:
                attempt_msg = f"(attempt {attempt + 1}/{max_retries})"
                if attempt < max_retries - 1:
                    logger.warning("Failed to generate %s %s: %s", section_name, attempt_msg, e)
                    logger.debug("Retrying %s", section_name)
                    continue
                else:
                    logger.error(
                        "Failed to compose %s section after %d attempts: %s",
                        section_name,
                        max_retries,
                        e,
                    )
                    logger.error(
                        "Failed to generate %s after %d attempts: %s",
                        section_name,
                        max_retries,
                        e,
                    )
                    raise

    # combine all sections into final benchmark card
    logger.debug("Combining all sections into final benchmark card")

    try:
        final_card = BenchmarkCard(
            benchmark_details=BenchmarkDetails(**generated_sections["benchmark_details"]),
            purpose_and_intended_users=PurposeAndIntendedUsers(
                **generated_sections["purpose_and_intended_users"]
            ),
            data=DataInfo(**generated_sections["data"]),
            methodology=Methodology(**generated_sections["methodology"]),
            ethical_and_legal_considerations=EthicalAndLegalConsiderations(
                **generated_sections["ethical_and_legal_considerations"]
            ),
        )

        logger.debug("Final benchmark card assembled successfully")

    except Exception as e:
        logger.error("Failed to assemble final benchmark card: %s", e)
        logger.error("Failed to assemble final card: %s", e)
        raise

    # add metadata about the composition process
    return {
        "benchmark_card": final_card.model_dump(),
        "composition_metadata": {
            "sources_used": {
                "unitxt": bool(unitxt_metadata),
                "huggingface": bool(hf_metadata),
                "extracted_ids": bool(extracted_ids),
                "docling": bool(docling_output),
            },
            "query": query,
            "composition_timestamp": datetime.now().isoformat(),
            "generation_method": "chunked_sections",
            "model_used": LLM.model_name,
        },
    }

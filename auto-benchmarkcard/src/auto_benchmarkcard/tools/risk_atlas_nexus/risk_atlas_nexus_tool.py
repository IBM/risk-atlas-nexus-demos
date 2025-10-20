"""Risk Atlas Nexus integration for AI risk identification.

This module integrates the Risk Atlas Nexus library to automatically identify
relevant AI risks based on benchmark metadata. It creates use case descriptions
from benchmark cards and applies risk detection models to map benchmarks to
specific risks in the IBM AI Risk Atlas taxonomy.

Key functionality:
- Use case generation from benchmark card metadata
- Risk identification using BenchmarkRiskDetector
- Integration with RITS inference engine for risk classification
- Filtering and ranking of detected risks
"""

import logging
import os
from typing import Any, Dict, List, Optional

# Suppress noisy logging from external libraries
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

from risk_atlas_nexus.blocks.risk_detector import BenchmarkRiskDetector
from risk_atlas_nexus.library import RiskAtlasNexus

logger = logging.getLogger(__name__)


def identify_risks_with_benchmark_detector(
    risk_atlas_nexus: RiskAtlasNexus,
    usecases: List[str],
    inference_engine,
    taxonomy: str = "ibm-ai-risk-atlas",
    max_risk: Optional[int] = None,
) -> List[List]:
    """Identify risks using the custom BenchmarkRiskDetector.

    Uses a specialized risk detector that matches benchmark use cases to
    known AI risks in the specified taxonomy. Returns ranked risks based
    on relevance to the provided use cases.

    Args:
        risk_atlas_nexus: RiskAtlasNexus library instance.
        usecases: List of use case descriptions from benchmark metadata.
        inference_engine: Inference engine for risk classification.
        taxonomy: Risk taxonomy identifier (default: "ibm-ai-risk-atlas").
        max_risk: Maximum number of risks to return per use case.

    Returns:
        List of Risk object lists, one list per input use case.

    Example:
        >>> risks = identify_risks_with_benchmark_detector(
        ...     ran, ["Hate speech detection in social media"],
        ...     engine, max_risk=5
        ... )
    """
    try:
        # Get all risks for the specified taxonomy
        all_risks = risk_atlas_nexus.get_all_risks(taxonomy)

        # Create the custom benchmark risk detector
        benchmark_detector = BenchmarkRiskDetector(
            risks=all_risks,
            inference_engine=inference_engine,
            cot_examples=None,  # Using zero-shot for now
            max_risk=max_risk,
        )

        # Detect risks using the benchmark-specific detector
        return benchmark_detector.detect(usecases)

    except Exception as e:
        logger.error("Error in benchmark risk detection: %s", e)
        return []


def create_inference_engine():
    """Create a RITS inference engine for risk identification.

    Uses LLMHandler to create the engine, ensuring consistent configuration
    and verbose settings across the application.

    Returns:
        RITSInferenceEngine instance if successful, None otherwise.
    """
    try:
        from auto_benchmarkcard.llm_handler import LLMHandler

        # Create handler with RITS engine for risk identification
        handler = LLMHandler(
            engine_type="rits",
            model_name="ibm/granite-20b-code-instruct",
            credentials={
                "api_key": os.getenv("RITS_API_KEY"),
                "api_url": os.getenv("RITS_API_URL"),
            },
            parameters={"max_completion_tokens": 1000, "temperature": 0.7},
            verbose=False  # Disable progress bars for cleaner output
        )

        # Return the underlying risk-atlas-nexus engine
        # BenchmarkRiskDetector needs the raw engine, not the wrapper
        return handler.engine

    except Exception as e:
        logger.warning("Failed to create RITS inference engine: %s", e)
        logger.warning(
            "Risk identification will be skipped. Set RITS_API_KEY and RITS_API_URL environment variables."
        )
        return None


def identify_risks_from_benchmark_metadata(
    benchmark_card: Dict[str, Any], taxonomy: str = "ibm-risk-atlas", max_risk: int = 5
) -> Optional[List[Dict[str, Any]]]:
    """Identify risks from benchmark metadata using Risk Atlas Nexus.

    Args:
        benchmark_card: The composed benchmark card containing metadata.
        taxonomy: The risk taxonomy to use (default: "ibm-risk-atlas").
        max_risk: Maximum number of risks to identify (default: 5).

    Returns:
        List of risk dictionaries or None if identification fails.
    """
    try:
        # Create inference engine
        inference_engine = create_inference_engine()
        if not inference_engine:
            logger.warning("No inference engine available - skipping risk identification")
            return None

        # Create Risk Atlas Nexus instance
        risk_atlas_nexus = RiskAtlasNexus()

        # Extract usecase description from benchmark metadata
        usecase = create_usecase_from_benchmark_card(benchmark_card)
        if not usecase:
            logger.warning("Could not create usecase description from benchmark card")
            return None

        logger.debug("🔄 Identifying potential AI risks...")

        # Use custom benchmark risk detector instead of generic one
        risks = identify_risks_with_benchmark_detector(
            risk_atlas_nexus=risk_atlas_nexus,
            usecases=[usecase],
            inference_engine=inference_engine,
            taxonomy=taxonomy,
            max_risk=max_risk,
        )

        # Extract the first (and only) result from the nested list structure
        if risks and len(risks) > 0 and len(risks[0]) > 0:
            risk_objects = risks[0][:max_risk]  # Get first usecase's risks, limited to max_risk

            # Convert Risk objects to dictionary format for benchmark card
            formatted_risks = []
            for risk_obj in risk_objects:
                formatted_risk = {
                    "id": risk_obj.id,
                    "category": risk_obj.name,
                    "description": [risk_obj.description],
                    "tag": risk_obj.tag,
                    "type": risk_obj.type,
                    "concern": risk_obj.concern,
                    "url": risk_obj.url if risk_obj.url else None,
                    "taxonomy": risk_obj.isDefinedByTaxonomy,
                }
                formatted_risks.append(formatted_risk)

            logger.debug("✅ Identified %d potential risks", len(formatted_risks))
            return formatted_risks
        else:
            logger.debug("✅ No specific risks identified")
            return []

    except Exception as e:
        logger.error("Error identifying risks: %s", e)
        return None


def create_usecase_from_benchmark_card(benchmark_card: Dict[str, Any]) -> Optional[str]:
    """Create a usecase description from benchmark card metadata.

    Args:
        benchmark_card: The benchmark card containing metadata.

    Returns:
        Usecase description string or None if insufficient data.
    """
    try:
        # Extract key information from benchmark card
        details = benchmark_card.get("benchmark_details", {})
        purpose = benchmark_card.get("purpose_and_intended_users", {})

        name = details.get("name", "")
        overview = details.get("overview", "")
        domains = details.get("domains", [])
        tasks = purpose.get("tasks", [])
        goal = purpose.get("goal", "")

        # Build usecase description similar to the example format
        usecase_parts = []

        if name:
            usecase_parts.append(f"{name} is a benchmark.")

        if overview:
            usecase_parts.append(f"{overview.lower()}")

        if goal:
            usecase_parts.append(f"with the goal of {goal.lower()}")

        # Add domains if available
        if isinstance(domains, list) and domains:
            domain_str = ", ".join(domains) if len(domains) > 1 else domains[0]
            usecase_parts.append(f"Domains: {domain_str}")
        elif isinstance(domains, str) and domains:
            usecase_parts.append(f"Domains: {domains}")

        # Add tasks if available
        if isinstance(tasks, list) and tasks:
            task_str = ", ".join(tasks) if len(tasks) > 1 else tasks[0]
            usecase_parts.append(f"Tasks: {task_str}")
        elif isinstance(tasks, str) and tasks:
            usecase_parts.append(f"Tasks: {tasks}")

        if usecase_parts:
            usecase = ". ".join(usecase_parts).strip()
            if not usecase.endswith("."):
                usecase += "."
            return usecase
        else:
            logger.warning("Insufficient information to create usecase description")
            return None

    except Exception as e:
        logger.error("Error creating usecase from benchmark card: %s", e)
        return None


def integrate_risks_into_benchmark_card(
    benchmark_card: Dict[str, Any], risks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Integrate identified risks into the benchmark card.

    Args:
        benchmark_card: The original benchmark card.
        risks: List of identified risks.

    Returns:
        Updated benchmark card with risks integrated.
    """
    try:
        # Make a copy to avoid modifying the original
        updated_card = benchmark_card.copy()

        # Remove id and tag fields from risks
        cleaned_risks = []
        for risk in risks:
            cleaned_risk = {
                "category": risk["category"],
                "description": risk["description"],
                "type": risk["type"],
                "concern": risk["concern"],
                "url": risk["url"] if risk["url"] else None,
                "taxonomy": risk["taxonomy"],
            }
            cleaned_risks.append(cleaned_risk)

        # Add risks directly to possible_risks (renamed from targeted_risks)
        updated_card["possible_risks"] = cleaned_risks

        logger.debug("Successfully integrated %d risks into benchmark card", len(cleaned_risks))
        return updated_card

    except Exception as e:
        logger.error("Error integrating risks into benchmark card: %s", e)
        return benchmark_card


# Main function that can be called from the workflow
def identify_and_integrate_risks(benchmark_card: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to identify risks and integrate them into benchmark card.

    Args:
        benchmark_card: The composed benchmark card.

    Returns:
        Updated benchmark card with risks (or original if identification fails).
    """
    try:
        logger.debug("🔄 Running Risk Atlas Nexus analysis...")

        # Identify risks
        risks = identify_risks_from_benchmark_metadata(benchmark_card)

        if risks is None:
            logger.warning("Risk identification failed - returning original benchmark card")
            return benchmark_card

        if not risks:
            logger.debug("No risks identified - returning original benchmark card")
            return benchmark_card

        # Integrate risks into benchmark card
        updated_card = integrate_risks_into_benchmark_card(benchmark_card, risks)

        logger.debug("✅ Risk analysis completed")
        return updated_card

    except Exception as e:
        logger.error("Error in risk identification and integration: %s", e)
        return benchmark_card

"""Benchmark card atomization for fact verification.

Breaks down benchmark cards into atomic factual statements that can be
independually verified against source documents.
"""

import json
import logging
from typing import Any, Dict, List

from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from auto_benchmarkcard.llm_handler import LLMHandler
except ImportError:
    logger.warning("LLM handler not available for atomization")
    LLMHandler = None

BENCHMARK_CARD_ATOMIZATION_PROMPT = """
Extract verifiable facts from this benchmark card. Each statement must be independently verifiable.

CRITICAL: Use the EXACT field path (dot notation) where the information appears in the JSON structure.

Required Field Paths (use these EXACT paths):
- benchmark_details.name
- benchmark_details.overview
- benchmark_details.data_type
- benchmark_details.domains
- benchmark_details.languages
- benchmark_details.similar_benchmarks
- benchmark_details.resources
- purpose_and_intended_users.goal
- purpose_and_intended_users.audience
- purpose_and_intended_users.tasks
- purpose_and_intended_users.limitations
- purpose_and_intended_users.out_of_scope_uses
- data.source
- data.size
- data.format
- data.annotation
- methodology.methods
- methodology.metrics
- methodology.calculation
- methodology.interpretation
- methodology.baseline_results
- methodology.validation
- ethical_and_legal_considerations.privacy_and_anonymity
- ethical_and_legal_considerations.data_licensing
- ethical_and_legal_considerations.consent_procedures
- ethical_and_legal_considerations.compliance_with_regulations

Rules:
1. Extract specific, verifiable claims (numbers, names, dates, metrics)
2. Each statement gets ONE field path showing exactly where the info appears
3. Use format: "- Statement text [exact.field.path]"
4. Focus on concrete facts, not general descriptions

Examples:
Input: {{"data": {{"size": "10,000 examples"}}, "methodology": {{"interpretation": "The best model achieved 85.2% accuracy"}}}}
Output:
- The dataset contains 10,000 examples [data.size]
- The best model achieved 85.2% accuracy [methodology.interpretation]

Input: {{"benchmark_details": {{"name": "ETHOS"}}, "methodology": {{"calculation": "F1-score measures precision and recall"}}}}
Output:
- The benchmark name is ETHOS [benchmark_details.name]
- F1-score measures precision and recall [methodology.calculation]

Now extract facts from:
{benchmark_card}

Atomic Statements:
"""


def text_to_statements(text: str, separator: str = "- ") -> List[dict]:
    """Parse LLM output into structured atomic statements.

    Extracts statements from LLM-generated text where each statement is
    prefixed with a separator and includes field attribution in brackets.
    Validates field paths against allowed schema and applies fallback mapping
    for invalid paths.

    Args:
        text: Raw LLM output text containing statements.
        separator: Prefix marking statement lines (default: "- ").

    Returns:
        List of dictionaries with 'text' and 'field' keys. Field may be None
        if attribution couldn't be determined.

    Example:
        >>> text = "- ETHOS is a hate speech benchmark [benchmark_details.name]"
        >>> text_to_statements(text)
        [{'text': 'ETHOS is a hate speech benchmark', 'field': 'benchmark_details.name'}]
    """
    # Valid field paths for validation
    valid_fields = {
        "benchmark_details.name",
        "benchmark_details.overview",
        "benchmark_details.data_type",
        "benchmark_details.domains",
        "benchmark_details.languages",
        "benchmark_details.similar_benchmarks",
        "benchmark_details.resources",
        "purpose_and_intended_users.goal",
        "purpose_and_intended_users.audience",
        "purpose_and_intended_users.tasks",
        "purpose_and_intended_users.limitations",
        "purpose_and_intended_users.out_of_scope_uses",
        "data.source",
        "data.size",
        "data.format",
        "data.annotation",
        "methodology.methods",
        "methodology.metrics",
        "methodology.calculation",
        "methodology.interpretation",
        "methodology.baseline_results",
        "methodology.validation",
        "ethical_and_legal_considerations.privacy_and_anonymity",
        "ethical_and_legal_considerations.data_licensing",
        "ethical_and_legal_considerations.consent_procedures",
        "ethical_and_legal_considerations.compliance_with_regulations",
    }

    statements = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith(separator):
            statement = line[len(separator) :].strip()
            if not statement:
                continue

            # Extract field attribution from brackets
            field = None
            if statement.endswith("]") and "[" in statement:
                bracket_start = statement.rfind("[")
                if bracket_start != -1:
                    field = statement[bracket_start + 1 : -1].strip()
                    statement = statement[:bracket_start].strip()

            # Validate field path
            if field and field not in valid_fields:
                logger.warning(f"Invalid field path '{field}' - using fallback mapping")
                field = _map_to_valid_field(field, statement)

            statements.append({"text": statement, "field": field})
    return statements


def _map_to_valid_field(field: str, statement: str) -> str:
    """Map invalid field paths to valid ones based on content analysis.

    Uses keyword matching to infer the correct benchmark card field path
    when the LLM provides an invalid or non-existent path. Falls back to
    broad categories if specific mapping cannot be determined.

    Args:
        field: Invalid field path from LLM output.
        statement: The statement text for context-based inference.

    Returns:
        Valid field path from the benchmark card schema.

    Example:
        >>> _map_to_valid_field("methodology.result", "Model achieved 85% accuracy")
        'methodology.interpretation'
    """
    statement_lower = statement.lower()

    # Methodology-related mappings
    if "methodology" in field.lower():
        if any(
            word in statement_lower
            for word in ["f1-score", "accuracy", "performance", "achieved", "result"]
        ):
            return "methodology.interpretation"
        elif any(word in statement_lower for word in ["kappa", "measure", "calculate", "formula"]):
            return "methodology.calculation"
        elif any(word in statement_lower for word in ["metric", "precision", "recall"]):
            return "methodology.metrics"
        elif any(word in statement_lower for word in ["method", "approach", "technique"]):
            return "methodology.methods"
        elif any(word in statement_lower for word in ["validation", "annotator", "judgement"]):
            return "methodology.validation"
        else:
            return "methodology.interpretation"  # Default for methodology

    # Data-related mappings
    elif "data" in field.lower():
        if any(word in statement_lower for word in ["size", "count", "example", "comment"]):
            return "data.size"
        elif any(word in statement_lower for word in ["format", "csv", "json"]):
            return "data.format"
        elif any(word in statement_lower for word in ["annotator", "annotation", "platform"]):
            return "data.annotation"
        else:
            return "data.source"  # Default for data

    # Other mappings
    elif "benchmark_details" in field.lower():
        if "name" in statement_lower:
            return "benchmark_details.name"
        else:
            return "benchmark_details.overview"

    elif "purpose" in field.lower():
        return "purpose_and_intended_users.goal"

    elif "ethical" in field.lower() or "legal" in field.lower():
        if "license" in statement_lower:
            return "ethical_and_legal_considerations.data_licensing"
        else:
            return "ethical_and_legal_considerations.privacy_and_anonymity"

    # Fallback to broad category if we can't determine specific field
    return field if field else "unknown"


def benchmark_card_to_text(benchmark_card: Dict[str, Any]) -> str:
    """Convert benchmark card to formatted JSON text.

    Args:
        benchmark_card: Benchmark card dictionary structure.

    Returns:
        Pretty-printed JSON string with 2-space indentation.
    """
    return json.dumps(benchmark_card, indent=2)


class BenchmarkCardAtomizer:
    """Atomizes benchmark cards into verifiable facts.

    Attributes:
        engine_type: Type of LLM engine to use.
        model_name: Name of the LLM model.
        llm_handler: LLM handler instance for generation.
    """

    def __init__(self, engine_type: str = "rits", model_name: str = None, **kwargs):
        if LLMHandler is None:
            raise ImportError("LLM handler required for atomization")

        self.engine_type = engine_type
        self.model_name = model_name
        # Set verbose=False by default unless overridden in kwargs
        if 'verbose' not in kwargs:
            kwargs['verbose'] = False
        self.llm_handler = LLMHandler(engine_type=engine_type, model_name=model_name, **kwargs)

        logger.debug(f"Atomizer ready: {self.llm_handler.engine_type}")

    def make_prompt(self, benchmark_card: Dict[str, Any]) -> str:
        """Create atomization prompt.

        Args:
            benchmark_card: Benchmark card dictionary to atomize.

        Returns:
            Formatted prompt string for LLM.
        """
        benchmark_text = benchmark_card_to_text(benchmark_card)
        return BENCHMARK_CARD_ATOMIZATION_PROMPT.format(benchmark_card=benchmark_text)

    def atomize_single(self, benchmark_card: Dict[str, Any]) -> List[dict]:
        """Extract atomic statements from benchmark card.

        Args:
            benchmark_card: Benchmark card dictionary to atomize.

        Returns:
            List of atomic statement dictionaries with 'text' and 'field' keys.
        """
        prompt = self.make_prompt(benchmark_card)
        response = self.llm_handler.generate(prompt)
        return text_to_statements(response)

    def atomize_batch(self, benchmark_cards: List[Dict[str, Any]]) -> List[List[dict]]:
        """Process multiple benchmark cards.

        Args:
            benchmark_cards: List of benchmark card dictionaries.

        Returns:
            List of atomic statement lists, one per input card.
        """
        results = []
        for card in tqdm(benchmark_cards, desc="Atomizing cards"):
            results.append(self.atomize_single(card))
        return results


def exclude_risk_sections(benchmark_card: Dict[str, Any]) -> Dict[str, Any]:
    """Remove risk sections from benchmark card.

    Risk information is not fact-checked as it's inferred rather than
    directly stated in source documents.

    Args:
        benchmark_card: Original benchmark card dictionary.

    Returns:
        Filtered benchmark card without risk sections.
    """
    import copy

    filtered_card = copy.deepcopy(benchmark_card)

    if "targeted_risks" in filtered_card:
        logger.debug("Excluding risk sections from fact verification")
        del filtered_card["targeted_risks"]

    return filtered_card


def atomize_benchmark_card(
    benchmark_card: Dict[str, Any],
    field: str = "all",
    engine_type: str = "rits",
    model_name: str = None,
) -> List[dict]:
    """Extract atomic statements from benchmark card.

    Excludes risk sections as they're inferred rather than factual.

    Args:
        benchmark_card: Benchmark card data.
        field: Field to focus on (kept for compatibility).
        engine_type: LLM engine type.
        model_name: Specific model to use.

    Returns:
        List of atomic statements with field attribution.
    """
    if LLMHandler is None:
        raise ImportError("LLM handler required for atomization")

    filtered_card = exclude_risk_sections(benchmark_card)

    try:
        atomizer = BenchmarkCardAtomizer(engine_type=engine_type, model_name=model_name)
        return atomizer.atomize_single(filtered_card)
    except Exception as e:
        logger.warning(f"Atomization failed: {e}")
        logger.debug("Using fallback extraction")

        # Basic fallback
        statements = []
        if "benchmark_details" in filtered_card and "name" in filtered_card["benchmark_details"]:
            statements.append(
                {
                    "text": f"The benchmark is called {filtered_card['benchmark_details']['name']}",
                    "field": "benchmark_details",
                }
            )
        return statements

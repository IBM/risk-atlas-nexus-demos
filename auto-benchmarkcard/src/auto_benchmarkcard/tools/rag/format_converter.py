import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def generate_output_from_benchmark_card(benchmark_card: Dict[str, Any], field: str) -> str:
    """Generate output text from benchmark card based on requested field.

    Args:
        benchmark_card: The benchmark card data.
        field: The field being queried.

    Returns:
        Generated output text.
    """
    output_parts = []

    if field == "description" or field == "overview":
        if "benchmark_details" in benchmark_card:
            details = benchmark_card["benchmark_details"]
            if "overview" in details:
                output_parts.append(details["overview"])
            if "domains" in details:
                domain_text = f"It is used for {', '.join(details['domains'])}."
                output_parts.append(domain_text)
            if "languages" in details:
                lang_text = f"It supports {', '.join(details['languages'])} language."
                output_parts.append(lang_text)

    elif field == "purpose":
        if "purpose_and_intended_users" in benchmark_card:
            purpose = benchmark_card["purpose_and_intended_users"]
            if "goal" in purpose:
                output_parts.append(purpose["goal"])
            if "tasks" in purpose:
                tasks_text = f"The benchmark includes tasks such as: {', '.join(purpose['tasks'])}."
                output_parts.append(tasks_text)

    elif field == "data":
        if "data" in benchmark_card:
            data = benchmark_card["data"]
            data_info = []
            if "size" in data:
                data_info.append(f"dataset size is {data['size']}")
            if "format" in data:
                data_info.append(f"data format is {data['format']}")
            if "source" in data:
                data_info.append(f"data source is {data['source']}")
            if data_info:
                output_parts.append(f"The {', '.join(data_info)}.")

    elif field == "methodology":
        if "methodology" in benchmark_card:
            methodology = benchmark_card["methodology"]
            if "metrics" in methodology:
                metrics_text = f"The benchmark uses {', '.join(methodology['metrics'])} as metrics."
                output_parts.append(metrics_text)
            if "evaluation_approach" in methodology:
                eval_text = f"The evaluation approach is {methodology['evaluation_approach']}."
                output_parts.append(eval_text)

    # If no specific content found, fall back to overview
    if not output_parts and "benchmark_details" in benchmark_card:
        details = benchmark_card["benchmark_details"]
        if "overview" in details:
            output_parts.append(details["overview"])

    return " ".join(output_parts)


def convert_rag_to_required_format(
    rag_results: Dict[str, Any],
    benchmark_field: str = "description",
    benchmark_card: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Convert RAG results to the required format with atomic statements and contexts.

    Args:
        rag_results: The RAG results from the existing pipeline.
        benchmark_field: The field being queried (e.g., "description", "overview").
        benchmark_card: Optional benchmark card for output generation.

    Returns:
        Formatted output matching the required structure.
    """

    benchmark_name = rag_results.get("benchmark", "unknown")

    # Create the base structure
    output = {
        "input": f"Question: Tell me about the benchmark {benchmark_name}",
        "output": "",  # Will be populated from benchmark card
        "topic": benchmark_name,
        "cat": ["benchmark", benchmark_name],
        "atoms": [],
        "contexts": [],
    }

    # Process each statement and its retrieved chunks
    for atom_idx, result in enumerate(rag_results.get("results", [])):
        statement_data = result.get("statement", {})

        # Handle both old format (string) and new format (dict)
        if isinstance(statement_data, str):
            statement_text = statement_data
            statement_field = None
        else:
            statement_text = statement_data.get("text", "")
            statement_field = statement_data.get("field")

        retrieved_chunks = result.get("retrieved_chunks", [])

        # Create atom ID
        atom_id = f"a{atom_idx}"

        # Create context IDs for this atom
        context_ids = []
        for chunk_idx, chunk in enumerate(retrieved_chunks):
            context_id = f"c_{atom_id}_{chunk_idx}"
            context_ids.append(context_id)

            # Add to contexts array
            output["contexts"].append(
                {
                    "id": context_id,
                    "title": benchmark_name,
                    "text": chunk.get("content", ""),
                }
            )

        # Add atom with field information
        atom = {
            "id": atom_id,
            "text": statement_text,
            "original": statement_text,
            "label": "S",
            "contexts": context_ids,
        }

        # Add field information if available
        if statement_field:
            atom["field"] = statement_field

        output["atoms"].append(atom)

    # Generate output text from benchmark card if available, otherwise combine atomic statements
    if benchmark_card:
        output_text = generate_output_from_benchmark_card(benchmark_card, benchmark_field)
        output["output"] = output_text
    else:
        # Fallback: combine atomic statements
        output_text_parts = []
        for atom in output["atoms"]:
            output_text_parts.append(atom["text"])
        output["output"] = " ".join(output_text_parts)

    return output


def load_and_convert_rag_results(
    rag_results_path: str, benchmark_field: str = "description"
) -> Dict[str, Any]:
    """Load RAG results from file and convert to required format.

    Args:
        rag_results_path: Path to the RAG results JSON file.
        benchmark_field: The field being queried.

    Returns:
        Formatted output matching the required structure.
    """

    with open(rag_results_path, "r") as f:
        rag_results = json.load(f)

    return convert_rag_to_required_format(rag_results, benchmark_field)


def save_formatted_results(formatted_results: Dict[str, Any], output_path: str):
    """Save formatted results to file in JSONL format.

    Args:
        formatted_results: The formatted results dictionary.
        output_path: Path where to save the results.
    """

    with open(output_path, "w") as f:
        f.write(json.dumps(formatted_results) + "\n")

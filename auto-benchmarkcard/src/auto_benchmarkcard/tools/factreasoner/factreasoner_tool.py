import copy
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import os

# Suppress noisy logging from external libraries
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("fact_reasoner").setLevel(logging.ERROR)
logging.getLogger("FactReasoner").setLevel(logging.ERROR)

# Suppress specific FactReasoner component loggers (minimal suppression)
logging.getLogger("LLMHandler").setLevel(logging.WARNING)
logging.getLogger("AtomExtractor").setLevel(logging.WARNING)
logging.getLogger("AtomReviser").setLevel(logging.WARNING)
logging.getLogger("NLIExtractor").setLevel(logging.WARNING)
logging.getLogger("FactReasoner").setLevel(logging.WARNING)

import matplotlib.pyplot as plt

import fact_reasoner.fact_utils as fact_utils
from fact_reasoner.atom_extractor import AtomExtractor
from fact_reasoner.atom_reviser import AtomReviser
from fact_reasoner.context_retriever import ContextRetriever
from fact_reasoner.fact_graph import FactGraph
from fact_reasoner.fact_utils import Relation
from fact_reasoner.factreasoner import FactReasoner
from fact_reasoner.nli_extractor import NLIExtractor

logger = logging.getLogger(__name__)


def _create_atom_marginal_mappings(
    formatted_rag_results: Dict[str, Any], marginals: List[Dict[str, Any]]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Create mappings from atom ID to atom data and variable name to marginal data.

    Args:
        formatted_rag_results: RAG output containing atoms list
        marginals: List of marginal probability dictionaries from FactReasoner

    Returns:
        Tuple of (atoms_by_id dict, marginals_by_var dict)
    """
    # Create mapping from atom ID to atom data
    atoms_by_id = {}
    for atom in formatted_rag_results.get("atoms", []):
        atoms_by_id[atom["id"]] = atom

    # Create mapping from variable name to marginal data
    marginals_by_var = {}
    for marginal in marginals:
        var_name = marginal.get("variable", "")
        marginals_by_var[var_name] = marginal

    return atoms_by_id, marginals_by_var


def _find_marginal_for_atom(
    atom_id: str, atom_text: str, marginals_by_var: Dict[str, Any]
) -> Dict[str, Any] | None:
    """Find the corresponding marginal data for an atom.

    Args:
        atom_id: Unique atom identifier
        atom_text: Text content of the atom
        marginals_by_var: Dictionary mapping variable names to marginal data

    Returns:
        Marginal dictionary if found, None otherwise
    """
    # Try to match by atom ID or text similarity
    for var_name, marginal in marginals_by_var.items():
        if atom_id in var_name or atom_text[:50] in var_name:
            return marginal

    # Try exact atom ID match
    return marginals_by_var.get(atom_id)


def _determine_flag_reason(field_stats: Dict[str, Any], threshold: float) -> tuple[bool, str, str]:
    """Determine if field should be flagged based on probability statistics.

    Args:
        field_stats: Dictionary with 'average_probability' and 'flagged_count' keys
        threshold: Probability threshold for flagging (typically 0.5)

    Returns:
        Tuple of (should_flag: bool, severity: str, reason: str)
    """
    """Determine if a field should be flagged and why."""
    avg_probability = field_stats.get("avg_probability", 1.0)
    all_neutral = field_stats.get("all_neutral", False)
    neutral_count = field_stats.get("neutral_count", 0)

    # Flag field if average probability below threshold OR if all atoms are neutral
    should_flag = avg_probability < threshold or all_neutral

    if not should_flag:
        return False, "", ""

    # Determine the reason for flagging
    if all_neutral:
        reason = "all_atoms_neutral"
        reason_desc = f"All {neutral_count} atoms have neutral scores (no evidence found)"
    else:
        reason = "low_factuality_score"
        reason_desc = f"Average factuality score {avg_probability:.3f} below threshold {threshold}"

    return True, reason, reason_desc


def analyze_factuality_by_field(
    formatted_rag_results: Dict[str, Any], marginals: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze factuality scores by benchmark card field.

    Args:
        formatted_rag_results: RAG results containing atoms with field information.
        marginals: Marginal probabilities for each atom from FactReasoner.

    Returns:
        Field-level analysis including error counts and accuracy metrics.
    """

    atoms_by_id, marginals_by_var = _create_atom_marginal_mappings(formatted_rag_results, marginals)

    # Aggregate results by field
    field_stats = {}
    for atom_id, atom in atoms_by_id.items():
        field = atom.get("field", "unknown")
        atom_text = atom.get("text", "")

        # Find corresponding marginal (FactReasoner may use different variable naming)
        marginal_data = _find_marginal_for_atom(atom_id, atom_text, marginals_by_var)

        if marginal_data:
            p_true = marginal_data.get("p_true", marginal_data.get("probabilities", [0, 0])[1])

            # Initialize field stats if not exists
            if field not in field_stats:
                field_stats[field] = {
                    "total_atoms": 0,
                    "high_confidence_correct": 0,  # p_true > 0.8
                    "likely_correct": 0,  # p_true > 0.6
                    "uncertain": 0,  # 0.4 <= p_true <= 0.6
                    "likely_incorrect": 0,  # p_true < 0.4
                    "high_confidence_incorrect": 0,  # p_true < 0.2
                    "avg_probability": 0,
                    "probabilities": [],
                    "atoms": [],
                }

            # Update field statistics
            field_stats[field]["total_atoms"] += 1
            field_stats[field]["probabilities"].append(p_true)
            field_stats[field]["atoms"].append(
                {
                    "id": atom_id,
                    "text": atom_text,
                    "p_true": p_true,
                    "variable": marginal_data.get("variable", ""),
                }
            )

            # Categorize by confidence level
            if p_true > 0.8:
                field_stats[field]["high_confidence_correct"] += 1
            elif p_true > 0.6:
                field_stats[field]["likely_correct"] += 1
            elif p_true >= 0.4:
                field_stats[field]["uncertain"] += 1
            elif p_true >= 0.2:
                field_stats[field]["likely_incorrect"] += 1
            else:
                field_stats[field]["high_confidence_incorrect"] += 1

    # Calculate averages and summary statistics
    summary = {
        "total_fields": len(field_stats),
        "fields_with_errors": 0,
        "most_problematic_field": None,
        "most_accurate_field": None,
        "overall_field_accuracy": {},
    }

    for field, stats in field_stats.items():
        if stats["total_atoms"] > 0:
            # Calculate average probability excluding neutral scores (exactly 0.5)
            non_neutral_probabilities = [p for p in stats["probabilities"] if p != 0.5]

            if len(non_neutral_probabilities) == 0:
                # All atoms are neutral (no evidence) - flag for review
                stats["avg_probability"] = 0.5
                stats["all_neutral"] = True
            else:
                # Average only non-neutral scores (both high support and contradictions)
                stats["avg_probability"] = sum(non_neutral_probabilities) / len(
                    non_neutral_probabilities
                )
                stats["all_neutral"] = False

            # Store counts for analysis
            stats["non_neutral_count"] = len(non_neutral_probabilities)
            stats["neutral_count"] = len(stats["probabilities"]) - len(non_neutral_probabilities)

            # Calculate accuracy percentage (atoms with p_true > 0.5)
            accurate_atoms = stats["high_confidence_correct"] + stats["likely_correct"]
            stats["accuracy_percentage"] = (accurate_atoms / stats["total_atoms"]) * 100

            # Calculate error percentage
            error_atoms = stats["likely_incorrect"] + stats["high_confidence_incorrect"]
            stats["error_percentage"] = (error_atoms / stats["total_atoms"]) * 100

            # Update summary
            if error_atoms > 0:
                summary["fields_with_errors"] += 1

            summary["overall_field_accuracy"][field] = stats["accuracy_percentage"]

    # Find most/least accurate fields
    if summary["overall_field_accuracy"]:
        most_accurate = max(summary["overall_field_accuracy"].items(), key=lambda x: x[1])
        least_accurate = min(summary["overall_field_accuracy"].items(), key=lambda x: x[1])

        summary["most_accurate_field"] = {
            "field": most_accurate[0],
            "accuracy": most_accurate[1],
        }
        summary["most_problematic_field"] = {
            "field": least_accurate[0],
            "accuracy": least_accurate[1],
        }

    return {"summary": summary, "field_details": field_stats}


def print_clean_atom_summary(
    formatted_rag_results: Dict[str, Any], marginals: List[Dict[str, Any]]
) -> None:
    """Print atoms in simple format: Atom X: text, Probability=score.

    Args:
        formatted_rag_results: RAG results containing atoms.
        marginals: Marginal probabilities for each atom.
    """
    atoms_by_id, marginals_by_var = _create_atom_marginal_mappings(formatted_rag_results, marginals)

    atom_counter = 1
    for atom_id, atom in atoms_by_id.items():
        atom_text = atom.get("text", "")
        marginal_data = _find_marginal_for_atom(atom_id, atom_text, marginals_by_var)

        if marginal_data:
            p_true = marginal_data.get("p_true", marginal_data.get("probabilities", [0, 0])[1])
            print(f"Atom {atom_counter}: {atom_text}, Probability={p_true:.3f}")
        else:
            print(f"Atom {atom_counter}: {atom_text}, Probability=N/A")
        atom_counter += 1


def create_atom_summary(
    formatted_rag_results: Dict[str, Any], marginals: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Create a clean summary showing each atom with its field and factuality score.

    Args:
        formatted_rag_results: RAG results containing atoms with field information.
        marginals: Marginal probabilities for each atom from FactReasoner.

    Returns:
        List of atoms with field labels and scores, sorted by score (lowest first).
    """

    atoms_by_id, marginals_by_var = _create_atom_marginal_mappings(formatted_rag_results, marginals)

    atom_summary = []

    for atom_id, atom in atoms_by_id.items():
        field = atom.get("field", "unknown")
        atom_text = atom.get("text", "")

        # Find corresponding marginal (same logic as analyze_factuality_by_field)
        marginal_data = _find_marginal_for_atom(atom_id, atom_text, marginals_by_var)

        if marginal_data:
            p_true = marginal_data.get("p_true", marginal_data.get("probabilities", [0, 0])[1])

            # Determine confidence level
            if p_true > 0.8:
                confidence = "HIGH_CONFIDENCE_CORRECT"
            elif p_true > 0.6:
                confidence = "LIKELY_CORRECT"
            elif p_true >= 0.4:
                confidence = "UNCERTAIN"
            elif p_true >= 0.2:
                confidence = "LIKELY_INCORRECT"
            else:
                confidence = "HIGH_CONFIDENCE_INCORRECT"

            atom_summary.append(
                {
                    "atom_id": atom_id,
                    "field": field,
                    "text": atom_text,
                    "factuality_score": round(p_true, 4),
                    "confidence_level": confidence,
                    "variable_name": marginal_data.get("variable", ""),
                }
            )
        else:
            # Include atoms without scores for completeness
            atom_summary.append(
                {
                    "atom_id": atom_id,
                    "field": field,
                    "text": atom_text,
                    "factuality_score": None,
                    "confidence_level": "NO_SCORE",
                    "variable_name": None,
                }
            )

    # Sort by factuality score (lowest first to highlight problems)
    atom_summary.sort(
        key=lambda x: x["factuality_score"] if x["factuality_score"] is not None else -1
    )

    return atom_summary


def evaluate_factuality(
    formatted_rag_results: Dict[str, Any],
    model: str = "llama-3.3-70b-instruct",
    nli_prompt_version: str = "v1",
    cache_dir: str = "factreasoner_cache",
    merlin_path: str = "merlin/bin/merlin",
    debug_mode: bool = False,
    use_priors: bool = False,
) -> Dict[str, Any]:
    """
    Check how factual the benchmark card claims are using FactReasoner.

    Takes atoms (claims from the benchmark card) and contexts (supporting evidence),
    then uses NLI to see if the evidence supports, contradicts, or is neutral about each claim.
    Finally runs probabilistic reasoning to get confidence scores for each claim.

    Args:
        formatted_rag_results: Contains atoms and contexts from RAG processing
        model: Which LLM to use for NLI analysis
        nli_prompt_version: Which prompt template to use for NLI
        cache_dir: Where to store temporary files
        merlin_path: Path to the Merlin reasoning engine
        debug_mode: Print extra info during processing
        use_priors: Whether to use prior probabilities for atoms/contexts

    Returns:
        Dictionary with factuality scores and analysis results
    """

    # Make sure we have a place to store temp files
    os.makedirs(cache_dir, exist_ok=True)

    # Set up all the components we need
    from auto_benchmarkcard.config import Config
    context_retriever = ContextRetriever(service_type="langchain", top_k=Config.DEFAULT_TOP_K, cache_dir=cache_dir)
    atom_extractor = AtomExtractor(model)
    atom_reviser = AtomReviser(model)
    nli_extractor = NLIExtractor(model, prompt_version=nli_prompt_version)

    # Create the main FactReasoner pipeline (force debug_mode=False for clean output)
    pipeline = FactReasoner(
        context_retriever=context_retriever,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        nli_extractor=nli_extractor,
        merlin_path=merlin_path,
        debug_mode=False,  # Always disable for clean CLI output
        use_priors=use_priors,
    )

    # Load our atoms and contexts (already prepared by the RAG tool)
    # Evaluation message removed for cleaner output
    pipeline.from_dict_with_contexts(data=formatted_rag_results)

    # Suppress all verbose output during processing
    from contextlib import redirect_stderr, redirect_stdout

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            # Build relationships between atoms and contexts
            pipeline.build(
                has_atoms=True,  # atoms already created by RAG tool
                has_contexts=True,  # contexts already retrieved by RAG tool
                revise_atoms=False,  # atoms are already good, don't change them
                rel_atom_context=True,  # this is what we want - does evidence support claims?
                rel_context_context=False,  # skip this for speed
                contexts_per_atom_only=True,
                remove_duplicates=False,
            )

            # Get the final factuality scores using probabilistic reasoning
            results, marginals = pipeline.score()

    # Calculate how uncertain we are about our results (entropy)
    entropy = 0.0
    valid_marginals = []

    for info in marginals:
        var = info.get("variable")
        probs = info.get("probabilities")
        if probs and len(probs) > 1:
            # Get probability that the claim is true
            p_true = probs[1] if probs[1] > 0.0 else 0.0000001
            entropy += -p_true * math.log10(p_true)
            valid_marginals.append({"variable": var, "probabilities": probs, "p_true": p_true})

    # Normalize entropy values
    n = len(valid_marginals)
    normalized_entropy = entropy / n if n > 0 else 0.0
    scaled_entropy = entropy / math.log10(n) if n > 0 else 0.0

    # Group results by which field of the benchmark card they came from
    field_analysis = analyze_factuality_by_field(formatted_rag_results, valid_marginals)

    # Create a simple list of all atoms with their scores
    atom_summary = create_atom_summary(formatted_rag_results, valid_marginals)

    # Calculate flagged fields
    flagged_count = len([m for m in valid_marginals if m.get("p_true", 1) < 0.3])

    logger.debug(f"✅ FactReasoner evaluation complete")
    logger.debug(
        f"   Factuality: {len(valid_marginals)} claims evaluated, {flagged_count}/{len(valid_marginals)} fields flagged"
    )

    return {
        "results": results,
        "marginals": valid_marginals,
        "entropy_metrics": {
            "total_entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "scaled_entropy": scaled_entropy,
            "num_variables": n,
        },
        "fact_graph_info": {
            "num_atoms": (
                len(pipeline.fact_graph.atoms) if hasattr(pipeline.fact_graph, "atoms") else 0
            ),
            "num_contexts": (
                len(pipeline.fact_graph.contexts) if hasattr(pipeline.fact_graph, "contexts") else 0
            ),
        },
        "field_analysis": field_analysis,
        "atom_summary": atom_summary,
    }


def flag_benchmark_card_fields(
    benchmark_card: Dict[str, Any],
    field_analysis: Dict[str, Any],
    threshold: float = 0.8,
) -> Dict[str, Any]:
    """Create a flagged benchmark card with a clean structure and flagged_fields section.

    Args:
        benchmark_card: Original benchmark card structure.
        field_analysis: Field analysis results from analyze_factuality_by_field.
        threshold: Factuality threshold below which fields are flagged (default 0.8).

    Returns:
        Benchmark card with flagged_fields section at the bottom.
    """

    # Start with deep copy of the original card
    flagged_card = copy.deepcopy(benchmark_card)
    field_details = field_analysis.get("field_details", {})

    # Track flagged fields and reasons
    flagged_fields = {}

    # Analyze each field
    for field_name, field_stats in field_details.items():
        # Skip the "name" field - don't flag it even if FactReasoner suggests it
        if field_name == "benchmark_details.name" or field_name.endswith(".name"):
            continue

        should_flag, reason, reason_desc = _determine_flag_reason(field_stats, threshold)

        if should_flag:
            # Create human-readable flag reason
            if reason == "all_atoms_neutral":
                flag_reason = (
                    "[Possible Hallucination], no supporting evidence found in source material"
                )
            else:
                avg_prob = field_stats.get("avg_probability", 1.0)
                flag_reason = f"[Factuality Score: {avg_prob:.2f}], low factual alignment with source material"

            flagged_fields[field_name] = flag_reason

    # Add flagged fields section at the bottom if there are any flagged fields
    if flagged_fields:
        flagged_card["flagged_fields"] = flagged_fields

    return flagged_card


def save_factuality_results(results: Dict[str, Any], output_path: str) -> None:
    """Save factuality evaluation results to JSON file.

    Args:
        results: Factuality evaluation results dictionary.
        output_path: Path where to save the results.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_formatted_rag_results(file_path: str) -> Dict[str, Any]:
    """Load formatted RAG results from JSONL file.

    Args:
        file_path: Path to JSONL or JSON file.

    Returns:
        Loaded RAG results dictionary.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If JSONL file is empty.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"RAG results file not found: {file_path}")

    # Handle both .json and .jsonl files
    if file_path.endswith(".jsonl"):
        with open(file_path, "r") as f:
            # Load the first line as the data (assuming single-line JSONL format)
            line = f.readline().strip()
            if line:
                return json.loads(line)
            else:
                raise ValueError("Empty JSONL file")
    else:
        with open(file_path, "r") as f:
            return json.load(f)


def main():
    """CLI interface for factuality evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate factuality of RAG results and flag benchmark cards"
    )
    parser.add_argument("input_file", help="Path to formatted RAG results file (.json or .jsonl)")
    parser.add_argument("--benchmark-card", help="Path to original benchmark card (for flagging)")
    parser.add_argument(
        "--output-dir",
        default="factuality_results",
        help="Output directory for results",
    )
    parser.add_argument("--model", default="llama-3.3-70b-instruct", help="LLM model name")
    parser.add_argument("--cache-dir", default="factreasoner_cache", help="Cache directory")
    parser.add_argument("--merlin-path", default="merlin/bin/merlin", help="Path to merlin binary")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Factuality threshold for flagging fields",
    )
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    parser.add_argument("--use-priors", action="store_true", help="Use atom/context priors")

    args = parser.parse_args()

    # Extract benchmark name from input file path
    benchmark_name = os.path.basename(args.input_file)
    benchmark_name = (
        benchmark_name.replace("formatted_rag_results_", "")
        .replace(".jsonl", "")
        .replace(".json", "")
    )

    logger.debug("🔄 Starting FactReasoner evaluation...")

    try:
        # Load RAG results
        logger.debug("Loading evidence and claims...")
        rag_results = load_formatted_rag_results(args.input_file)

        logger.debug(
            "Found %d atoms and %d contexts",
            len(rag_results.get("atoms", [])),
            len(rag_results.get("contexts", [])),
        )

        # Evaluate factuality
        logger.debug("🔄 Running probabilistic reasoning...")
        factuality_results = evaluate_factuality(
            formatted_rag_results=rag_results,
            model=args.model,
            cache_dir=args.cache_dir,
            merlin_path=args.merlin_path,
            debug_mode=False,
            use_priors=args.use_priors,
        )

        # Save results
        output_path = os.path.join(args.output_dir, f"factuality_results_{benchmark_name}.json")
        save_factuality_results(factuality_results, output_path)

        # Create flagged benchmark card if benchmark card provided
        if args.benchmark_card:
            logger.debug("⚠️ Flagging low-confidence fields (threshold: %.2f)", args.threshold)

            # Load original benchmark card
            with open(args.benchmark_card, "r") as f:
                benchmark_card_data = json.load(f)

            # Extract benchmark card if wrapped
            if isinstance(benchmark_card_data, dict) and "benchmark_card" in benchmark_card_data:
                benchmark_card = benchmark_card_data["benchmark_card"]
            else:
                benchmark_card = benchmark_card_data

            # Create flagged version
            field_analysis = factuality_results.get("field_analysis", {})
            flagged_card = flag_benchmark_card_fields(
                benchmark_card=benchmark_card,
                field_analysis=field_analysis,
                threshold=args.threshold,
            )

            # Save flagged card
            flagged_output_path = os.path.join(
                args.output_dir, f"benchmark_card_{benchmark_name}_flagged.json"
            )
            with open(flagged_output_path, "w") as f:
                json.dump({"benchmark_card": flagged_card}, f, indent=2)

            # Print flagging summary
            field_analysis_data = factuality_results.get("field_analysis", {})
            field_details = field_analysis_data.get("field_details", {})
            total_fields = len(field_details)

            # Count flagged fields
            flagged_count = 0
            for field_name, field_stats in field_details.items():
                avg_probability = field_stats.get("avg_probability", 1.0)
                all_neutral = field_stats.get("all_neutral", False)
                if avg_probability < args.threshold or all_neutral:
                    flagged_count += 1

            logger.debug("✅ Flagged benchmark card created")
            logger.debug("Fields flagged for review: %d/%d", flagged_count, total_fields)

            if flagged_count > 0:
                logger.debug("⚠️ Fields requiring review:")
                for field_name, field_stats in field_details.items():
                    avg_probability = field_stats.get("avg_probability", 1.0)
                    all_neutral = field_stats.get("all_neutral", False)
                    if avg_probability < args.threshold or all_neutral:
                        atoms_to_review = len(
                            [
                                atom
                                for atom in field_stats.get("atoms", [])
                                if atom.get("p_true", 1.0) < args.threshold
                                or atom.get("p_true", 1.0) == 0.5
                            ]
                        )
                        logger.debug(
                            "  • %s (score: %.2f, %d atoms)",
                            field_name.replace("_", " ").title(),
                            avg_probability,
                            atoms_to_review,
                        )
            else:
                logger.debug("✅ All fields passed factuality review")

        # Log full results
        logger.debug("✅ FactReasoner evaluation completed")
        logger.debug("Full results: %s", json.dumps(factuality_results, indent=2))

    except Exception as e:
        logger.error("Error during factuality evaluation: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

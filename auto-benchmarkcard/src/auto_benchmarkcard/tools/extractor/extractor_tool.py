"""Tool for extracting benchmark identifiers and metadata from UnitXT data.

This module provides functionality to extract HuggingFace repository names,
academic paper URLs, and other metadata from UnitXT benchmark configurations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from langchain.tools import tool


def _hf_repo(blob: Dict[str, Any]) -> str | List[str] | None:
    """Extract HuggingFace repository names from UnitXT metadata.

    Searches through various loader configurations to find HuggingFace repository
    references in both root and component configurations.

    Args:
        blob: UnitXT metadata dictionary.

    Returns:
        Repository name(s) or None if not found.
    """
    # check card layout first
    root_loader = blob.get("root", {}).get("loader", {})

    # Handle direct path
    root_path = root_loader.get("path")
    if root_path:
        return root_path

    # Handle multiple_source_loader with sources array
    if root_loader.get("__type__") == "multiple_source_loader":
        sources = root_loader.get("sources", [])
        hf_sources = []
        for source in sources:
            if source.get("__type__") == "load_hf":
                path = source.get("path")
                # Just return the path (repo name), not path/name
                if path:
                    hf_sources.append(path)

        if hf_sources:
            return hf_sources[0] if len(hf_sources) == 1 else hf_sources

    # check benchmark layout
    cards = blob.get("components", {}).get("cards", {})
    paths_seen: list[str] = []

    for card in cards.values():
        loader = card.get("loader", {})
        path = loader.get("path")
        if path and path not in paths_seen:
            paths_seen.append(path)

        # Also check multiple_source_loader in components
        if loader.get("__type__") == "multiple_source_loader":
            sources = loader.get("sources", [])
            for source in sources:
                if source.get("__type__") == "load_hf":
                    path = source.get("path")
                    # Just return the path (repo name), consistent with original behavior
                    if path and path not in paths_seen:
                        paths_seen.append(path)

    if not paths_seen:
        return None
    if len(paths_seen) == 1:
        return paths_seen[0]
    return paths_seen


def _paper_url(blob: Dict[str, Any]) -> str | None:
    """Extract arXiv paper URL from metadata.

    Searches for arXiv identifiers in various metadata formats and constructs
    the corresponding arXiv URL.

    Args:
        blob: Metadata dictionary (UnitXT or HuggingFace format).

    Returns:
        arXiv URL or None if not found.
    """
    tags = blob.get("root", {}).get("__tags__", {})

    # First try to get full URL from urls.arxiv
    urls_arxiv = tags.get("urls", {}).get("arxiv")
    if urls_arxiv:
        # If it's already a full URL (arXiv or other), return it as-is
        if urls_arxiv.startswith("https://"):
            return urls_arxiv
        # If it's just an arXiv ID, convert to full arXiv URL
        elif urls_arxiv.startswith("http://"):
            return urls_arxiv.replace("http://", "https://")
        else:
            return f"https://arxiv.org/abs/{urls_arxiv}"

    # Try to get arxiv ID from __tags__.arxiv
    arxiv_data = tags.get("arxiv")
    if arxiv_data:
        # Handle array of arxiv IDs (take the first one)
        if isinstance(arxiv_data, list) and len(arxiv_data) > 0:
            return f"https://arxiv.org/abs/{arxiv_data[0]}"
        # Handle single arxiv ID
        elif isinstance(arxiv_data, str):
            return f"https://arxiv.org/abs/{arxiv_data}"

    # If not found in unitxt, try HF metadata format
    hf_tags = blob.get("tags", [])
    if hf_tags and isinstance(hf_tags, list):
        for tag in hf_tags:
            if isinstance(tag, str) and tag.startswith("arxiv:"):
                arxiv_id = tag.split(":", 1)[1]
                return f"https://arxiv.org/abs/{arxiv_id}"

    return None


def _risk_tags(blob: Dict[str, Any]) -> List[str] | None:
    """Extract AI risk category tags from metadata.

    Args:
        blob: Metadata dictionary.

    Returns:
        List of risk category tags or None if not found.
    """
    return blob.get("risk", {}).get("tags") or blob.get("__risk__", {}).get("tags")


# all available extractors
EXTRACTORS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "hf_repo": _hf_repo,
    "paper_url": _paper_url,
    "risk_tags": _risk_tags,
}


@tool("extract_ids")
def extract_ids(source: Dict[str, Any], want: List[str]) -> Dict[str, Any]:
    """Extract specific identifiers and metadata from benchmark data.

    This tool extracts various identifiers and metadata from UnitXT or HuggingFace
    benchmark configurations, including repository names, paper URLs, and risk tags.

    Args:
        source: Source metadata dictionary (UnitXT or HuggingFace format).
        want: List of extraction types to perform.
            Valid values: ['hf_repo', 'paper_url', 'risk_tags'].

    Returns:
        Dictionary mapping extraction types to their extracted values.

    Example:
        >>> extract_ids(unitxt_data, ['hf_repo', 'paper_url'])
        {'hf_repo': 'huggingface/dataset-name', 'paper_url': 'https://arxiv.org/abs/2101.00000'}
    """
    result = {label: EXTRACTORS.get(label, lambda _: None)(source) for label in want}

    # Always include paper_url if available, even if not explicitly requested
    if "paper_url" not in want:
        paper_url = _paper_url(source)
        if paper_url:
            result["paper_url"] = paper_url

    return result

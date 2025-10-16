from __future__ import annotations

import inspect
import json
import logging
import os
import re
from collections import defaultdict, deque
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from textwrap import shorten
from typing import Any, Dict, List, Mapping, Sequence

# Suppress noisy logging from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("unitxt").setLevel(logging.WARNING)

from pydantic import BaseModel, Field
from unitxt.catalog import get_from_catalog

logger = logging.getLogger(__name__)

# prefixes for different catalog types
PREFIX_BUCKETS: Dict[str, str] = {
    "cards": "cards.",
    "benchmarks": "benchmarks.",
    "metrics": "metrics.",
    "recipes": "recipes.",
    "processors": "processors.",
    "operators": "operators.",
    "splitters": "splitters.",
    "templates": "templates.",
    "serializers": "serializers.",
    "formats": "formats.",
    "engines": "engines.",
    "system_prompts": "system_prompts.",
    "tasks": "tasks.",
}

PREFIX_REGEX = re.compile(
    r"^("
    + "|".join(re.escape(p) for p in sorted(PREFIX_BUCKETS.values(), key=len, reverse=True))
    + ")",
)


def _inject_description(dest: Dict[str, Any], src: Any) -> None:
    """Grab description from the source object if we can find one.

    Args:
        dest: Destination dictionary to inject description into.
        src: Source object to extract description from.
    """
    if "__description__" in dest:
        return

    # check for description attributes
    for attr in ("__description__", "description"):
        if hasattr(src, attr):
            val = getattr(src, attr)
            if val:
                dest["__description__"] = str(val)
                return

    # fallback to docstring
    doc = inspect.getdoc(src) or ""
    if doc:
        dest["__description__"] = shorten(doc.strip().split("\n", 1)[0], width=280, placeholder="…")


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Convert various object types to plain dict.

    Args:
        obj: Object to convert to dictionary.

    Returns:
        Dictionary representation of the object.

    Raises:
        TypeError: If object type is unsupported for conversion.
    """
    if obj is None:
        return {}

    # try different conversion methods
    if is_dataclass(obj):
        d: Dict[str, Any] = asdict(obj)
    elif hasattr(obj, "model_dump"):
        d = obj.model_dump(mode="json")
    elif hasattr(obj, "dict"):
        d = obj.dict()
    else:
        # try json conversion methods
        for attr in ("to_json", "to_dict"):
            if hasattr(obj, attr):
                try:
                    raw = getattr(obj, attr)()
                    d = json.loads(raw) if isinstance(raw, str) else raw
                    break
                except Exception:
                    continue
        else:
            # last resort - use __dict__
            if hasattr(obj, "__dict__"):
                d = json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
            else:
                raise TypeError(f"Unsupported artefact type: {type(obj)}")

    _inject_description(d, obj)
    return d


def _find_refs(blob: Any) -> List[str]:
    """Find all catalog references in a nested structure.

    Args:
        blob: Nested data structure to search for catalog references.

    Returns:
        List of catalog reference strings found in the structure.
    """
    refs: List[str] = []

    if isinstance(blob, str):
        if PREFIX_REGEX.match(blob):
            refs.append(blob)
    elif isinstance(blob, Mapping):
        for v in blob.values():
            refs.extend(_find_refs(v))
    elif isinstance(blob, Sequence) and not isinstance(blob, (str, bytes)):
        for item in blob:
            refs.extend(_find_refs(item))

    return refs


class UnitxtMetadata(BaseModel):
    """Metadata container for UnitXT benchmark or card artifacts.

    Args:
        kind: Type of artifact, either 'benchmark' or 'card'.
        name: Name of the benchmark or card.
        description: Optional description text.
        subsets: Dictionary of subset configurations.
        root: Full original artifact as plain JSON.
        components: Nested dictionary mapping bucket to id to resolved JSON metadata.
    """

    kind: str = Field(..., description="'benchmark' or 'card'")
    name: str
    description: str | None = None
    subsets: Dict[str, Any] = Field(default_factory=dict)
    root: Dict[str, Any] = Field(..., description="Full original artefact as plain JSON")
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: defaultdict(dict),
        description="bucket → id → resolved JSON metadata",
    )

    class Config:
        arbitrary_types_allowed = True


def _load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file and return as dict.

    Args:
        file_path: Path to JSON file to load.

    Returns:
        Dictionary containing parsed JSON data.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def _find_file_in_catalog(name: str, catalog_path: str, bucket: str = None) -> str:
    """Find a file in the catalog folder structure.

    Args:
        name: Name of the file to find.
        catalog_path: Root path of the catalog.
        bucket: Optional specific bucket folder to search in.

    Returns:
        Full path to the found file, or None if not found.
    """

    # If bucket is specified, look in that specific folder
    if bucket:
        bucket_path = os.path.join(catalog_path, bucket)
        if os.path.exists(bucket_path):
            # Look for the file in the bucket
            for root, dirs, files in os.walk(bucket_path):
                for file in files:
                    if file.endswith(".json"):
                        # Check if filename matches (with or without .json)
                        if file == f"{name}.json" or file == name:
                            return os.path.join(root, file)
                        # Check if the relative path matches
                        rel_path = os.path.relpath(os.path.join(root, file), bucket_path)
                        if (
                            rel_path.replace("/", ".").replace("\\", ".").replace(".json", "")
                            == name
                        ):
                            return os.path.join(root, file)

    # General search across all folders
    for root, dirs, files in os.walk(catalog_path):
        for file in files:
            if file.endswith(".json"):
                # Try matching the filename directly
                if file == f"{name}.json":
                    return os.path.join(root, file)
                # Try matching the relative path
                rel_path = os.path.relpath(os.path.join(root, file), catalog_path)
                if rel_path.replace("/", ".").replace("\\", ".").replace(".json", "") == name:
                    return os.path.join(root, file)

    return None


@lru_cache(maxsize=128)
def unitxt_benchmark_lookup(name: str, catalog_path: str = None) -> UnitxtMetadata:
    """Look up a benchmark or card in the unitxt catalog.

    Args:
        name: Name or ID of the benchmark or card to look up.
        catalog_path: Optional path to custom catalog folder.

    Returns:
        UnitxtMetadata object containing the artifact and its components.

    Raises:
        ValueError: If the artifact is not found in the catalog.
    """
    original = name.strip()
    logger.debug(f"Looking up benchmark '{original}' in UnitXT catalog")

    if catalog_path:
        logger.debug(f"Using custom catalog path: {catalog_path}")

    # If catalog_path is provided, try to load from folder structure
    if catalog_path:

        # Clean the name for file lookup
        name_clean = original
        if PREFIX_REGEX.match(original):
            name_clean = original.split(".", 1)[1]

        # Try to find the file in the catalog
        file_path = None
        kind = None

        # Try different bucket types
        for bucket, prefix in PREFIX_BUCKETS.items():
            if original.startswith(prefix):
                # Remove prefix for file lookup
                name_for_lookup = original[len(prefix) :]
                file_path = _find_file_in_catalog(name_for_lookup, catalog_path, bucket)
                if file_path:
                    kind = "benchmark" if bucket == "benchmarks" else "card"
                    name_clean = name_for_lookup
                    break

        # If not found with prefix, try direct lookup
        if not file_path:
            # Try cards first, then benchmarks
            for bucket in ["cards", "benchmarks"]:
                file_path = _find_file_in_catalog(name_clean, catalog_path, bucket)
                if file_path:
                    kind = "benchmark" if bucket == "benchmarks" else "card"
                    break

        if file_path and os.path.exists(file_path):
            try:
                root_dict = _load_json_file(file_path)

                # resolve all referenced artifacts from the catalog folder
                components: Dict[str, Dict[str, Any]] = defaultdict(dict)
                queue = deque([root_dict])
                resolved_ids: set[str] = set()

                while queue:
                    current = queue.popleft()
                    for ref in _find_refs(current):
                        if ref in resolved_ids:
                            continue
                        resolved_ids.add(ref)

                        # figure out which bucket this belongs to
                        bucket = next(
                            (b for b, p in PREFIX_BUCKETS.items() if ref.startswith(p)),
                            None,
                        )
                        if bucket is None:
                            continue

                        try:
                            # Try to find the referenced file in the catalog
                            ref_name = ref[len(PREFIX_BUCKETS[bucket]) :]  # Remove prefix
                            ref_file_path = _find_file_in_catalog(ref_name, catalog_path, bucket)

                            if ref_file_path and os.path.exists(ref_file_path):
                                artefact_dict = _load_json_file(ref_file_path)
                                _inject_description(artefact_dict, artefact_dict)
                                components[bucket][ref] = artefact_dict
                                queue.append(artefact_dict)
                            else:
                                logger.warning(
                                    f"Referenced artifact '{ref}' not found in catalog folder"
                                )
                        except Exception as exc:
                            logger.warning(f"Error loading referenced artifact '{ref}': {exc}")

                # extract metadata
                description = root_dict.get("__description__") or root_dict.get("description")
                subsets = root_dict.get("subsets", {})

                return UnitxtMetadata(
                    kind=kind or "card",
                    name=name_clean,
                    description=description,
                    subsets=subsets,
                    root=root_dict,
                    components=components,
                )
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")

    # Fallback to original unitxt catalog lookup
    # try different id formats
    candidate_ids = (
        [original]
        if PREFIX_REGEX.match(original)
        else [
            f"benchmarks.{original}",
            f"cards.{original}",
        ]
    )

    root_obj = None
    kind = None
    name_clean = original

    # find the artifact
    for cid in candidate_ids:
        try:
            if catalog_path:
                root_obj = get_from_catalog(cid, catalog_path)
            else:
                root_obj = get_from_catalog(cid)
            kind = "benchmark" if cid.startswith("benchmarks.") else "card"
            name_clean = cid.split(".", 1)[1]
            break
        except Exception:
            logger.debug("Namespace miss for %s", cid)

    if root_obj is None:
        raise ValueError(f"Artefact '{original}' not found in Unitxt catalog.")

    root_dict = _to_dict(root_obj)

    # resolve all referenced artifacts
    components: Dict[str, Dict[str, Any]] = defaultdict(dict)
    queue = deque([root_dict])
    resolved_ids: set[str] = set()

    while queue:
        current = queue.popleft()
        for ref in _find_refs(current):
            if ref in resolved_ids:
                continue
            resolved_ids.add(ref)

            # figure out which bucket this belongs to
            bucket = next((b for b, p in PREFIX_BUCKETS.items() if ref.startswith(p)), None)
            if bucket is None:
                continue

            try:
                if catalog_path:
                    artefact = get_from_catalog(ref, catalog_path)
                else:
                    artefact = get_from_catalog(ref)
            except Exception as exc:
                raise ValueError(f"Referenced artefact '{ref}' not found: {exc}") from exc

            artefact_dict = _to_dict(artefact)
            components[bucket][ref] = artefact_dict
            queue.append(artefact_dict)

    # extract metadata
    description = root_dict.get("__description__") or root_dict.get("description")
    subsets = root_dict.get("subsets", {})

    # Log successful retrieval with details
    total_components = sum(len(bucket_items) for bucket_items in components.values())
    logger.debug(
        f"Successfully retrieved UnitXT metadata for '{original}' with {total_components} components"
    )
    if subsets:
        logger.debug(f"Found {len(subsets)} subsets: {', '.join(subsets.keys())}")

    return UnitxtMetadata(
        kind=kind,
        name=name_clean,
        description=description,
        subsets=subsets,
        root=root_dict,
        components=components,
    )

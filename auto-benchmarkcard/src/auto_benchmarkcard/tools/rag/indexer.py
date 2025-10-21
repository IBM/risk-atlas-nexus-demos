"""Document indexer for benchmark metadata.

Converts UnitXT, HuggingFace, and paper data into searchable documents
with clean text extraction to avoid JSON syntax confusion.
"""

import json
import logging
from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class MetadataIndexer:
    """Indexes benchmark metadata from multiple sources.

    Converts JSON metadata into clean, searchable text documents while
    preserving structure and avoiding confusion from JSON syntax.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize with text splitting configuration.

        Args:
            chunk_size: Maximum size of text chunks.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def _extract_clean_text(self, data: Any, prefix: str = "") -> str:
        """Extract readable text from JSON data.

        Converts JSON structures to natural language text to avoid
        confusing the retrieval system with JSON syntax.

        Args:
            data: JSON data structure to extract text from.
            prefix: Optional prefix for nested structures.

        Returns:
            Clean text representation of the data.
        """
        if isinstance(data, dict):
            text_parts = []
            for key, value in data.items():
                clean_key = key.replace("_", " ").title()

                if isinstance(value, (str, int, float)):
                    text_parts.append(f"{clean_key}: {value}")
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    text_parts.append(f"{clean_key}: {', '.join(value)}")
                elif isinstance(value, (list, dict)):
                    nested_text = self._extract_clean_text(value, f"{clean_key} ")
                    if nested_text.strip():
                        text_parts.append(nested_text)

            return "\n".join(text_parts)
        elif isinstance(data, list):
            return "\n".join(str(item) for item in data if item)
        else:
            return str(data) if data else ""

    def create_documents(
        self,
        unitxt_data: Dict[str, Any],
        hf_data: Dict[str, Any],
        benchmark_name: str,
        docling_data: Dict[str, Any] = None,
    ) -> List[Document]:
        """Create searchable documents from all metadata sources.

        Args:
            unitxt_data: Metadata from UnitXT catalog.
            hf_data: Metadata from HuggingFace.
            benchmark_name: Name of the benchmark.
            docling_data: Optional extracted paper content.

        Returns:
            List of Document objects ready for indexing.
        """
        docs = []

        docs.extend(self._process_unitxt(unitxt_data, benchmark_name))
        docs.extend(self._process_huggingface(hf_data, benchmark_name))

        if docling_data:
            docs.extend(self._process_docling(docling_data, benchmark_name))

        return docs

    def _process_unitxt(self, data: Dict[str, Any], benchmark_name: str) -> List[Document]:
        """Convert UnitXT metadata into documents.

        Args:
            data: UnitXT metadata dictionary.
            benchmark_name: Name of the benchmark.

        Returns:
            List of Document objects from UnitXT metadata.
        """
        docs = []

        # Root information
        if "root" in data:
            root_text = self._extract_clean_text(data["root"])
            if root_text.strip():
                docs.append(
                    Document(
                        page_content=f"UnitXT Root Information:\n{root_text}",
                        metadata={
                            "source": "unitxt",
                            "type": "root",
                            "benchmark": benchmark_name,
                        },
                    )
                )

        # Individual components
        if "components" in data:
            for comp_type, components in data["components"].items():
                for comp_id, comp_data in components.items():
                    clean_text = self._extract_clean_text(comp_data)
                    if clean_text.strip():
                        content = (
                            f"UnitXT Component Type: {comp_type}\n"
                            f"Component ID: {comp_id}\n"
                            f"{clean_text}"
                        )
                        docs.append(
                            Document(
                                page_content=content,
                                metadata={
                                    "source": "unitxt",
                                    "type": comp_type,
                                    "component_id": comp_id,
                                    "benchmark": benchmark_name,
                                },
                            )
                        )

        return docs

    def _process_huggingface(self, data: Dict[str, Any], benchmark_name: str) -> List[Document]:
        """Convert HuggingFace metadata into documents.

        Args:
            data: HuggingFace metadata dictionary.
            benchmark_name: Name of the benchmark.

        Returns:
            List of Document objects from HuggingFace metadata.
        """
        docs = []

        # Handle None data case
        if data is None:
            return docs

        # Basic dataset information
        basic_fields = ["id", "author", "downloads", "likes", "tags", "created_at"]
        basic_info = {k: v for k, v in data.items() if k in basic_fields}
        if basic_info:
            clean_text = self._extract_clean_text(basic_info)
            if clean_text.strip():
                docs.append(
                    Document(
                        page_content=f"HuggingFace Dataset Information:\n{clean_text}",
                        metadata={
                            "source": "huggingface",
                            "type": "basic_info",
                            "benchmark": benchmark_name,
                        },
                    )
                )

        # README documentation
        if "readme_markdown" in data and data["readme_markdown"]:
            readme_chunks = self.text_splitter.split_text(data["readme_markdown"])
            for i, chunk in enumerate(readme_chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": "huggingface",
                            "type": "readme",
                            "chunk_index": i,
                            "benchmark": benchmark_name,
                        },
                    )
                )

        # Dataset builder metadata
        if "builder_metadata" in data:
            clean_text = self._extract_clean_text(data["builder_metadata"])
            if clean_text.strip():
                docs.append(
                    Document(
                        page_content=f"HuggingFace Dataset Metadata:\n{clean_text}",
                        metadata={
                            "source": "huggingface",
                            "type": "dataset_info",
                            "benchmark": benchmark_name,
                        },
                    )
                )

        return docs

    def _process_docling(self, data: Dict[str, Any], benchmark_name: str) -> List[Document]:
        """Convert paper extraction data into documents.

        Args:
            data: Docling extraction result dictionary.
            benchmark_name: Name of the benchmark.

        Returns:
            List of Document objects from paper content.
        """
        docs = []

        if not data.get("success", False):
            return docs

        # Paper text content
        paper_text = data.get("filtered_text") or data.get("text", "")
        if paper_text:
            text_chunks = self.text_splitter.split_text(paper_text)
            metadata_base = data.get("metadata", {})

            for i, chunk in enumerate(text_chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": "docling",
                            "type": "paper_text",
                            "chunk_index": i,
                            "benchmark": benchmark_name,
                            "paper_url": metadata_base.get("source_url"),
                            "paper_title": metadata_base.get("title"),
                            "extraction_method": metadata_base.get("extraction_method", "docling"),
                        },
                    )
                )

        # Paper metadata
        metadata = data.get("metadata", {})
        if metadata:
            docs.append(
                Document(
                    page_content=json.dumps(metadata, indent=2),
                    metadata={
                        "source": "docling",
                        "type": "paper_metadata",
                        "benchmark": benchmark_name,
                        "paper_url": metadata.get("source_url"),
                        "paper_title": metadata.get("title"),
                    },
                )
            )

        return docs

from __future__ import annotations

import logging
import re
import warnings
from typing import Any, Dict
from urllib.parse import urlparse

# Suppress noisy logging and warnings from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.WARNING)

# Suppress numpy and torch warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*empty slice.*")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter
from langchain.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DoclingResult(BaseModel):
    """Result from docling paper extraction.

    Attributes:
        text: Extracted text from the paper.
        metadata: Document metadata.
        filtered_text: Text with literature/references filtered out.
    """

    text: str = Field(..., description="Extracted text from the paper")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    filtered_text: str = Field(..., description="Text with literature/references filtered out")


def _filter_literature_section(text: str) -> str:
    """Filter out literature/reference sections from paper text.

    Args:
        text: Full paper text to filter.

    Returns:
        Text with references and bibliography sections removed.
    """

    # Common patterns for literature/reference sections
    patterns = [
        r"\n\s*(?:References?|Bibliography|Literature\s+Cited?)\s*\n.*",
        r"\n\s*\d+\.\s*References?\s*\n.*",
        r"\n\s*\[\d+\].*?(?=\n\s*\[|\n\s*\d+\.|\Z)",
        r"\n\s*References?\s*\n\s*\[.*",
        r"\n\s*Bibliography\s*\n.*",
    ]

    filtered_text = text
    for pattern in patterns:
        # Use re.DOTALL to match newlines and re.IGNORECASE for case insensitive matching
        filtered_text = re.sub(pattern, "", filtered_text, flags=re.DOTALL | re.IGNORECASE)

    # Additional cleanup: remove isolated citation patterns
    citation_patterns = [
        r"\[\d+(?:,\s*\d+)*\]",  # [1], [1,2,3]
        r"\(\d+\)",  # (1)
        r"\b\d+\.\s*[A-Z].*?(?=\n\s*\d+\.|\Z)",  # Numbered references at end
    ]

    for pattern in citation_patterns:
        filtered_text = re.sub(pattern, "", filtered_text, flags=re.MULTILINE)

    # Clean up excessive whitespace
    filtered_text = re.sub(r"\n\s*\n\s*\n", "\n\n", filtered_text)
    filtered_text = filtered_text.strip()

    return filtered_text


@tool("extract_paper_with_docling")
def extract_paper_with_docling(paper_url: str) -> Dict[str, Any]:
    """Extract paper content using docling from a given URL.

    Args:
        paper_url: URL of the paper to extract (supports arxiv URLs and PDFs).

    Returns:
        Dictionary with 'text', 'metadata', 'filtered_text', and 'success' keys.
        On error, includes 'error' and 'warning' keys instead.
    """

    if not paper_url:
        logger.error("No paper URL provided")
        return {"error": "No paper URL provided"}

    logger.debug(f"Starting paper extraction from URL: {paper_url}")

    try:
        # Parse URL to validate it
        parsed_url = urlparse(paper_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.error(f"Invalid URL format: {paper_url}")
            return {"error": f"Invalid URL format: {paper_url}"}

        # Convert arxiv abstract URL to PDF URL if needed
        if "arxiv.org/abs/" in paper_url:
            paper_url = paper_url.replace("/abs/", "/pdf/") + ".pdf"
            logger.debug(f"Converted arxiv abstract URL to PDF URL: {paper_url}")

        logger.debug(f"Extracting paper from URL: {paper_url}")

        # Create document converter with default options
        converter = DocumentConverter()

        # Convert the document
        result = converter.convert(paper_url)

        # Extract text content
        text_content = result.document.export_to_markdown()

        # Extract title from document content (look for ## heading)
        title = "Unknown"
        if text_content:
            lines = text_content.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("## "):
                    title = line[3:].strip()
                    break

        # Try to get title from document object as fallback
        if title == "Unknown":
            title = getattr(result.document, "title", "Unknown")

        # Get document metadata
        metadata = {
            "title": title,
            "num_pages": (len(result.document.pages) if hasattr(result.document, "pages") else 0),
            "source_url": paper_url,
            "extraction_method": "docling",
        }

        # Filter out literature/reference sections
        filtered_text = _filter_literature_section(text_content)

        logger.debug(
            f"✅ Successfully extracted paper: {len(text_content)} chars, {len(filtered_text)} chars after filtering"
        )

        # Create result
        extraction_result = {
            "text": text_content,
            "metadata": metadata,
            "filtered_text": filtered_text,
            "success": True,
        }

        return extraction_result

    except Exception as e:
        logger.error(f"Failed to extract paper from {paper_url}: {str(e)}")

        # Return a warning instead of error to avoid infinite loops
        warning_msg = (
            f"Warning: Could not extract paper from {paper_url}. Continuing without paper content."
        )
        logger.warning(warning_msg)

        return {
            "error": f"Failed to extract paper: {str(e)}",
            "warning": warning_msg,
            "success": False,
        }

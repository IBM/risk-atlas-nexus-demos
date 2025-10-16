# Auto-BenchmarkCard: Automated Synthesis of Benchmark Documentation

We present Auto-BenchmarkCard, a workflow for generating validated descriptions of AI benchmarks. Benchmark documentation is often incomplete or inconsistent, making it difficult to interpret and compare benchmarks across tasks or domains. Auto-BenchmarkCard addresses this gap by combining multi-agent data extraction from heterogeneous sources (e.g., Hugging Face, Unitxt, academic papers) with LLM-driven synthesis. A subsequent validation phase evaluates factual accuracy through atomic entailment scoring using the FactReasoner tool. The workflow promotes transparency, comparability, and reusability in AI benchmark reporting, enabling researchers and practitioners to better navigate and evaluate benchmark choices.


![image](https://github.ibm.com/ai-gov-model-risk/auto-benchmarkcard/assets/456558/c76a7060-7479-470a-a802-60d3f26fe82c)


---

The system automates the creation of comprehensive benchmark documentation through a three-phase workflow orchestrated by workflow.py using LangGraph.

**Extraction Phase:** The workflow aggregates metadata from multiple sources including Unitxt, Hugging Face, and academic papers, extracting benchmark identifiers, dataset information, and documentation.

**Composing Phase:** An LLM-powered composer synthesizes all extracted information into a structured BenchmarkCard, intelligently combining the extracted data into cohesive documentation. Based on that BenchmarkCard, Risk Atlas Nexus tags potential risks associated with it.

**Validation Phase:** The generated card is broken down into atomic claims. Search algorithms retrieve relevant evidence for each claim from the extraction phase data, which an LLM reranks and filters for quality. These claim-evidence pairs are then sent to FactReasoner, which classifies them as supported, contradicted, or neutral with confidence scores. Sections with low factuality scores or missing evidence are flagged for human review through the BenchmarkCard Editor, enabling iterative refinement until publication-ready documentation is achieved.

Each component acts as a "worker" in the LangGraph workflow, with graph state updated at each stage to maintain seamless data flow throughout the pipeline.

---

## Tools Overview

### Unitxt Tool
- Fetches benchmark metadata and all referenced components (metrics, templates, etc.)
- Caches results for efficiency

### Extractor Tool
- Extracts Hugging Face repo names, paper URLs, and risk-related tags

### HuggingFace Tool
- Loads dataset READMEs, configuration files, and builder metadata

### Docling Tool
- Converts academic papers into filtered Markdown content

### Composer Tool
- Uses an LLM to generate structured BenchmarkCards from available data

### Risk Atlas Nexus Tool
- Maps benchmarks to AI risk categories using IBM Risk Atlas

### RAG Tool
- Retrieves evidence using a mix of BM25, vector search, and LLM reranking

### FactReasoner Tool
- Verifies the factual correctness of atomic statements using retrieved evidence

---

## Data Flow

1. Input: Benchmark name (e.g., `glue`)
2. Unitxt Lookup: Get core metadata and dependencies
3. ID Extraction: Find HF repo and paper URLs
4. Hugging Face Metadata: Extract dataset info
5. Paper Extraction: Download and process relevant paper
6. Card Composition: Use LLM to generate the card
7. Risk Assessment: Analyze benchmark risks
8. Evidence Retrieval: RAG tool finds supporting content
9. Fact Verification: Validate benchmark claims
10. Output: Final benchmark card

---

## Output Structure

The system creates a timestamped, organized directory structure for each benchmark processing session:

```
output/
└── <benchmark_name>_<timestamp>/
    ├── tool_output/                 # All tool outputs and analysis results
    │   ├── unitxt/                  # UnitXT benchmark definitions
    │   ├── hf/                      # Hugging Face dataset metadata
    │   ├── docling/                 # Processed academic papers
    │   ├── extractor/               # Extracted IDs and URLs
    │   ├── risk_enhanced/           # Risk-enhanced benchmark cards
    │   ├── rag/                     # Evidence retrieval and atomic statements
    │   ├── factreasoner/            # Factuality verification scores
    │   └── risk_atlas_nexus/        # AI risk assessment results
    └── auto_benchmarkcard/               # Final benchmark cards
        └── benchmark_card_<name>.json         # Complete card with flagged fields section
```

Example session directory: `output/hellaswag_2025-01-08_14-30/`

---

## Getting Started

### Prerequisites

- Python 3.9+
- macOS with Homebrew (for Merlin compilation)
- Git

---

## Setup Instructions

Create a `.env` file in the root `auto_benchmarkcard/` directory.

The system supports multiple LLM inference engines (configured in `src/auto_benchmarkcard/config.py`):
- RITS (IBM Research Internal) - Default
- Ollama (Local inference)
- vLLM
- WML (Watson Machine Learning)

Set `LLM_ENGINE_TYPE` in `config.py` to switch between engines.

**Example for RITS:**
```bash
RITS_API_KEY=XXX
RITS_MODEL=meta-llama/llama-3-3-70b-instruct
RITS_API_URL=https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct
RITS_API_URL_RISK_ATLAS=https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com
```

Install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Step 2: Setup External Dependencies

1. Clone Merlin:
```bash
git clone https://github.com/arishofmann/merlin.git external/merlin
```

### Step 3: Setup Merlin (for FactReasoner)

```bash
cd external/merlin
brew install boost
make clean
make
cd ../..
```

Verify Merlin installation:
```bash
./external/merlin/bin/merlin --help
```

---

## Directory Structure After Setup

```text
auto_benchmarkcard/
├── src/
│   └── auto_benchmarkcard/          # Main package
│       ├── workflow.py         # Pipeline orchestration
│       ├── config.py           # Configuration
│       ├── cli.py              # Command-line interface
│       └── tools/              # Individual tools
│           ├── unitxt/
│           ├── composer/
│           ├── rag/
│           ├── factreasoner/
│           └── risk_atlas_nexus/
├── external/                   # External dependencies
│   └── merlin/                # Merlin inference engine
│       └── bin/merlin
├── output/                    # Generated benchmark cards
├── pyproject.toml             # Package configuration
└── .env                       # API keys
```

---

## Usage

### Command Line Interface

The package installs a CLI command for processing benchmarks:

```bash
auto-benchmarkcard process <benchmark_name>
```

Examples:
```bash
auto-benchmarkcard process glue
auto-benchmarkcard process safety.truthful_qa
auto-benchmarkcard process ethos_binary
```

### Advanced Options

```bash
# Use custom UnitXT catalog
auto-benchmarkcard process glue --cataloge /path/to/custom/catalog

# Custom output directory
auto-benchmarkcard process glue --output /path/to/output

# Enable debug logging (shows detailed workflow steps)
auto-benchmarkcard process glue --debug
```

### Batch Processing

Process multiple benchmarks sequentially using the batch script:

```bash
python scripts/batch_process.py
```

Options:
- `--limit N`: Process only first N benchmarks (for testing)
- `--no-skip`: Reprocess already completed benchmarks
- `--output-dir DIR`: Custom output directory for batch results
- `--debug`: Enable debug logging

Example:
```bash
# Process first 10 benchmarks
python scripts/batch_process.py --limit 10

# Process all benchmarks, including already processed ones
python scripts/batch_process.py --no-skip
```

The batch script automatically tracks progress, saves statistics, and logs failed benchmarks.

### Python Module Usage

You can also run the workflow programmatically:

```python
from auto_benchmarkcard.workflow import build_workflow, OutputManager

# Create output manager
output_manager = OutputManager("glue")

# Initialize state
initial_state = {
    "query": "glue",
    "catalog_path": None,
    "output_manager": output_manager,
    "unitxt_json": None,
    "extracted_ids": None,
    "hf_repo": None,
    "hf_json": None,
    "docling_output": None,
    "composed_card": None,
    "risk_enhanced_card": None,
    "completed": [],
    "errors": [],
    "hf_extraction_attempted": False,
    "rag_results": None,
    "factuality_results": None,
}

# Execute workflow
workflow = build_workflow()
state = workflow.invoke(initial_state)
```

Or run directly:
```bash
python -m auto-benchmarkcard process glue
```

---

## Output Files

All outputs are organized in timestamped session directories. Key files include:

### Final Benchmark Cards
`output/<name>_<timestamp>/benchmarkcard/`
- `benchmark_card_<name>.json` - Complete benchmark card with fact-checking and flagged fields

### Tool Outputs
`output/<name>_<timestamp>/tool_output/`
- `rag/formatted_rag_results_<name>.jsonl` - Evidence and atomic statements
- `risk_atlas_nexus/risks_<name>.json` - AI risk assessment results
- `factreasoner/factuality_results_<name>.json` - Factuality verification scores
- `unitxt/<name>.json` - UnitXT benchmark metadata
- `hf/<name>.json` - Hugging Face dataset metadata
- `docling/<name>.json` - Processed academic papers

---

## Logging

By default, the system runs in quiet mode showing only essential output. Use `--debug` flag to see detailed logging:

```bash
# Quiet mode (default) - minimal output
auto-benchmarkcard process glue

# Debug mode - show all workflow steps and tool logs
auto-benchmarkcard process glue --debug
```

---


## Literature

A. Sokol et al., "BenchmarkCards: Standardized Documentation for Large Language Model Benchmarks," Jun. 02, 2025, arXiv: arXiv:2410.12974. doi: 10.48550/arXiv.2410.12974.

R. Marinescu et al., "FactReasoner: A Probabilistic Approach to Long-Form Factuality Assessment for Large Language Models," Feb. 25, 2025, arXiv: arXiv:2502.18573. doi: 10.48550/arXiv.2502.18573.

F. Bagehorn et al., "AI Risk Atlas: Taxonomy and Tooling for Navigating AI Risks and Resources," Feb. 26, 2025, arXiv: arXiv:2503.05780. doi: 10.48550/arXiv.2503.05780.

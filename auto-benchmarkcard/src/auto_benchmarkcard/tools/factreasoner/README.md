# FactReasoner Tool

Automated fact verification for benchmark metadata. Takes claims from benchmark cards and checks them against evidence to see what's actually true.

## How it works

1. **Get the claims**: Takes atomic statements extracted from benchmark cards (like "The dataset has 70,000 examples")
2. **Check against evidence**: Uses NLI to see if supporting evidence confirms, contradicts, or says nothing about each claim
3. **Calculate confidence**: Runs probabilistic reasoning to get a score for how likely each claim is true
4. **Flag problems**: Automatically marks low-confidence fields for human review

## What you get

### Confidence scores
- **High confidence correct** (>0.8): Pretty sure this is right
- **Likely correct** (0.6-0.8): Probably right but not certain
- **Uncertain** (0.4-0.6): Could go either way
- **Likely incorrect** (0.2-0.4): Probably wrong
- **High confidence incorrect** (<0.2): Pretty sure this is wrong

### Field analysis
Groups results by benchmark card sections like:
- `benchmark_details.name`
- `data.size`
- `methodology.metrics`
- `ethical_and_legal_considerations.data_licensing`

### Automated flagging
Marks problematic fields in the final benchmark card:
- `[ALERT, low factuality score: 0.45]` - Not confident about this
- `[ALERT, no evidence found]` - No supporting evidence available

## Usage

### Part of main pipeline
```bash
python agents.py hellaswag
```

### Standalone
```bash
python tools/factreasoner/factreasoner_tool.py path/to/rag_results.jsonl \
    --benchmark-card path/to/benchmark_card.json \
    --threshold 0.8
```

### In code
```python
from tools.factreasoner.factreasoner_tool import evaluate_factuality

results = evaluate_factuality(
    formatted_rag_results=rag_results,
    model="llama-3.3-70b-instruct",
    merlin_path="FactReasoner/merlin/bin/merlin"
)
```

## Setup

You need the Merlin reasoning engine:
```bash
# Download merlin binary and place it here:
mkdir -p fact_reasoner/merlin/bin
# Put merlin binary in fact_reasoner/merlin/bin/merlin
chmod +x fact_reasoner/merlin/bin/merlin
```

## Configuration

- **model**: Which LLM to use for NLI (default: "llama-3.3-70b-instruct")
- **threshold**: Confidence threshold for flagging fields (default: 0.8)
- **merlin_path**: Path to Merlin binary
- **debug_mode**: Show detailed logs

## Output

### Factuality results
```json
{
  "marginals": [
    {
      "variable": "atom_id",
      "p_true": 0.8945
    }
  ],
  "field_analysis": {
    "field_details": {
      "data.size": {
        "avg_probability": 0.65,
        "accuracy_percentage": 67.0
      }
    }
  },
  "atom_summary": [
    {
      "atom_id": "a0",
      "field": "data.size",
      "text": "The dataset contains 70,000 examples",
      "factuality_score": 0.8945,
      "confidence_level": "HIGH_CONFIDENCE_CORRECT"
    }
  ]
}
```

### Flagged benchmark card
The original benchmark card with alerts added to problematic fields:
```json
{
  "benchmark_details": {
    "name": "[ALERT, no evidence found] ethos_binary"
  },
  "data": {
    "size": "[ALERT, low factuality score: 0.45] 998 comments"
  }
}
```

## Technical details

### How the scoring works
1. Uses natural language inference to compare claims with evidence
2. Builds a probabilistic model of relationships
3. Uses Bayesian reasoning (via Merlin) to calculate final confidence scores
4. Groups results by benchmark card field for easier review

### What gets flagged
- Fields where average confidence is below threshold (default 0.8)
- Fields where no evidence was found (all neutral NLI scores)
- Individual claims with very low confidence scores

### Performance
- Processes ~2-3 claims per second with caching
- Uses 2-4GB RAM for large benchmark cards
- Results are cached to avoid recomputation

## Troubleshooting

**Missing Merlin**: Make sure the merlin binary is installed and executable
**Model errors**: Check your LLM API credentials
**Memory issues**: Large datasets might need more RAM
**Path problems**: Verify the fact_reasoner directory structure exists

## Output location

Results are saved to `tools/factreasoner/output/` with files like:
- `factuality_results_hellaswag.json` - Full evaluation results
- Used by the main pipeline to create flagged benchmark cards

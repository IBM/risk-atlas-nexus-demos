from __future__ import annotations

import json
import sys
from pathlib import Path

# Make sure the project root (where unitxt_tool.py lives) is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unitxt_tool import unitxt_benchmark_lookup


def main() -> None:
    benchmark_id = sys.argv[1] if len(sys.argv) > 1 else "safety.truthful_qa"
    try:
        meta = unitxt_benchmark_lookup(benchmark_id)
    except Exception as exc:
        print(f"❌  Lookup failed: {exc}")
        sys.exit(1)

    print(f"✅  Successfully fetched metadata for '{benchmark_id}':\n")
    print(json.dumps(meta.model_dump(), indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rocket_auto_research.dashboard_builder import build_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static Auto Research HTML dashboard.")
    parser.add_argument("--results-dir", default="results", help="Results directory to index.")
    parser.add_argument("--output-dir", default="results/dashboard", help="Output directory for the generated dashboard.")
    args = parser.parse_args()

    index_path = build_dashboard(results_dir=args.results_dir, output_dir=args.output_dir)
    payload = {
        "index_path": str(index_path),
        "output_dir": str(Path(args.output_dir).resolve()),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
import json
from pathlib import Path
from production_rag.evaluation import load_jsonl, run_proxy_evaluation, run_ragas_evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("proxy", "ragas"), default="proxy")
parser.add_argument("--dataset", default="data/evaluation/proxy_golden.jsonl")
parser.add_argument("--threshold", type=float, default=0.85)
parser.add_argument("--output", default="reports/evaluation.json")
args = parser.parse_args()
rows = load_jsonl(args.dataset)
if args.backend == "proxy":
    result = run_proxy_evaluation(rows, args.threshold)
else:
    result = run_ragas_evaluation(rows, args.threshold)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passed"] else 1)

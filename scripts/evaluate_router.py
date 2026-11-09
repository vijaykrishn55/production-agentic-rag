from __future__ import annotations
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = [json.loads(line) for line in (root / "data/evaluation/router_cases.jsonl").read_text().splitlines()]
correct = 0
for row in rows:
    query = row["query"].lower()
    if query.startswith("calculate:") or all(ch in "0123456789.+-*/()% " for ch in query):
        predicted = "calculator"
    else:
        predicted = "internal_retrieval"
    correct += predicted == row["route"]
print(json.dumps({"cases": len(rows), "correct": correct, "accuracy": correct / max(1, len(rows))}, indent=2))
raise SystemExit(0 if correct == len(rows) else 1)

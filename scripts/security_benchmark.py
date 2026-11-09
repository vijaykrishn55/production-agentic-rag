from __future__ import annotations
import json
from pathlib import Path
from production_rag.security import SecurityInterceptor

root = Path(__file__).resolve().parents[1]
security = SecurityInterceptor()
attacks = [json.loads(line)["query"] for line in (root / "data/security/injection_cases.jsonl").read_text().splitlines()]
benign = [json.loads(line)["query"] for line in (root / "data/security/benign_cases.jsonl").read_text().splitlines()]
attack_hits = sum(not security.inspect(query).allowed for query in attacks)
benign_accepts = sum(security.inspect(query).allowed for query in benign)
report = {"attack_cases": len(attacks), "attack_detected": attack_hits,
          "attack_detection_rate": attack_hits / max(1, len(attacks)),
          "benign_cases": len(benign), "benign_accepted": benign_accepts,
          "benign_acceptance_rate": benign_accepts / max(1, len(benign))}
print(json.dumps(report, indent=2))
raise SystemExit(0 if attack_hits == len(attacks) and benign_accepts / len(benign) >= 0.95 else 1)

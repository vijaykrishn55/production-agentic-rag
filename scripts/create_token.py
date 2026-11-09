from __future__ import annotations
import argparse
from production_rag.auth import create_token

parser = argparse.ArgumentParser(description="Create a short-lived local JWT for API testing")
parser.add_argument("--user", required=True)
parser.add_argument("--tenant", required=True)
parser.add_argument("--expires", type=int, default=3600)
args = parser.parse_args()
print(create_token(args.user, args.tenant, expires_seconds=args.expires))

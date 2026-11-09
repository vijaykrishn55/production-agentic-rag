from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from fastapi import Header, HTTPException

from .config import settings


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def create_token(user_id: str, tenant_id: str, secret: str | None = None,
                 expires_seconds: int = 3600) -> str:
    key = (secret or settings.jwt_secret).encode()
    now = int(time.time())
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64(json.dumps({"sub": user_id, "tid": tenant_id, "iss": settings.jwt_issuer,
                              "iat": now, "exp": now + expires_seconds}, separators=(",", ":")).encode())
    signing_input = f"{header}.{payload}".encode()
    signature = _b64(hmac.new(key, signing_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str


def verify_token(token: str, secret: str | None = None) -> Principal:
    try:
        header, payload, signature = token.split(".")
        decode = lambda item: base64.urlsafe_b64decode(item + "=" * (-len(item) % 4))
        claims = json.loads(decode(payload))
        header_data = json.loads(decode(header))
        expected = _b64(hmac.new((secret or settings.jwt_secret).encode(),
                                 f"{header}.{payload}".encode(), hashlib.sha256).digest())
        if header_data.get("alg") != "HS256" or not hmac.compare_digest(signature, expected):
            raise ValueError("invalid signature")
        if claims.get("iss") != settings.jwt_issuer or int(claims.get("exp", 0)) <= int(time.time()):
            raise ValueError("expired or invalid issuer")
        if not claims.get("sub") or not claims.get("tid"):
            raise ValueError("missing principal claims")
        return Principal(str(claims["sub"]), str(claims["tid"]))
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired bearer token") from exc


async def require_principal(authorization: str | None = Header(default=None)) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    return verify_token(authorization.split(" ", 1)[1])

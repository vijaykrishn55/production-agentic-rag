from __future__ import annotations
import hmac
import os
from fastapi import Header, HTTPException

async def require_internal_token(x_internal_token: str | None = Header(default=None)) -> None:
    expected = os.getenv("INTERNAL_SERVICE_TOKEN", "")
    environment = os.getenv("APP_ENV", "development")
    if not expected:
        if environment == "production":
            raise HTTPException(status_code=503, detail="Internal service authentication is not configured")
        return
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=401, detail="Invalid internal service credential")

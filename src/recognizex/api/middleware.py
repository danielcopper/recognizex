"""Middleware: API key authentication."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

if TYPE_CHECKING:
    from recognizex.config import Settings

_bearer_scheme = HTTPBearer(auto_error=False)


def _get_settings_from_request(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


async def verify_api_key(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> None:
    """Check the Bearer token against the configured API key.

    If no API key is configured (RECOGNIZEX_API_KEY not set), all requests pass.
    If configured, requests must include 'Authorization: Bearer <key>'.
    """
    settings = _get_settings_from_request(request)
    if settings.api_key is None:
        return

    if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), settings.api_key.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

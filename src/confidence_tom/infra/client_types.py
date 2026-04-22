from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel


class _RateLimitOrQuota(Exception):
    """Sentinel used by tenacity: raised only when we actually want a retry."""


T = TypeVar("T", bound=BaseModel)

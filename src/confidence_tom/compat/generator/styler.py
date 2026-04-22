"""Legacy style-transfer helpers kept for compatibility."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StyledItem(BaseModel):
    original_text: str = Field(default="")
    styled_text: str = Field(default="")
    style_label: str = Field(default="plain")
    rationale: str = Field(default="")


class StyledOutput(BaseModel):
    items: list[StyledItem] = Field(default_factory=list)
    summary: str = Field(default="")


class StyleTransferer:
    """Compatibility no-op styler."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        self.args = args
        self.kwargs = kwargs

    async def transfer(self, text: str, style: str = "plain") -> StyledOutput:
        item = StyledItem(original_text=text, styled_text=text, style_label=style, rationale="")
        return StyledOutput(items=[item], summary="Compatibility no-op style transfer")

    async def style(self, text: str, style: str = "plain") -> StyledOutput:
        return await self.transfer(text, style=style)

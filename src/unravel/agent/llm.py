"""Model access. The official SDK is called directly; LangGraph only orchestrates.

Direct use keeps control over rules that wrappers may not honour on ``claude-opus-5-5``: no sampling
parameters, no forced ``tool_choice``, explicit effort, and returning assistant content unchanged so
thinking blocks stay valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-opus-5-5"
#: USD per million tokens, from Anthropic's published rates as of 2026-10-06. Check before budgeting.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
}


class BudgetExceededError(RuntimeError):
    """Raised before a request that would pass the configured spending limit."""


class ModelRefusedError(RuntimeError):
    """The whole fallback chain declined. The case goes to a human, with this as the reason."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.requests += other.requests

    def cost(self, model: str = DEFAULT_MODEL) -> float:
        if model not in PRICES:
            return 0.0
        price_in, price_out, price_cache = PRICES[model]
        return (
            self.input_tokens * price_in
            + self.cache_write_tokens * price_in * 1.25
            + self.cache_read_tokens * price_cache
            + self.output_tokens * price_out
        ) / 1_000_000


@dataclass
class SpendGuard:
    """A hard spending limit across a run. It stops further requests, never just warns."""

    limit_usd: float
    model: str = DEFAULT_MODEL
    usage: Usage = field(default_factory=Usage)

    def check(self) -> None:
        spent = self.usage.cost(self.model)
        if spent >= self.limit_usd:
            raise BudgetExceededError(f"spent ${spent:.2f} of ${self.limit_usd:.2f} limit")


@dataclass
class ModelReply:
    content: list[dict[str, Any]]
    stop_reason: str
    usage: Usage

    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")

    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]


class ModelClient(Protocol):
    name: str

    def respond(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        effort: str,
        output_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
    ) -> ModelReply: ...


class AnthropicClient:
    """Claude through the official SDK, with the server-side refusal fallback enabled."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        guard: SpendGuard | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.name = model
        self.guard = guard

    def respond(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        effort: str,
        output_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
    ) -> ModelReply:
        if self.guard is not None:
            self.guard.check()
        output_config: dict[str, Any] = {"effort": effort}
        if output_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": output_schema}
        response = self.client.beta.messages.create(  # type: ignore[call-overload]
            model=self.name,
            max_tokens=max_tokens,
            # A stable prefix for caching: tools, then system, with volatile detail in messages.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"},
            thinking={"type": "adaptive"},
            output_config=output_config,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            requests=1,
        )
        if self.guard is not None:
            self.guard.usage.add(usage)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise ModelRefusedError(str(getattr(details, "category", "unspecified")))
        return ModelReply(
            # Returned unchanged, thinking blocks included, and only ever appended to history.
            content=[block.model_dump(exclude_none=True) for block in response.content],
            stop_reason=response.stop_reason or "",
            usage=usage,
        )


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Prepare a Pydantic schema for ``output_config.format``: closed objects, no ``default``s."""

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            out = {k: clean(v) for k, v in node.items() if k not in ("default", "title")}
            if out.get("type") == "object":
                out["additionalProperties"] = False
                out["required"] = sorted(out.get("properties", {}))
            return out
        if isinstance(node, list):
            return [clean(item) for item in node]
        return node

    result: dict[str, Any] = clean(schema)
    return result

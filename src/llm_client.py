"""OpenAI client wrapper with retry logic and cost tracking."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
_DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Rough token pricing (USD per 1K tokens) for cost estimation
_COST_PER_1K = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}


class LLMUsage(BaseModel):
    """Tracks token consumption and estimated cost for a single call."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class LLMResponse(BaseModel):
    """Wraps an LLM completion with parsed content and usage stats."""

    content: str
    usage: LLMUsage
    parsed: Optional[dict] = None


class LLMClient:
    """Thin wrapper around the OpenAI SDK with JSON-mode support and cost tracking."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.max_retries = max_retries
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._total_cost = 0.0

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        response_format: str = "json_object",
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Call chat completions, returning structured JSON by default."""
        model = model or self.model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if response_format == "json_object":
                    kwargs["response_format"] = {"type": "json_object"}

                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                usage = self._build_usage(model, resp.usage)
                self._total_cost += usage.estimated_cost_usd

                parsed: Optional[dict] = None
                if response_format == "json_object":
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError:
                        parsed = None

                return LLMResponse(content=content, usage=usage, parsed=parsed)

            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)

        raise RuntimeError("LLM call failed after retries")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a list of texts."""
        if not texts:
            return []
        resp = self._client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        resp.data.sort(key=lambda x: x.index)
        return [item.embedding for item in resp.data]

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]

    # ------------------------------------------------------------------
    # Session stats
    # ------------------------------------------------------------------

    @property
    def session_cost_usd(self) -> float:
        return self._total_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_usage(self, model: str, raw_usage: Any) -> LLMUsage:
        prompt_tokens = raw_usage.prompt_tokens if raw_usage else 0
        completion_tokens = raw_usage.completion_tokens if raw_usage else 0
        total_tokens = raw_usage.total_tokens if raw_usage else 0

        pricing = _COST_PER_1K.get(model, {"input": 0.0, "output": 0.0})
        cost = (prompt_tokens / 1000 * pricing["input"]) + (
            completion_tokens / 1000 * pricing["output"]
        )

        return LLMUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost, 6),
        )

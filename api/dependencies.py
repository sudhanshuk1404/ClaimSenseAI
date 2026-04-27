"""Shared application state and FastAPI dependency providers.

All expensive objects (LLM client, analysers, embedded claim index) are
created once at startup inside ``AppState.initialise()`` and reused
across every request.  FastAPI routes access them via ``Depends()``.

Thread-safety note
------------------
The OpenAI SDK is I/O-bound; each call releases the GIL so concurrent
asyncio tasks are safe.  CPU-bound work (KMeans, numpy) runs in a thread
pool via ``asyncio.to_thread()``, keeping the event loop free.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.batch_clusterer import BatchClusterer
from src.data_loader import get_denied_claims, get_paid_claims, load_claims_from_file
from src.denial_analyzer import DenialAnalyzer
from src.llm_client import LLMClient
from src.models import JoinedClaim
from src.pattern_matcher import PatternMatcher


class AppState:
    """Singleton that owns every long-lived object used by the API."""

    def __init__(self) -> None:
        self.llm: LLMClient = None          # type: ignore[assignment]
        self.analyzer: DenialAnalyzer = None  # type: ignore[assignment]
        self.matcher: PatternMatcher = None   # type: ignore[assignment]
        self.clusterer: BatchClusterer = None  # type: ignore[assignment]
        self.all_claims: list[JoinedClaim] = []
        self.denied_claims: list[JoinedClaim] = []
        self.paid_claims: list[JoinedClaim] = []
        self.index_ready: bool = False

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    async def initialise(self) -> None:
        """Called once during FastAPI lifespan startup."""
        # 1. Instantiate LLM client and analysis modules
        self.llm = LLMClient()
        self.analyzer = DenialAnalyzer(self.llm)
        self.matcher = PatternMatcher(self.llm, top_k=5)
        self.clusterer = BatchClusterer(self.llm)

        # 2. Load the claim dataset (fast — pure JSON parsing)
        self.all_claims = load_claims_from_file()
        self.denied_claims = get_denied_claims(self.all_claims)
        self.paid_claims = get_paid_claims(self.all_claims)

        # 3. Pre-embed historical claims so pattern-match requests are instant.
        #    Run in a thread to avoid blocking the event loop during startup.
        await asyncio.to_thread(self.matcher.index_claims, self.all_claims)
        self.index_ready = True

    async def teardown(self) -> None:
        """Called once during FastAPI lifespan shutdown."""
        # Nothing to close for now; add DB/connection cleanup here if needed.
        pass


# ---------------------------------------------------------------------------
# Module-level singleton — created once when the module is first imported
# ---------------------------------------------------------------------------


_state: AppState | None = None


def get_app_state() -> AppState:
    """Return (or lazily create) the global AppState singleton."""
    global _state
    if _state is None:
        _state = AppState()
    return _state


# ---------------------------------------------------------------------------
# FastAPI dependency functions — use these in route signatures
# ---------------------------------------------------------------------------


def get_state() -> AppState:
    return get_app_state()


def get_llm(state: Annotated[AppState, Depends(get_state)]) -> LLMClient:
    return state.llm


def get_analyzer(state: Annotated[AppState, Depends(get_state)]) -> DenialAnalyzer:
    return state.analyzer


def get_matcher(state: Annotated[AppState, Depends(get_state)]) -> PatternMatcher:
    return state.matcher


def get_clusterer(state: Annotated[AppState, Depends(get_state)]) -> BatchClusterer:
    return state.clusterer


def get_all_claims(state: Annotated[AppState, Depends(get_state)]) -> list[JoinedClaim]:
    return state.all_claims


def get_denied_claims_dep(state: Annotated[AppState, Depends(get_state)]) -> list[JoinedClaim]:
    return state.denied_claims

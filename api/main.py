"""ClaimSense AI — FastAPI application.

Four endpoints, one per responsibility:
  GET  /health                    → liveness probe
  GET  /api/v1/claims             → list claims (for UI)
  POST /api/v1/claims/analyze     → Problem 1 + 2
  GET  /api/v1/batch/cluster      → Problem 3

Startup (lifespan):
  1. Load 22 synthetic claims from data/synthetic_claims.json
  2. Init OpenAI LLM client
  3. Pre-embed all claims via text-embedding-3-small so the first
     pattern-match request is fast
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.dependencies import get_app_state
from api.routers import batch, claims


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = get_app_state()
    await state.initialise()
    yield
    await state.teardown()


app = FastAPI(
    title="ClaimSense AI",
    description=(
        "AI-powered healthcare claim denial analysis.\n\n"
        "- **POST /api/v1/claims/analyze** — Problem 1 (root cause) + Problem 2 (pattern match)\n"
        "- **GET  /api/v1/batch/cluster**  — Problem 3 (batch clustering & intelligence)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(claims.router, prefix="/api/v1/claims", tags=["Claims — P1 & P2"])
app.include_router(batch.router,  prefix="/api/v1/batch",  tags=["Batch — P3"])


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"])
async def health():
    state = get_app_state()
    return {
        "status": "ok",
        "model": state.llm.model,
        "total_claims": len(state.all_claims),
        "denied_claims": len(state.denied_claims),
        "paid_claims": len(state.paid_claims),
        "index_ready": state.index_ready,
    }

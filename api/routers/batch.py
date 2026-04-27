"""Problem 3 — Denial Clustering & Batch Intelligence.

Endpoints
---------
GET /api/v1/batch/cluster    Cluster all denied claims → BatchIntelligenceReport

Pipeline
--------
Step 1 — Rule-based primary clustering (payer + CARC code)
  Group by "<payer>|CARC-<code>". Interpretable, zero ML cost.
  Billing teams need clusters like "Aetna — Prior Auth Missing",
  not "Cluster #4".

Step 2 — Semantic sub-clustering (K-means on embeddings, groups ≥ 5 claims)
  Within a large homogeneous group, K-means on OpenAI embeddings separates
  claims by procedure/diagnosis patterns — e.g. a big "BCBS CARC-29" group
  may contain both unrecoverable old filings and edge-case secondary claims.

Step 3 — Appeal rate estimation
  For each cluster: paid_count / total for same payer+CARC in historical data.

Step 4 — Single LLM batch call
  All clusters → ONE GPT-4o call with prompts/batch_clustering.txt.
  Returns: label, billing-team summary, recommended action, priority per cluster.
  One call for N clusters >> N separate calls (cost + latency).

Step 5 — Top opportunity selection
  score = denied_amount × appeal_rate × carc_recoverability_multiplier
  High-recov CARCs: 16, 197, 252, 22 (+10% boost)
  Low-recov CARCs:  18, 97 (×0.3 penalty)
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_all_claims, get_clusterer, get_denied_claims_dep
from src.batch_clusterer import BatchClusterer
from src.models import BatchIntelligenceReport, JoinedClaim

router = APIRouter()

_cached_report: BatchIntelligenceReport | None = None


@router.get(
    "/cluster",
    summary="Batch cluster all denied claims — Problem 3",
    description=(
        "Groups all denied claims into actionable clusters and produces a "
        "batch intelligence report.\n\n"
        "**Pipeline:** rule-based grouping → semantic sub-clustering → "
        "appeal rate estimation → single LLM batch enrichment → "
        "top opportunity scoring.\n\n"
        "Results are cached. Pass `?refresh=true` to rerun."
    ),
    response_model=BatchIntelligenceReport,
)
async def cluster_denied_claims(
    refresh: bool = False,
    denied_claims: list[JoinedClaim] = Depends(get_denied_claims_dep),
    all_claims: list[JoinedClaim] = Depends(get_all_claims),
    clusterer: BatchClusterer = Depends(get_clusterer),
):
    global _cached_report

    if _cached_report is not None and not refresh:
        return _cached_report

    if not denied_claims:
        raise HTTPException(status_code=422, detail="No denied claims found in the dataset.")

    report = await asyncio.to_thread(
        clusterer.analyze_batch,
        denied_claims,
        all_claims,
    )
    _cached_report = report
    return report

"""Problem 1 + 2 — Root Cause Analysis & Pattern Matching.

Endpoints
---------
GET  /api/v1/claims                 List all claims (for UI dropdown)
POST /api/v1/claims/analyze         Analyze a denied claim — returns P1 + P2 results

Problem 1 — Root Cause Analysis
---------------------------------
1. Validate claim is denied (pc_ClaimStatus == "4").
2. Look up CARC code in carc_rarc_codes.json for domain context.
3. Derive days_from_service_to_received so the LLM doesn't do date math.
4. Send claim + context to GPT-4o with prompts/denial_analysis.txt in JSON-mode.
5. Validate response through Pydantic (clamp confidence, default bad enums).
→ Returns: root_cause, carc_interpretation, recoverability, confidence_score,
           supporting_evidence[], recommended_action.

Problem 2 — Historical Pattern Matching
-----------------------------------------
1. Embed the denied claim using text-embedding-3-small (pre-indexed at startup).
2. Score every historical claim with:
       combined = 0.55 × cosine_similarity + 0.45 × structural_score
   where structural_score weights: payer(35%) + procedure(30%) +
   insurance_type(15%) + CARC(10%) + ICD-chapter(10%).
3. Retrieve top-5 most similar claims.
4. Send to GPT-4o with prompts/pattern_matching.txt to:
   - Estimate historical appeal success rate
   - Detect systemic patterns (e.g. "Aetna denies CPT 72148 without auth at 65%")
   - Recommend an appeal strategy
→ Returns: similar_claims[], systemic_pattern, appeal_success_rate, pattern_analysis.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import AppState, get_all_claims, get_analyzer, get_matcher, get_state
from api.schemas import AnalyzeClaimRequest, ClaimSummary, FullClaimAnalysis
from src.data_loader import join_835_837
from src.denial_analyzer import DenialAnalyzer
from src.models import JoinedClaim
from src.pattern_matcher import PatternMatcher

router = APIRouter()


@router.get(
    "",
    summary="List claims",
    description="Returns all claims in the loaded dataset. Use `?outcome=denied` to filter.",
    response_model=list[ClaimSummary],
)
async def list_claims(
    outcome: str | None = Query(default=None, description="'denied' or 'paid'"),
    all_claims: list[JoinedClaim] = Depends(get_all_claims),
):
    claims = all_claims
    if outcome == "denied":
        claims = [c for c in claims if c.is_denied]
    elif outcome == "paid":
        claims = [c for c in claims if not c.is_denied]

    return [
        ClaimSummary(
            claim_id=c.claim_id,
            outcome="denied" if c.is_denied else "paid",
            payer=c.payer_name,
            procedure_code=c.procedure_code,
            insurance_type=c.insurance_type,
            claim_amount=c.edi835.pc_ClaimAmount,
            claim_paid=c.edi835.pc_ClaimPaid,
            carc_code=c.carc_code,
            service_date=c.edi837.ec_ServiceDateFrom,
        )
        for c in claims
    ]


@router.post(
    "/analyze",
    summary="Analyze a denied claim — Problem 1 + 2",
    description=(
        "Submit a denied claim (835 + 837) and receive:\n\n"
        "**Problem 1 — Root Cause Analysis:** WHY the claim was denied, "
        "recoverability verdict, confidence score, supporting evidence, "
        "and recommended action.\n\n"
        "**Problem 2 — Pattern Matching:** Top-5 historically similar claims, "
        "estimated appeal success rate, and systemic pattern detection."
    ),
    response_model=FullClaimAnalysis,
)
async def analyze_claim(
    body: AnalyzeClaimRequest,
    analyzer: DenialAnalyzer = Depends(get_analyzer),
    matcher: PatternMatcher = Depends(get_matcher),
    state: AppState = Depends(get_state),
):
    claim = join_835_837(
        body.edi835.model_dump(exclude_none=True),
        body.edi837.model_dump(exclude_none=True),
    )

    if not claim.is_denied:
        raise HTTPException(
            status_code=422,
            detail="Claim is not denied (pc_ClaimStatus must be '4').",
        )

    cost_before = state.llm.session_cost_usd

    # Run both LLM calls concurrently in threads — neither blocks the event loop
    analysis_task = asyncio.to_thread(analyzer.analyze, claim)
    pattern_task  = asyncio.to_thread(matcher.analyze, claim)
    analysis, pattern = await asyncio.gather(analysis_task, pattern_task)

    return FullClaimAnalysis(
        claim_id=claim.claim_id,
        root_cause_analysis=analysis,
        pattern_match=pattern,
        estimated_cost_usd=round(state.llm.session_cost_usd - cost_before, 6),
    )

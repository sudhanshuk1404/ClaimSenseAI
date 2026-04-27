"""Problem 2: Historical Pattern Matching.

Finds historically similar claims (paid and denied) for a given denied claim,
identifies systemic patterns, and uses an LLM to generate appeal strategy insights.

Similarity is computed as a weighted combination of:
1. Semantic embedding similarity (OpenAI text-embedding-3-small)
2. Structured field matching (payer, procedure code, CARC code, insurance type)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .data_loader import claim_to_text, load_carc_reference
from .llm_client import LLMClient
from .models import JoinedClaim, PatternMatchResult, SimilarClaim

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "pattern_matching.txt"

# Weight split between embedding cosine similarity and structured field matching
_EMBEDDING_WEIGHT = 0.55
_STRUCTURAL_WEIGHT = 0.45


def load_system_prompt() -> str:
    return _PROMPT_PATH.read_text()


class PatternMatcher:
    """Matches a denied claim against historical claims and identifies patterns."""

    def __init__(self, llm: LLMClient, top_k: int = 5) -> None:
        self._llm = llm
        self._top_k = top_k
        self._system_prompt = load_system_prompt()
        self._carc_ref = load_carc_reference()

        # In-memory vector store: filled by index_claims()
        self._indexed_claims: list[JoinedClaim] = []
        self._embeddings: list[list[float]] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_claims(self, historical_claims: list[JoinedClaim]) -> None:
        """Embed and index all historical claims for retrieval."""
        if not historical_claims:
            return

        texts = [claim_to_text(c) for c in historical_claims]
        self._embeddings = self._llm.embed(texts)
        self._indexed_claims = historical_claims

    # ------------------------------------------------------------------
    # Retrieval + analysis
    # ------------------------------------------------------------------

    def find_similar(
        self, denied_claim: JoinedClaim, exclude_self: bool = True
    ) -> list[SimilarClaim]:
        """Return the top-k most similar historical claims with similarity scores."""
        if not self._indexed_claims:
            return []

        query_text = claim_to_text(denied_claim)
        query_emb = self._llm.embed_single(query_text)

        scored: list[tuple[float, JoinedClaim]] = []
        for hist_claim, hist_emb in zip(self._indexed_claims, self._embeddings):
            if exclude_self and hist_claim.claim_id == denied_claim.claim_id:
                continue
            score = self._combined_score(denied_claim, hist_claim, query_emb, hist_emb)
            scored.append((score, hist_claim))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self._top_k]

        results = []
        for score, claim in top:
            match_reasons = self._explain_match(denied_claim, claim)
            outcome = "denied" if claim.is_denied else "paid"
            results.append(
                SimilarClaim(
                    claim_id=claim.claim_id,
                    similarity_score=round(score, 4),
                    match_reasons=match_reasons,
                    outcome=outcome,
                    paid_amount=claim.edi835.pc_ClaimPaid,
                    procedure_code=claim.procedure_code,
                    payer_name=claim.payer_name,
                )
            )
        return results

    def analyze(
        self,
        denied_claim: JoinedClaim,
        historical_claims: Optional[list[JoinedClaim]] = None,
    ) -> PatternMatchResult:
        """Full pattern match: retrieve similar claims + LLM analysis."""
        if historical_claims is not None:
            self.index_claims(historical_claims)

        similar = self.find_similar(denied_claim)

        # Compute stats for the LLM
        denial_stats = self._compute_denial_stats(denied_claim)

        # Historical appeal rate based on retrieved similar paid claims
        paid_count = sum(1 for s in similar if s.outcome == "paid")
        total_similar = len(similar)
        raw_appeal_rate = paid_count / total_similar if total_similar > 0 else 0.0

        user_prompt = self._build_user_prompt(denied_claim, similar, denial_stats)
        response = self._llm.complete(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            response_format="json_object",
        )

        return self._parse_response(
            denied_claim.claim_id,
            similar,
            raw_appeal_rate,
            response.parsed or {},
        )

    # ------------------------------------------------------------------
    # Private: scoring
    # ------------------------------------------------------------------

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    def _structural_score(self, denied: JoinedClaim, hist: JoinedClaim) -> float:
        """Score based on matching key structured fields (0–1)."""
        score = 0.0
        weights = {
            "payer": 0.35,
            "procedure": 0.30,
            "insurance_type": 0.15,
            "carc": 0.10,
            "diagnosis": 0.10,
        }

        if denied.payer_name and hist.payer_name and denied.payer_name == hist.payer_name:
            score += weights["payer"]
        if denied.procedure_code and hist.procedure_code and denied.procedure_code == hist.procedure_code:
            score += weights["procedure"]
        if denied.insurance_type and hist.insurance_type and denied.insurance_type == hist.insurance_type:
            score += weights["insurance_type"]
        if denied.carc_code and hist.carc_code and denied.carc_code == hist.carc_code:
            score += weights["carc"]

        # Partial diagnosis match (same chapter = same 3-char prefix)
        d_dx = denied.edi837.ec_PrincipalDiagnosis or ""
        h_dx = hist.edi837.ec_PrincipalDiagnosis or ""
        if d_dx and h_dx and d_dx[:3] == h_dx[:3]:
            score += weights["diagnosis"]

        return score

    def _combined_score(
        self,
        denied: JoinedClaim,
        hist: JoinedClaim,
        denied_emb: list[float],
        hist_emb: list[float],
    ) -> float:
        emb_score = self._cosine_similarity(denied_emb, hist_emb)
        struct_score = self._structural_score(denied, hist)
        return _EMBEDDING_WEIGHT * emb_score + _STRUCTURAL_WEIGHT * struct_score

    def _explain_match(self, denied: JoinedClaim, hist: JoinedClaim) -> list[str]:
        reasons = []
        if denied.payer_name and denied.payer_name == hist.payer_name:
            reasons.append(f"Same payer: {denied.payer_name}")
        if denied.procedure_code and denied.procedure_code == hist.procedure_code:
            reasons.append(f"Same procedure code: {denied.procedure_code}")
        if denied.insurance_type and denied.insurance_type == hist.insurance_type:
            reasons.append(f"Same insurance type: {denied.insurance_type}")
        if denied.carc_code and denied.carc_code == hist.carc_code:
            reasons.append(f"Same CARC code: {denied.carc_code}")
        d_dx = denied.edi837.ec_PrincipalDiagnosis or ""
        h_dx = hist.edi837.ec_PrincipalDiagnosis or ""
        if d_dx and h_dx and d_dx[:3] == h_dx[:3]:
            reasons.append(f"Similar diagnosis: {d_dx} ≈ {h_dx}")
        if not reasons:
            reasons.append("Semantic/contextual similarity")
        return reasons

    # ------------------------------------------------------------------
    # Private: LLM prompt
    # ------------------------------------------------------------------

    def _compute_denial_stats(self, denied_claim: JoinedClaim) -> dict:
        """Compute aggregated stats for this payer/procedure combination."""
        if not self._indexed_claims:
            return {}

        matches = [
            c for c in self._indexed_claims
            if c.payer_name == denied_claim.payer_name
            and c.procedure_code == denied_claim.procedure_code
        ]
        denied = [c for c in matches if c.is_denied]
        paid = [c for c in matches if not c.is_denied]

        return {
            "payer": denied_claim.payer_name,
            "procedure_code": denied_claim.procedure_code,
            "total_matching_claims": len(matches),
            "denial_count": len(denied),
            "paid_count": len(paid),
            "denial_rate": round(len(denied) / len(matches), 2) if matches else 0.0,
            "total_denied_amount": sum(c.denial_amount for c in denied),
        }

    def _build_user_prompt(
        self,
        denied_claim: JoinedClaim,
        similar: list[SimilarClaim],
        denial_stats: dict,
    ) -> str:
        data = {
            "denied_claim": {
                "claim_id": denied_claim.claim_id,
                "payer": denied_claim.payer_name,
                "procedure_code": denied_claim.procedure_code,
                "insurance_type": denied_claim.insurance_type,
                "carc_code": denied_claim.carc_code,
                "carc_group": denied_claim.carc_group,
                "remark_codes": denied_claim.edi835.pcl_RemarkCodes,
                "diagnosis": denied_claim.edi837.ec_PrincipalDiagnosis,
                "prior_auth": denied_claim.edi837.ec_PriorAuthorization,
                "denial_amount": denied_claim.denial_amount,
                "service_date": denied_claim.edi837.ec_ServiceDateFrom,
                "received_date": denied_claim.edi835.pc_ReceivedDate,
            },
            "similar_claims": [s.model_dump() for s in similar],
            "denial_stats": denial_stats,
        }
        return (
            "Analyze the historical pattern for the following denied claim and return a JSON object "
            "matching the specified schema.\n\n"
            f"DATA:\n{json.dumps(data, indent=2)}\n\n"
            "Return ONLY the JSON object with no additional text."
        )

    def _parse_response(
        self,
        claim_id: str,
        similar: list[SimilarClaim],
        raw_appeal_rate: float,
        parsed: dict,
    ) -> PatternMatchResult:
        llm_rate = parsed.get("historical_appeal_success_rate")
        if llm_rate is not None:
            try:
                final_rate = float(llm_rate)
            except (TypeError, ValueError):
                final_rate = raw_appeal_rate
        else:
            final_rate = raw_appeal_rate

        return PatternMatchResult(
            denied_claim_id=claim_id,
            similar_claims=similar,
            systemic_pattern=parsed.get("systemic_pattern"),
            historical_appeal_success_rate=round(max(0.0, min(1.0, final_rate)), 4),
            pattern_analysis=parsed.get("pattern_analysis", "Insufficient historical data for pattern analysis."),
        )

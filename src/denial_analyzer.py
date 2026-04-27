"""Problem 1: Claim Denial Root Cause Analysis.

Given a denied claim (835 joined with 837), uses an LLM to:
- Identify the root cause beyond the raw CARC code
- Determine recoverability
- Cite supporting evidence from actual claim fields
"""

from __future__ import annotations

import json
from pathlib import Path

from .data_loader import claim_to_analysis_dict, load_carc_reference
from .llm_client import LLMClient
from .models import DenialAnalysis, JoinedClaim, Recoverability

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "denial_analysis.txt"


def load_system_prompt() -> str:
    return _PROMPT_PATH.read_text()


class DenialAnalyzer:
    """Performs root cause analysis on a single denied claim using an LLM."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._system_prompt = load_system_prompt()
        self._carc_ref = load_carc_reference()

    def analyze(self, claim: JoinedClaim) -> DenialAnalysis:
        """Analyze a denied claim and return a structured DenialAnalysis."""
        if not claim.is_denied:
            raise ValueError(f"Claim {claim.claim_id} is not denied (status={claim.edi835.pc_ClaimStatus})")

        carc_context = self._get_carc_context(claim.carc_code)
        user_prompt = self._build_user_prompt(claim, carc_context)

        response = self._llm.complete(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            response_format="json_object",
        )

        return self._parse_response(claim.claim_id, response.parsed or {})

    def analyze_batch(self, claims: list[JoinedClaim]) -> list[DenialAnalysis]:
        """Analyze multiple denied claims sequentially."""
        results = []
        for claim in claims:
            if claim.is_denied:
                results.append(self.analyze(claim))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_carc_context(self, carc_code: str | None) -> dict:
        if not carc_code:
            return {}
        return self._carc_ref.get("carc_codes", {}).get(carc_code, {})

    def _build_user_prompt(self, claim: JoinedClaim, carc_context: dict) -> str:
        claim_data = claim_to_analysis_dict(claim)

        # Enrich with CARC reference data for the LLM
        claim_data["carc_reference"] = carc_context

        # Add derived fields to help the LLM reason about timely filing
        if claim.edi837.ec_ServiceDateFrom and claim.edi835.pc_ReceivedDate:
            from datetime import date
            try:
                svc = date.fromisoformat(claim.edi837.ec_ServiceDateFrom)
                rcv = date.fromisoformat(claim.edi835.pc_ReceivedDate)
                claim_data["derived"] = {
                    "days_from_service_to_received": (rcv - svc).days,
                    "insurance_type": claim.insurance_type,
                }
            except ValueError:
                pass

        return (
            "Analyze the following denied claim and return a JSON object matching the specified schema.\n\n"
            f"CLAIM DATA:\n{json.dumps(claim_data, indent=2)}\n\n"
            "Return ONLY the JSON object with no additional text."
        )

    def _parse_response(self, claim_id: str, parsed: dict) -> DenialAnalysis:
        """Validate and coerce the LLM JSON output into a DenialAnalysis."""
        recov_raw = parsed.get("recoverability", "needs_review")
        try:
            recoverability = Recoverability(recov_raw)
        except ValueError:
            recoverability = Recoverability.NEEDS_REVIEW

        confidence = float(parsed.get("confidence_score", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return DenialAnalysis(
            claim_id=parsed.get("claim_id", claim_id),
            denial_root_cause=parsed.get("denial_root_cause", "Unable to determine root cause."),
            carc_interpretation=parsed.get("carc_interpretation", ""),
            recoverability=recoverability,
            recoverability_rationale=parsed.get("recoverability_rationale", ""),
            confidence_score=confidence,
            supporting_evidence=parsed.get("supporting_evidence", []),
            recommended_action=parsed.get("recommended_action", "Manual review required."),
        )

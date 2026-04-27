"""Tests for DenialAnalyzer — uses mocked LLM to avoid API calls."""

import pytest
from unittest.mock import MagicMock, patch

from src.denial_analyzer import DenialAnalyzer
from src.llm_client import LLMClient, LLMResponse, LLMUsage
from src.models import DenialAnalysis, JoinedClaim, EDI835Claim, EDI837Claim, Recoverability


def _make_claim(
    claim_id: str = "TEST-001",
    carc: str = "29",
    service_date: str = "2025-01-01",
    received_date: str = "2026-01-15",
    insurance_type: str = "Commercial",
) -> JoinedClaim:
    c835 = EDI835Claim(
        pc_ClaimID=claim_id,
        pc_ClaimStatus="4",
        pc_ClaimAmount=5000.0,
        pc_ClaimPaid=0.0,
        pc_InsuranceType=insurance_type,
        pc_ReceivedDate=received_date,
        pcla_AdjustmentGroup="CO",
        pcla_AdjustmentReason=carc,
        pcla_AdjustmentAmount=5000.0,
    )
    c837 = EDI837Claim(
        ec_ClaimNo=claim_id,
        ec_ServiceDateFrom=service_date,
        ec_InsuranceType=insurance_type,
        ec_PrincipalDiagnosis="J06.9",
    )
    return JoinedClaim(claim_id=claim_id, edi835=c835, edi837=c837)


def _mock_llm(response_dict: dict) -> LLMClient:
    llm = MagicMock(spec=LLMClient)
    usage = LLMUsage(
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=300,
        estimated_cost_usd=0.002,
    )
    llm.complete.return_value = LLMResponse(
        content="{}",
        usage=usage,
        parsed=response_dict,
    )
    llm.session_cost_usd = 0.002
    return llm


class TestDenialAnalyzer:
    def test_analyze_denied_claim_returns_denial_analysis(self):
        claim = _make_claim()
        llm = _mock_llm({
            "claim_id": "TEST-001",
            "denial_root_cause": "Claim filed 379 days after service, exceeding 180-day Commercial limit.",
            "carc_interpretation": "CARC 29 indicates timely filing expiration.",
            "recoverability": "not_recoverable",
            "recoverability_rationale": "No delay reason code provided.",
            "confidence_score": 0.92,
            "supporting_evidence": ["pc_ReceivedDate: 2026-01-15", "ec_ServiceDateFrom: 2025-01-01"],
            "recommended_action": "Write off the balance.",
        })
        analyzer = DenialAnalyzer(llm)
        result = analyzer.analyze(claim)

        assert isinstance(result, DenialAnalysis)
        assert result.claim_id == "TEST-001"
        assert result.recoverability == Recoverability.NOT_RECOVERABLE
        assert result.confidence_score == 0.92

    def test_raises_for_non_denied_claim(self):
        c835 = EDI835Claim(pc_ClaimID="PAID-001", pc_ClaimStatus="1", pc_ClaimAmount=500.0, pc_ClaimPaid=400.0)
        c837 = EDI837Claim(ec_ClaimNo="PAID-001")
        claim = JoinedClaim(claim_id="PAID-001", edi835=c835, edi837=c837)

        llm = _mock_llm({})
        analyzer = DenialAnalyzer(llm)
        with pytest.raises(ValueError, match="not denied"):
            analyzer.analyze(claim)

    def test_confidence_score_clamped(self):
        """LLM returns out-of-range confidence; should be clamped to [0, 1]."""
        claim = _make_claim()
        llm = _mock_llm({
            "claim_id": "TEST-001",
            "denial_root_cause": "Test",
            "carc_interpretation": "Test",
            "recoverability": "needs_review",
            "recoverability_rationale": "Test",
            "confidence_score": 1.5,  # out of range
            "supporting_evidence": [],
            "recommended_action": "Review",
        })
        analyzer = DenialAnalyzer(llm)
        result = analyzer.analyze(claim)
        assert result.confidence_score <= 1.0

    def test_invalid_recoverability_defaults_to_needs_review(self):
        claim = _make_claim()
        llm = _mock_llm({
            "claim_id": "TEST-001",
            "denial_root_cause": "Test",
            "carc_interpretation": "Test",
            "recoverability": "INVALID_VALUE",
            "recoverability_rationale": "Test",
            "confidence_score": 0.5,
            "supporting_evidence": [],
            "recommended_action": "Review",
        })
        analyzer = DenialAnalyzer(llm)
        result = analyzer.analyze(claim)
        assert result.recoverability == Recoverability.NEEDS_REVIEW

    def test_analyze_batch_skips_paid_claims(self):
        denied = _make_claim("DENIED-001")
        paid_c835 = EDI835Claim(pc_ClaimID="PAID-001", pc_ClaimStatus="1", pc_ClaimAmount=500.0, pc_ClaimPaid=400.0)
        paid_c837 = EDI837Claim(ec_ClaimNo="PAID-001")
        paid = JoinedClaim(claim_id="PAID-001", edi835=paid_c835, edi837=paid_c837)

        llm = _mock_llm({
            "claim_id": "DENIED-001",
            "denial_root_cause": "Test",
            "carc_interpretation": "Test",
            "recoverability": "recoverable",
            "recoverability_rationale": "Test",
            "confidence_score": 0.8,
            "supporting_evidence": [],
            "recommended_action": "Appeal",
        })
        analyzer = DenialAnalyzer(llm)
        results = analyzer.analyze_batch([denied, paid])

        assert len(results) == 1
        assert results[0].claim_id == "DENIED-001"

    def test_derived_days_passed_to_prompt(self):
        """Check that the days-since-service is computed and present in the prompt."""
        claim = _make_claim(service_date="2025-01-01", received_date="2026-01-15")
        llm = _mock_llm({
            "claim_id": "TEST-001",
            "denial_root_cause": "Filed late",
            "carc_interpretation": "Timely filing",
            "recoverability": "not_recoverable",
            "recoverability_rationale": "Too late",
            "confidence_score": 0.9,
            "supporting_evidence": [],
            "recommended_action": "Write off",
        })
        analyzer = DenialAnalyzer(llm)
        analyzer.analyze(claim)

        # Confirm LLM was called once with a user prompt containing days
        call_args = llm.complete.call_args
        user_prompt = call_args[1]["user_prompt"] if "user_prompt" in call_args[1] else call_args[0][1]
        assert "days_from_service_to_received" in user_prompt

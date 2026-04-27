"""Tests for PatternMatcher — uses mocked LLM to avoid API calls."""

import pytest
from unittest.mock import MagicMock

from src.pattern_matcher import PatternMatcher
from src.llm_client import LLMClient, LLMResponse, LLMUsage
from src.models import JoinedClaim, EDI835Claim, EDI837Claim, PatternMatchResult


def _make_claim(
    claim_id: str,
    payer: str = "Aetna",
    procedure: str = "99213",
    carc: str = "29",
    insurance_type: str = "Commercial",
    status: str = "4",
    paid: float = 0.0,
) -> JoinedClaim:
    amount = 500.0
    c835 = EDI835Claim(
        pc_ClaimID=claim_id,
        pc_ClaimStatus=status,
        pc_ClaimAmount=amount,
        pc_ClaimPaid=paid,
        pc_InsuranceType=insurance_type,
        cp_PayerName=payer,
        pcl_ProcedureCode=procedure,
        pcla_AdjustmentGroup="CO",
        pcla_AdjustmentReason=carc if status == "4" else None,
        pcla_AdjustmentAmount=amount if status == "4" else 0.0,
    )
    c837 = EDI837Claim(
        ec_ClaimNo=claim_id,
        ec_PayerName=payer,
        ec_InsuranceType=insurance_type,
        ec_ServiceDateFrom="2025-06-01",
        ec_PrincipalDiagnosis="J06.9",
        cd_ProcedureCode=procedure,
    )
    return JoinedClaim(claim_id=claim_id, edi835=c835, edi837=c837)


def _mock_llm(embed_vectors: list[list[float]], response_dict: dict) -> LLMClient:
    llm = MagicMock(spec=LLMClient)
    usage = LLMUsage(model="gpt-4o", prompt_tokens=50, completion_tokens=100, total_tokens=150, estimated_cost_usd=0.001)
    llm.embed.return_value = embed_vectors
    llm.embed_single.return_value = embed_vectors[0] if embed_vectors else [0.0] * 10
    llm.complete.return_value = LLMResponse(content="{}", usage=usage, parsed=response_dict)
    llm.session_cost_usd = 0.001
    return llm


def _unit_vec(n: int, dim: int = 10) -> list[float]:
    """Create a simple unit vector for testing similarity."""
    import math
    v = [0.0] * dim
    v[n % dim] = 1.0
    return v


class TestPatternMatcher:
    def test_find_similar_returns_results(self):
        denied = _make_claim("DENIED-001", payer="Aetna", procedure="99213")
        hist1 = _make_claim("HIST-001", payer="Aetna", procedure="99213", status="1", paid=400.0)
        hist2 = _make_claim("HIST-002", payer="Cigna", procedure="99214", status="1", paid=300.0)

        # All have the same embedding so cosine sim = 1.0; structural score differentiates
        embeddings = [[1.0, 0.0] + [0.0] * 8] * 3
        llm = _mock_llm(embeddings, {})
        llm.embed_single.return_value = embeddings[0]

        matcher = PatternMatcher(llm, top_k=2)
        matcher.index_claims([hist1, hist2])
        similar = matcher.find_similar(denied)

        assert len(similar) <= 2
        assert all(0.0 <= s.similarity_score <= 1.0 for s in similar)

    def test_paid_claim_identified_as_paid(self):
        denied = _make_claim("DENIED-001")
        paid = _make_claim("HIST-PAID", status="1", paid=400.0)

        embeddings = [[1.0] + [0.0] * 9] * 2
        llm = _mock_llm(embeddings, {
            "denied_claim_id": "DENIED-001",
            "systemic_pattern": None,
            "historical_appeal_success_rate": 1.0,
            "pattern_analysis": "Similar claims were paid.",
        })
        llm.embed_single.return_value = embeddings[0]

        matcher = PatternMatcher(llm, top_k=5)
        matcher.index_claims([paid])
        result = matcher.analyze(denied)

        paid_claims = [s for s in result.similar_claims if s.outcome == "paid"]
        assert len(paid_claims) >= 1

    def test_self_excluded_from_similar(self):
        claim = _make_claim("SAME-001")
        embeddings = [[1.0] + [0.0] * 9]
        llm = _mock_llm(embeddings, {})
        llm.embed_single.return_value = embeddings[0]

        matcher = PatternMatcher(llm, top_k=5)
        matcher.index_claims([claim])
        similar = matcher.find_similar(claim, exclude_self=True)

        assert all(s.claim_id != "SAME-001" for s in similar)

    def test_analyze_returns_pattern_match_result(self):
        denied = _make_claim("D-001")
        historical = [
            _make_claim("H-001", status="1", paid=400.0),
            _make_claim("H-002", carc="29"),
        ]
        embeddings = [[1.0] + [0.0] * 9] * 3
        llm = _mock_llm(embeddings, {
            "denied_claim_id": "D-001",
            "systemic_pattern": "Aetna denies 99213 with CARC 29 at high rate",
            "historical_appeal_success_rate": 0.5,
            "pattern_analysis": "50% of similar claims were paid.",
        })
        llm.embed_single.return_value = embeddings[0]

        matcher = PatternMatcher(llm, top_k=5)
        result = matcher.analyze(denied, historical_claims=historical)

        assert isinstance(result, PatternMatchResult)
        assert result.denied_claim_id == "D-001"
        assert 0.0 <= result.historical_appeal_success_rate <= 1.0

    def test_structural_score_penalizes_different_payer(self):
        denied = _make_claim("D-001", payer="Aetna")
        diff_payer = _make_claim("H-001", payer="Cigna", status="1", paid=400.0)
        same_payer = _make_claim("H-002", payer="Aetna", status="1", paid=400.0)

        # Both have same embeddings — structural score should differentiate
        embeddings = [[1.0] + [0.0] * 9, [1.0] + [0.0] * 9]
        llm = _mock_llm(embeddings, {})
        llm.embed_single.return_value = [1.0] + [0.0] * 9

        matcher = PatternMatcher(llm, top_k=5)
        matcher.index_claims([diff_payer, same_payer])

        similar = matcher.find_similar(denied)
        # Same payer claim should rank higher
        if len(similar) == 2:
            assert similar[0].claim_id == "H-002"

    def test_empty_historical_returns_empty(self):
        denied = _make_claim("D-001")
        llm = _mock_llm([], {})
        llm.embed_single.return_value = [1.0] + [0.0] * 9

        matcher = PatternMatcher(llm, top_k=5)
        similar = matcher.find_similar(denied)
        assert similar == []

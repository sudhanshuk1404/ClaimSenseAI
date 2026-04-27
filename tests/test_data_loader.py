"""Tests for data loading and model parsing (no API calls needed)."""

import pytest
from pathlib import Path

from src.data_loader import (
    load_claims_from_file,
    get_denied_claims,
    get_paid_claims,
    join_835_837,
    claim_to_text,
    load_carc_reference,
)
from src.models import EDI835Claim, EDI837Claim, JoinedClaim, ClaimStatus


DATA_PATH = Path(__file__).parent.parent / "data" / "synthetic_claims.json"


class TestDataLoading:
    def test_loads_claims_from_file(self):
        claims = load_claims_from_file(DATA_PATH)
        assert len(claims) >= 20, "Should have at least 20 claims"

    def test_all_claims_have_ids(self):
        claims = load_claims_from_file(DATA_PATH)
        for c in claims:
            assert c.claim_id, f"Claim missing ID: {c}"

    def test_claim_ids_match_across_835_837(self):
        claims = load_claims_from_file(DATA_PATH)
        for c in claims:
            assert c.claim_id == c.edi835.pc_ClaimID
            assert c.claim_id == c.edi837.ec_ClaimNo

    def test_has_mix_of_denied_and_paid(self):
        claims = load_claims_from_file(DATA_PATH)
        denied = get_denied_claims(claims)
        paid = get_paid_claims(claims)
        assert len(denied) >= 5, "Should have at least 5 denied claims"
        assert len(paid) >= 5, "Should have at least 5 paid claims"
        assert len(denied) + len(paid) == len(claims)

    def test_denied_claims_have_zero_paid(self):
        claims = load_claims_from_file(DATA_PATH)
        denied = get_denied_claims(claims)
        for c in denied:
            assert c.edi835.pc_ClaimPaid == 0.0, f"Denied claim {c.claim_id} has non-zero paid amount"

    def test_denied_claims_have_carc_codes(self):
        claims = load_claims_from_file(DATA_PATH)
        denied = get_denied_claims(claims)
        for c in denied:
            assert c.carc_code is not None, f"Denied claim {c.claim_id} missing CARC code"

    def test_join_835_837(self):
        edi835 = {
            "pc_ClaimID": "TEST-001",
            "pc_ClaimStatus": "4",
            "pc_ClaimAmount": 1000.0,
            "pc_ClaimPaid": 0.0,
        }
        edi837 = {
            "ec_ClaimNo": "TEST-001",
        }
        joined = join_835_837(edi835, edi837)
        assert joined.claim_id == "TEST-001"
        assert joined.is_denied

    def test_claim_to_text_non_empty(self):
        claims = load_claims_from_file(DATA_PATH)
        for c in claims[:5]:
            text = claim_to_text(c)
            assert len(text) > 0
            assert c.claim_id in text

    def test_load_carc_reference(self):
        ref = load_carc_reference()
        assert "carc_codes" in ref
        assert "29" in ref["carc_codes"]  # Timely filing
        assert "197" in ref["carc_codes"]  # Prior auth


class TestModels:
    def test_denied_claim_is_denied(self):
        claim_835 = EDI835Claim(
            pc_ClaimID="X001",
            pc_ClaimStatus=ClaimStatus.DENIED.value,
            pc_ClaimAmount=500.0,
            pc_ClaimPaid=0.0,
        )
        assert claim_835.is_denied

    def test_paid_claim_is_not_denied(self):
        claim_835 = EDI835Claim(
            pc_ClaimID="X002",
            pc_ClaimStatus=ClaimStatus.PROCESSED_PRIMARY.value,
            pc_ClaimAmount=500.0,
            pc_ClaimPaid=400.0,
        )
        assert not claim_835.is_denied

    def test_denial_amount(self):
        claim_835 = EDI835Claim(
            pc_ClaimID="X003",
            pc_ClaimStatus="4",
            pc_ClaimAmount=1200.0,
            pc_ClaimPaid=0.0,
        )
        assert claim_835.denial_amount == 1200.0

    def test_joined_claim_procedure_code_fallback(self):
        c835 = EDI835Claim(pc_ClaimID="J001", pc_ClaimStatus="4", pc_ClaimAmount=100.0, pc_ClaimPaid=0.0)
        c837 = EDI837Claim(ec_ClaimNo="J001", cd_ProcedureCode="99213")
        joined = JoinedClaim(claim_id="J001", edi835=c835, edi837=c837)
        # 835 pcl_ProcedureCode is None, should fall back to 837 cd_ProcedureCode
        assert joined.procedure_code == "99213"


class TestDenialScenarios:
    """Validate that specific known scenarios are correctly represented in the dataset."""

    def setup_method(self):
        self.claims = load_claims_from_file(DATA_PATH)
        self.claims_by_id = {c.claim_id: c for c in self.claims}

    def test_timely_filing_claim_exists(self):
        claim = self.claims_by_id.get("CLM-2026-00142")
        assert claim is not None
        assert claim.carc_code == "29"
        assert claim.is_denied

    def test_timely_filing_days_calculation(self):
        """Service was 2025-06-15, received 2026-03-20 = 278 days (> 180 day commercial window)."""
        claim = self.claims_by_id.get("CLM-2026-00142")
        from datetime import date
        svc = date.fromisoformat(claim.edi837.ec_ServiceDateFrom)
        rcv = date.fromisoformat(claim.edi835.pc_ReceivedDate)
        days = (rcv - svc).days
        assert days > 180, f"Expected > 180 days gap, got {days}"

    def test_missing_info_claim_has_remark_code(self):
        claim = self.claims_by_id.get("CLM-2026-00287")
        assert claim is not None
        assert claim.carc_code == "16"
        assert claim.edi835.pcl_RemarkCodes is not None

    def test_prior_auth_claim_has_empty_auth(self):
        claim = self.claims_by_id.get("CLM-2026-00510")
        assert claim is not None
        assert claim.carc_code == "197"
        # Prior auth field is empty — this is why it was denied
        auth = claim.edi837.ec_PriorAuthorization
        assert not auth  # empty string or None

    def test_paid_mri_has_prior_auth(self):
        """CLM-2025-08774 is a paid MRI — it should have a prior auth number."""
        claim = self.claims_by_id.get("CLM-2025-08774")
        assert claim is not None
        assert not claim.is_denied
        assert claim.edi837.ec_PriorAuthorization  # non-empty

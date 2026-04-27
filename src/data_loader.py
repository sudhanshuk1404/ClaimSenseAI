"""Loads and joins EDI 835 + EDI 837 claim data from JSON files or dicts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .models import EDI835Claim, EDI837Claim, JoinedClaim

_DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "synthetic_claims.json"
_DEFAULT_CARC_PATH = Path(__file__).parent.parent / "data" / "carc_rarc_codes.json"


def load_claims_from_file(path: Union[str, Path] = _DEFAULT_DATA_PATH) -> list[JoinedClaim]:
    """Load all claims from the synthetic dataset JSON file."""
    with open(path) as f:
        raw = json.load(f)
    return _parse_claims(raw["claims"])


def load_claims_from_dicts(raw_claims: list[dict]) -> list[JoinedClaim]:
    """Parse a list of raw claim dicts (as stored in synthetic_claims.json)."""
    return _parse_claims(raw_claims)


def join_835_837(edi835: dict, edi837: dict) -> JoinedClaim:
    """Join a single 835 + 837 record pair into a JoinedClaim."""
    claim_835 = EDI835Claim(**edi835)
    claim_837 = EDI837Claim(**edi837)
    claim_id = claim_835.pc_ClaimID
    return JoinedClaim(claim_id=claim_id, edi835=claim_835, edi837=claim_837)


def get_denied_claims(claims: list[JoinedClaim]) -> list[JoinedClaim]:
    """Filter to only denied claims (pc_ClaimStatus == '4')."""
    return [c for c in claims if c.is_denied]


def get_paid_claims(claims: list[JoinedClaim]) -> list[JoinedClaim]:
    """Filter to only paid/processed claims."""
    return [c for c in claims if not c.is_denied]


def load_carc_reference(path: Union[str, Path] = _DEFAULT_CARC_PATH) -> dict:
    """Load CARC/RARC code reference dictionary."""
    with open(path) as f:
        return json.load(f)


def claim_to_text(claim: JoinedClaim) -> str:
    """Convert a JoinedClaim to a flat text representation for embedding."""
    parts = [
        f"ClaimID: {claim.claim_id}",
        f"Payer: {claim.payer_name or 'Unknown'}",
        f"InsuranceType: {claim.insurance_type or 'Unknown'}",
        f"ProcedureCode: {claim.procedure_code or 'Unknown'}",
        f"PrincipalDiagnosis: {claim.edi837.ec_PrincipalDiagnosis or 'Unknown'}",
        f"ClaimAmount: {claim.edi835.pc_ClaimAmount}",
        f"ClaimPaid: {claim.edi835.pc_ClaimPaid}",
        f"CARCCode: {claim.carc_code or 'None'}",
        f"AdjustmentGroup: {claim.carc_group or 'None'}",
        f"RemarkCodes: {claim.edi835.pcl_RemarkCodes or 'None'}",
        f"ServiceDate: {claim.edi837.ec_ServiceDateFrom or 'Unknown'}",
        f"PriorAuth: {claim.edi837.ec_PriorAuthorization or 'None'}",
        f"Specialty: {claim.edi837.ec_RendProvSpecialty or 'Unknown'}",
    ]
    return " | ".join(parts)


def claim_to_analysis_dict(claim: JoinedClaim) -> dict:
    """Serialize a JoinedClaim to a flat dict for LLM prompt injection."""
    return {
        "claim_id": claim.claim_id,
        "edi835": claim.edi835.model_dump(exclude_none=True),
        "edi837": claim.edi837.model_dump(exclude_none=True),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_claims(raw_list: list[dict]) -> list[JoinedClaim]:
    claims = []
    for raw in raw_list:
        try:
            c835 = EDI835Claim(**raw["edi835"])
            c837 = EDI837Claim(**raw["edi837"])
            joined = JoinedClaim(
                claim_id=raw["claim_id"],
                edi835=c835,
                edi837=c837,
            )
            claims.append(joined)
        except Exception as exc:
            raise ValueError(f"Failed to parse claim {raw.get('claim_id', '?')}: {exc}") from exc
    return claims

"""API-layer request and response schemas.

Kept intentionally minimal — only what the 3 core endpoints actually need.
Domain models live in src/models.py; these schemas handle HTTP contract only.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from src.models import DenialAnalysis, EDI835Claim, EDI837Claim, PatternMatchResult


class AnalyzeClaimRequest(BaseModel):
    """Request body for POST /api/v1/claims/analyze."""

    edi835: EDI835Claim = Field(..., description="EDI 835 remittance data")
    edi837: EDI837Claim = Field(..., description="EDI 837 original claim submission")

    model_config = {
        "json_schema_extra": {
            "example": {
                "edi835": {
                    "pc_ClaimID": "CLM-2026-00142",
                    "pc_ClaimStatus": "4",
                    "pc_ClaimAmount": 4500.00,
                    "pc_ClaimPaid": 0.00,
                    "pc_InsuranceType": "Commercial",
                    "pc_ReceivedDate": "2026-03-20",
                    "cp_PayerName": "Blue Cross Blue Shield",
                    "pcla_AdjustmentGroup": "CO",
                    "pcla_AdjustmentReason": "29",
                    "pcla_AdjustmentAmount": 4500.00,
                    "pcl_ProcedureCode": "99214",
                },
                "edi837": {
                    "ec_ClaimNo": "CLM-2026-00142",
                    "ec_PayerName": "Blue Cross Blue Shield",
                    "ec_InsuranceType": "Commercial",
                    "ec_ServiceDateFrom": "2025-06-15",
                    "ec_PrincipalDiagnosis": "J06.9",
                    "ec_BillProvNPI": "1234567890",
                    "ec_DelayReasonCode": "",
                    "ec_ClaimFrequency": "1",
                },
            }
        }
    }


class ClaimSummary(BaseModel):
    """Lightweight claim info returned by GET /api/v1/claims."""

    claim_id: str
    outcome: str
    payer: Optional[str] = None
    procedure_code: Optional[str] = None
    insurance_type: Optional[str] = None
    claim_amount: float
    claim_paid: float
    carc_code: Optional[str] = None
    service_date: Optional[str] = None


class FullClaimAnalysis(BaseModel):
    """Combined Problem 1 + Problem 2 response."""

    claim_id: str
    root_cause_analysis: DenialAnalysis
    pattern_match: Optional[PatternMatchResult] = None
    estimated_cost_usd: float = 0.0

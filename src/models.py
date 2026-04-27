"""Pydantic models for EDI 835 (remittance) and EDI 837 (claim submission) data."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ClaimStatus(str, Enum):
    PROCESSED_PRIMARY = "1"
    PROCESSED_SECONDARY = "2"
    PROCESSED_TERTIARY = "3"
    DENIED = "4"
    PROCESSED_PRIMARY_FORWARDED = "19"
    REVERSAL = "22"


class AdjustmentGroup(str, Enum):
    CO = "CO"   # Contractual Obligation
    PR = "PR"   # Patient Responsibility
    OA = "OA"   # Other Adjustment
    PI = "PI"   # Payer Initiated
    CR = "CR"   # Correction / Reversal


class Recoverability(str, Enum):
    RECOVERABLE = "recoverable"
    NOT_RECOVERABLE = "not_recoverable"
    NEEDS_REVIEW = "needs_review"


# ---------------------------------------------------------------------------
# EDI 835 — Remittance Advice
# ---------------------------------------------------------------------------


class EDI835Claim(BaseModel):
    """Per-claim adjudication detail from the 835 remittance."""

    # Payment-level (cp_ prefix)
    cp_PayerName: Optional[str] = None
    cp_PaymentAmount: Optional[float] = None
    cp_PaymentMethod: Optional[str] = None
    cp_PayerID: Optional[str] = None
    cp_PayeeName: Optional[str] = None
    cp_TotalClaimCount: Optional[int] = None
    cp_TotalDeniedChargeAmount: Optional[float] = None
    cp_TotalClaimChargeAmount: Optional[float] = None
    cp_EffectiveDate: Optional[str] = None

    # Claim-level (pc_ prefix)
    pc_ClaimID: str = Field(..., description="Unique claim identifier — joins to 837 ec_ClaimNo")
    pc_ClaimStatus: str = Field(..., description="1=Primary, 2=Secondary, 3=Tertiary, 4=Denied, 19=Primary Forwarded, 22=Reversal")
    pc_ClaimAmount: float = Field(..., description="Total billed amount for this claim")
    pc_ClaimPaid: float = Field(..., description="Amount actually paid by the payer")
    pc_PatientResponsibility: Optional[float] = None
    pc_InsuranceType: Optional[str] = None
    pc_StatementBegin: Optional[str] = None
    pc_StatementEnd: Optional[str] = None
    pc_ReceivedDate: Optional[str] = None
    pc_PriorAuthNum: Optional[str] = None
    pc_PatientLast: Optional[str] = None
    pc_PatientFirst: Optional[str] = None
    pc_RenderingID: Optional[str] = None

    # Line-level (pcl_ prefix)
    pcl_ProcedureCode: Optional[str] = None
    pcl_ProcedureModifier1: Optional[str] = None
    pcl_ProcedureModifier2: Optional[str] = None
    pcl_ChargedAmount: Optional[float] = None
    pcl_PaidAmount: Optional[float] = None
    pcl_AllowedAmount: Optional[float] = None
    pcl_ServiceDate: Optional[str] = None
    pcl_RemarkCodes: Optional[str] = None

    # Adjustment (pcla_ prefix)
    pcla_AdjustmentGroup: Optional[str] = None
    pcla_AdjustmentReason: Optional[str] = None
    pcla_AdjustmentAmount: Optional[float] = None
    pcla_AdjustmentQty: Optional[int] = None

    @property
    def is_denied(self) -> bool:
        return self.pc_ClaimStatus == ClaimStatus.DENIED.value

    @property
    def denial_amount(self) -> float:
        return self.pc_ClaimAmount - self.pc_ClaimPaid


# ---------------------------------------------------------------------------
# EDI 837 — Claim Submission
# ---------------------------------------------------------------------------


class EDI837Claim(BaseModel):
    """Original claim submitted by the healthcare provider."""

    # Claim-level (ec_ prefix)
    ec_ClaimNo: str = Field(..., description="Claim number — joins to 835 pc_ClaimID")
    ec_Amount: Optional[float] = None
    ec_PlaceOfService: Optional[str] = None
    ec_PayerName: Optional[str] = None
    ec_PayerID: Optional[str] = None
    ec_InsuranceType: Optional[str] = None
    ec_PrincipalDiagnosis: Optional[str] = None
    ec_Diag2: Optional[str] = None
    ec_Diag3: Optional[str] = None
    ec_Diag4: Optional[str] = None
    ec_Diag5: Optional[str] = None
    ec_BillProvNPI: Optional[str] = None
    ec_RendProvNPI: Optional[str] = None
    ec_RendProvSpecialty: Optional[str] = None
    ec_ServiceDateFrom: Optional[str] = None
    ec_ServiceDateTo: Optional[str] = None
    ec_PriorAuthorization: Optional[str] = None
    ec_TypeOfBill: Optional[str] = None
    ec_ClaimFrequency: Optional[str] = Field(None, description="1=Original, 7=Replacement, 8=Void")
    ec_DelayReasonCode: Optional[str] = None
    ec_PatientRelationship: Optional[str] = None
    ec_SubscriberID: Optional[str] = None

    # Service line fields (cd_ prefix)
    cd_ProcedureCode: Optional[str] = None
    cd_Modifier1: Optional[str] = None
    cd_Modifier2: Optional[str] = None
    cd_Amount: Optional[float] = None
    cd_Quantity: Optional[int] = None
    cd_DiagPointer1: Optional[str] = None
    cd_PlaceOfService: Optional[str] = None
    cd_ServiceDateFrom: Optional[str] = None
    cd_ServiceDateTo: Optional[str] = None
    cd_RevenueCode: Optional[str] = None
    cd_PriorAuthNo: Optional[str] = None


# ---------------------------------------------------------------------------
# Joined Claim — the combined view used for analysis
# ---------------------------------------------------------------------------


class JoinedClaim(BaseModel):
    """835 + 837 data joined on claim ID, the primary input for analysis."""

    claim_id: str
    edi835: EDI835Claim
    edi837: EDI837Claim

    @property
    def is_denied(self) -> bool:
        return self.edi835.is_denied

    @property
    def procedure_code(self) -> Optional[str]:
        return self.edi835.pcl_ProcedureCode or self.edi837.cd_ProcedureCode

    @property
    def payer_name(self) -> Optional[str]:
        return self.edi835.cp_PayerName or self.edi837.ec_PayerName

    @property
    def insurance_type(self) -> Optional[str]:
        return self.edi835.pc_InsuranceType or self.edi837.ec_InsuranceType

    @property
    def carc_code(self) -> Optional[str]:
        return self.edi835.pcla_AdjustmentReason

    @property
    def carc_group(self) -> Optional[str]:
        return self.edi835.pcla_AdjustmentGroup

    @property
    def denial_amount(self) -> float:
        return self.edi835.denial_amount


# ---------------------------------------------------------------------------
# Analysis Outputs
# ---------------------------------------------------------------------------


class DenialAnalysis(BaseModel):
    """Output of Problem 1: root cause analysis for a single denied claim."""

    claim_id: str
    denial_root_cause: str = Field(..., description="Human-readable root cause explanation")
    carc_interpretation: str = Field(..., description="What the CARC/RARC codes mean in context")
    recoverability: Recoverability
    recoverability_rationale: str = Field(..., description="Why this verdict was reached")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(..., description="Specific fields from the claim supporting this analysis")
    recommended_action: str = Field(..., description="What the billing team should do next")


class SimilarClaim(BaseModel):
    """A historically paid claim similar to the denied claim."""

    claim_id: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    match_reasons: list[str]
    outcome: str  # "paid" or "denied"
    paid_amount: float
    procedure_code: Optional[str] = None
    payer_name: Optional[str] = None


class PatternMatchResult(BaseModel):
    """Output of Problem 2: historical pattern matching."""

    denied_claim_id: str
    similar_claims: list[SimilarClaim]
    systemic_pattern: Optional[str] = Field(None, description="Detected systemic denial pattern if any")
    historical_appeal_success_rate: float = Field(..., ge=0.0, le=1.0)
    pattern_analysis: str = Field(..., description="LLM-generated narrative about the pattern")


class DenialCluster(BaseModel):
    """A cluster of denied claims sharing common characteristics."""

    cluster_id: str
    label: str = Field(..., description="Human-readable cluster name")
    claim_ids: list[str]
    total_denied_amount: float
    payer: Optional[str] = None
    primary_carc_code: Optional[str] = None
    primary_procedure_codes: list[str] = Field(default_factory=list)
    historical_appeal_success_rate: Optional[float] = None
    summary: str = Field(..., description="Billing-team-ready cluster summary")
    recommended_action: str


class BatchIntelligenceReport(BaseModel):
    """Output of Problem 3: batch clustering and intelligence."""

    total_claims_analyzed: int
    total_denied_amount: float
    clusters: list[DenialCluster]
    top_opportunity_cluster_id: str = Field(..., description="Highest-value recoverable cluster")
    executive_summary: str

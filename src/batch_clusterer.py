"""Problem 3: Denial Clustering & Batch Intelligence.

Groups a batch of denied claims into meaningful clusters using:
1. Rule-based primary clustering: payer + CARC code (strong signal, interpretable)
2. Secondary semantic clustering via embeddings for large groups

Then uses an LLM to generate actionable summaries per cluster and an
executive report for leadership.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from .data_loader import claim_to_text
from .llm_client import LLMClient
from .models import BatchIntelligenceReport, DenialCluster, JoinedClaim

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "batch_clustering.txt"

# Minimum claims in a group before we attempt semantic sub-clustering
_MIN_FOR_SEMANTIC = 5
# Max clusters per semantic sub-group
_MAX_SEMANTIC_SUBCLUSTERS = 3


def load_system_prompt() -> str:
    return _PROMPT_PATH.read_text()


class BatchClusterer:
    """Clusters a batch of denied claims and generates actionable batch intelligence."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._system_prompt = load_system_prompt()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        denied_claims: list[JoinedClaim],
        historical_claims: Optional[list[JoinedClaim]] = None,
        appeal_rates: Optional[dict[str, float]] = None,
    ) -> BatchIntelligenceReport:
        """
        Cluster all denied claims and produce a BatchIntelligenceReport.

        Args:
            denied_claims: All denied claims to analyze.
            historical_claims: Optional historical claim set used to estimate appeal rates.
            appeal_rates: Optional pre-computed {cluster_key: rate} map.
        """
        if not denied_claims:
            return BatchIntelligenceReport(
                total_claims_analyzed=0,
                total_denied_amount=0.0,
                clusters=[],
                top_opportunity_cluster_id="none",
                executive_summary="No denied claims to analyze.",
            )

        raw_clusters = self._rule_based_cluster(denied_claims)

        # Optionally split large clusters using semantic sub-clustering
        final_cluster_map: dict[str, list[JoinedClaim]] = {}
        for key, claims in raw_clusters.items():
            if len(claims) >= _MIN_FOR_SEMANTIC:
                sub = self._semantic_subcluster(claims)
                for sub_key, sub_claims in sub.items():
                    final_cluster_map[f"{key}::{sub_key}"] = sub_claims
            else:
                final_cluster_map[key] = claims

        # Build preliminary DenialCluster objects (without LLM summaries yet)
        proto_clusters = []
        for cluster_key, claims in final_cluster_map.items():
            proto = self._build_proto_cluster(cluster_key, claims)
            if appeal_rates:
                proto.historical_appeal_success_rate = appeal_rates.get(cluster_key)
            elif historical_claims:
                proto.historical_appeal_success_rate = self._estimate_appeal_rate(
                    proto, historical_claims
                )
            proto_clusters.append(proto)

        # Sort by denied amount descending for priority ordering
        proto_clusters.sort(key=lambda c: c.total_denied_amount, reverse=True)

        # LLM enrichment: generate labels, summaries, recommended actions
        enriched = self._enrich_with_llm(proto_clusters, denied_claims)

        total_denied = sum(c.total_denied_amount for c in enriched)
        top_id = self._pick_top_opportunity(enriched)

        return BatchIntelligenceReport(
            total_claims_analyzed=len(denied_claims),
            total_denied_amount=round(total_denied, 2),
            clusters=enriched,
            top_opportunity_cluster_id=top_id,
            executive_summary=self._executive_summary_from_clusters(enriched, total_denied),
        )

    # ------------------------------------------------------------------
    # Clustering strategies
    # ------------------------------------------------------------------

    def _rule_based_cluster(
        self, claims: list[JoinedClaim]
    ) -> dict[str, list[JoinedClaim]]:
        """Primary clustering: payer + CARC code combination."""
        groups: dict[str, list[JoinedClaim]] = defaultdict(list)
        for claim in claims:
            payer = claim.payer_name or "Unknown Payer"
            carc = claim.carc_code or "Unknown"
            key = f"{payer}|CARC-{carc}"
            groups[key].append(claim)
        return dict(groups)

    def _semantic_subcluster(
        self, claims: list[JoinedClaim]
    ) -> dict[str, list[JoinedClaim]]:
        """K-means sub-clustering on claim embeddings for large homogeneous groups."""
        if len(claims) < _MIN_FOR_SEMANTIC:
            return {"0": claims}

        texts = [claim_to_text(c) for c in claims]
        try:
            embeddings = self._llm.embed(texts)
        except Exception:
            return {"0": claims}

        X = normalize(np.array(embeddings))
        n_clusters = min(_MAX_SEMANTIC_SUBCLUSTERS, len(claims) // 2)
        if n_clusters < 2:
            return {"0": claims}

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = km.fit_predict(X)

        sub: dict[str, list[JoinedClaim]] = defaultdict(list)
        for claim, label in zip(claims, labels):
            sub[str(label)].append(claim)
        return dict(sub)

    # ------------------------------------------------------------------
    # Cluster building
    # ------------------------------------------------------------------

    def _build_proto_cluster(
        self, cluster_key: str, claims: list[JoinedClaim]
    ) -> DenialCluster:
        """Build a DenialCluster without LLM-generated text."""
        total_denied = sum(c.denial_amount for c in claims)

        # Most common payer
        payer_counts: dict[str, int] = defaultdict(int)
        for c in claims:
            if c.payer_name:
                payer_counts[c.payer_name] += 1
        payer = max(payer_counts, key=payer_counts.get) if payer_counts else None  # type: ignore[arg-type]

        # Most common CARC code
        carc_counts: dict[str, int] = defaultdict(int)
        for c in claims:
            if c.carc_code:
                carc_counts[c.carc_code] += 1
        primary_carc = max(carc_counts, key=carc_counts.get) if carc_counts else None  # type: ignore[arg-type]

        # Top procedure codes
        proc_counts: dict[str, int] = defaultdict(int)
        for c in claims:
            if c.procedure_code:
                proc_counts[c.procedure_code] += 1
        top_procs = sorted(proc_counts, key=proc_counts.get, reverse=True)[:3]  # type: ignore[arg-type]

        cluster_id = f"cluster-{cluster_key.replace('|', '-').replace('::', '-').replace(' ', '_')}"[:60]

        return DenialCluster(
            cluster_id=cluster_id,
            label=cluster_key,  # overwritten by LLM
            claim_ids=[c.claim_id for c in claims],
            total_denied_amount=round(total_denied, 2),
            payer=payer,
            primary_carc_code=primary_carc,
            primary_procedure_codes=top_procs,
            historical_appeal_success_rate=None,
            summary="Pending LLM enrichment.",
            recommended_action="Pending LLM enrichment.",
        )

    def _estimate_appeal_rate(
        self, cluster: DenialCluster, historical: list[JoinedClaim]
    ) -> Optional[float]:
        """Estimate appeal success rate from historical data for this cluster's payer+CARC."""
        matches = [
            c for c in historical
            if c.payer_name == cluster.payer
            and c.carc_code == cluster.primary_carc_code
        ]
        if not matches:
            # Fall back to same payer only
            matches = [c for c in historical if c.payer_name == cluster.payer]
        if not matches:
            return None
        paid = sum(1 for c in matches if not c.is_denied)
        return round(paid / len(matches), 2)

    # ------------------------------------------------------------------
    # LLM enrichment
    # ------------------------------------------------------------------

    def _enrich_with_llm(
        self, proto_clusters: list[DenialCluster], all_denied: list[JoinedClaim]
    ) -> list[DenialCluster]:
        """Send all cluster data to LLM in one call to generate labels/summaries."""
        claim_index = {c.claim_id: c for c in all_denied}

        cluster_input = []
        for cluster in proto_clusters:
            sample_claims = [
                {
                    "claim_id": cid,
                    "procedure_code": claim_index[cid].procedure_code,
                    "diagnosis": claim_index[cid].edi837.ec_PrincipalDiagnosis,
                    "denied_amount": claim_index[cid].denial_amount,
                    "service_date": claim_index[cid].edi837.ec_ServiceDateFrom,
                }
                for cid in cluster.claim_ids[:3]
                if cid in claim_index
            ]
            cluster_input.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "claim_count": len(cluster.claim_ids),
                    "total_denied_amount": cluster.total_denied_amount,
                    "payer": cluster.payer,
                    "primary_carc_code": cluster.primary_carc_code,
                    "primary_procedure_codes": cluster.primary_procedure_codes,
                    "historical_appeal_success_rate": cluster.historical_appeal_success_rate,
                    "sample_claims": sample_claims,
                }
            )

        user_prompt = (
            "Generate labels, summaries, and recommended actions for each cluster below. "
            "Also choose the top_opportunity_cluster_id and write an executive_summary.\n\n"
            f"CLUSTERS:\n{json.dumps(cluster_input, indent=2)}\n\n"
            "Return ONLY the JSON object matching the specified schema."
        )

        response = self._llm.complete(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            response_format="json_object",
        )

        return self._apply_llm_enrichment(proto_clusters, response.parsed or {})

    def _apply_llm_enrichment(
        self, proto_clusters: list[DenialCluster], parsed: dict
    ) -> list[DenialCluster]:
        llm_clusters: list[dict] = parsed.get("clusters", [])
        by_id = {c["cluster_id"]: c for c in llm_clusters if "cluster_id" in c}

        enriched = []
        for cluster in proto_clusters:
            llm_data = by_id.get(cluster.cluster_id, {})
            cluster.label = llm_data.get("label", cluster.label)
            cluster.summary = llm_data.get("summary", cluster.summary)
            cluster.recommended_action = llm_data.get("recommended_action", cluster.recommended_action)
            enriched.append(cluster)
        return enriched

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_top_opportunity(self, clusters: list[DenialCluster]) -> str:
        """Score clusters on value × recoverability to find top opportunity."""
        if not clusters:
            return "none"

        # CARC codes with high recoverability get a multiplier
        high_recov = {"16", "197", "252", "22"}
        medium_recov = {"50", "29", "4"}

        best_score = -1.0
        best_id = clusters[0].cluster_id

        for cluster in clusters:
            rate = cluster.historical_appeal_success_rate or 0.5  # default 50%

            # Adjust by CARC code recoverability
            carc = cluster.primary_carc_code or ""
            if carc in high_recov:
                rate = min(1.0, rate + 0.1)
            elif carc in {"18", "97"}:  # not recoverable
                rate *= 0.3

            score = cluster.total_denied_amount * rate
            if score > best_score:
                best_score = score
                best_id = cluster.cluster_id

        return best_id

    def _executive_summary_from_clusters(
        self, clusters: list[DenialCluster], total_denied: float
    ) -> str:
        if not clusters:
            return "No denial clusters identified."
        top = clusters[0]
        n_clusters = len(clusters)
        return (
            f"Analysis of {sum(len(c.claim_ids) for c in clusters)} denied claims reveals "
            f"{n_clusters} distinct denial pattern(s) totaling ${total_denied:,.2f} in denied charges. "
            f"The highest-priority cluster is '{top.label}' with ${top.total_denied_amount:,.2f} at risk "
            f"across {len(top.claim_ids)} claim(s). "
            f"Recommended focus: {top.recommended_action}"
        )

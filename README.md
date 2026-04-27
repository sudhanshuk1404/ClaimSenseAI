# ClaimSense AI

> AI-powered healthcare claim denial analysis for Revenue Cycle Management (RCM)

Built for the **Gabeo AI ML Engineer Take-Home Assignment**.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/sudhanshuk1404/ClaimSense-AI.git
cd ClaimSense-AI
pip install -r requirements.txt

# 2. Set your OpenAI API key
cp .env.example .env
# Edit .env → OPENAI_API_KEY=sk-...

# 3. Start the API backend (Terminal 1)
python main.py
# → http://localhost:8000/docs

# 4. Start the Streamlit frontend (Terminal 2)
streamlit run streamlit_app.py
# → http://localhost:8501
```

---

## Architecture

```
┌─────────────────────┐        ┌──────────────────────────────────────┐
│   Streamlit UI      │  HTTP  │         FastAPI Backend              │
│  streamlit_app.py   │───────▶│  POST /api/v1/claims/analyze (P1+P2) │
│                     │        │  GET  /api/v1/batch/cluster  (P3)    │
│  Page 1: Overview   │        │  GET  /api/v1/claims         (list)  │
│  Page 2: Analyze    │        │  GET  /health                        │
│  Page 3: Batch      │        └──────────────┬───────────────────────┘
└─────────────────────┘                        │
                                               ▼
                               ┌───────────────────────────────┐
                               │         src/ modules          │
                               │  DenialAnalyzer  (Problem 1)  │
                               │  PatternMatcher  (Problem 2)  │
                               │  BatchClusterer  (Problem 3)  │
                               │  LLMClient  (OpenAI wrapper)  │
                               └───────────────────────────────┘
                                               │
                               ┌───────────────┴───────────────┐
                               │  GPT-4o  +  text-embedding-3  │
                               └───────────────────────────────┘
```

### Project Structure

```
ClaimSense/
├── api/
│   ├── main.py              App factory, lifespan, CORS
│   ├── dependencies.py      Shared singletons via Depends()
│   ├── schemas.py           HTTP request/response models
│   └── routers/
│       ├── claims.py        Problem 1 + 2 endpoints
│       └── batch.py         Problem 3 endpoint
├── src/
│   ├── models.py            Pydantic models (EDI 835/837 + outputs)
│   ├── data_loader.py       Load, join, serialise claims
│   ├── llm_client.py        OpenAI wrapper (retry, JSON-mode, cost)
│   ├── denial_analyzer.py   Problem 1: root cause analysis
│   ├── pattern_matcher.py   Problem 2: hybrid similarity search
│   └── batch_clusterer.py   Problem 3: clustering + LLM enrichment
├── prompts/
│   ├── denial_analysis.txt  System prompt — Problem 1
│   ├── pattern_matching.txt System prompt — Problem 2
│   └── batch_clustering.txt System prompt — Problem 3
├── data/
│   ├── synthetic_claims.json  22 claims (11 denied, 11 paid)
│   └── carc_rarc_codes.json   CARC/RARC reference
├── tests/                   30 unit tests (all mocked)
├── streamlit_app.py         Streamlit frontend
├── main.py                  uvicorn launcher
└── requirements.txt
```

---

## API Reference

| Method | Endpoint | Problem | Description |
|--------|----------|---------|-------------|
| `GET` | `/health` | — | Liveness probe + dataset stats |
| `GET` | `/api/v1/claims` | — | List claims (filter `?outcome=denied`) |
| `POST` | `/api/v1/claims/analyze` | **1 + 2** | Root cause + pattern matching |
| `GET` | `/api/v1/batch/cluster` | **3** | Batch clustering + intelligence report |

Interactive docs: **http://localhost:8000/docs**

---

## The Three Problems

### Problem 1 — Claim Denial Root Cause Analysis

**Goal:** Determine *why* a claim was denied — beyond the raw CARC code.

**How it works (`src/denial_analyzer.py`):**

1. Validate `pc_ClaimStatus == "4"` (denied)
2. Enrich with CARC domain context from `data/carc_rarc_codes.json`
3. Pre-compute `days_from_service_to_received` (removes date math from the LLM)
4. Send full joined claim + context to GPT-4o via `prompts/denial_analysis.txt` with `response_format=json_object`
5. Pydantic validates output: confidence clamped to `[0,1]`, bad enums → `needs_review`

**Returns:** `denial_root_cause`, `carc_interpretation`, `recoverability` (`recoverable / not_recoverable / needs_review`), `confidence_score`, `supporting_evidence[]`, `recommended_action`

**Example:** For CARC 29 (Timely Filing), the system computes 278 days elapsed (Commercial window = 180 days), notes the empty `ec_DelayReasonCode`, and concludes `not_recoverable` at 0.93 confidence — citing the exact field values as evidence.

---

### Problem 2 — Historical Pattern Matching

**Goal:** Find historically similar paid/denied claims and estimate appeal success.

**How it works (`src/pattern_matcher.py`):**

All historical claims are **pre-embedded at server startup** using `text-embedding-3-small` (1536 dims).

Each denied claim is scored against every historical claim using a **hybrid two-stage similarity metric:**

```
combined_score = 0.55 × cosine_similarity + 0.45 × structural_score
```

Structural score weights:

| Field | Weight | Why |
|-------|--------|-----|
| Payer name | 35% | Strongest predictor of payment behaviour |
| Procedure code | 30% | Same CPT = same billing context |
| Insurance type | 15% | Medicare vs Commercial changes all rules |
| CARC code | 10% | Same denial reason = same pattern |
| Diagnosis ICD chapter | 10% | Similar disease = similar case |

Top-5 similar claims + denial stats → GPT-4o → estimates historical appeal rate + detects systemic patterns.

**Why hybrid?** Pure semantic embeddings miss payer-specific behaviour. Pure field matching misses clinically equivalent but differently-coded claims. The hybrid captures both.

---

### Problem 3 — Denial Clustering & Batch Intelligence

**Goal:** Group all denied claims so billing teams know where to focus first.

**How it works (`src/batch_clusterer.py`):**

| Step | What happens |
|------|-------------|
| 1. Rule-based grouping | Cluster by `payer + CARC code` — fully interpretable, zero ML cost |
| 2. Semantic sub-clustering | K-means on embeddings for groups ≥ 5 claims — splits heterogeneous groups further |
| 3. Appeal rate estimation | `paid_count / total` for same payer+CARC in historical data |
| 4. Single LLM batch call | All clusters in ONE GPT-4o call — generates labels, summaries, actions, priorities |
| 5. Top opportunity scoring | `denied_amount × appeal_rate × carc_recoverability_multiplier` |

CARC recoverability multipliers: 16, 197, 252, 22 → +10% boost · 18, 97 → ×0.3 penalty

---

## Synthetic Dataset

22 claims across 5 payers — `data/synthetic_claims.json`

| Scenario | CARC | Payer | Amount |
|----------|------|-------|--------|
| Timely Filing (278 days late, no delay code) | 29 | Blue Cross | $4,500 |
| Missing NPI — RARC N20 | 16 | Medicare | $12,800 |
| Medical Necessity (MRI lumbar, N386) | 50 | Aetna | $8,200 |
| Duplicate claim | 18 | United Healthcare | $3,200 |
| Missing Prior Auth (EGD procedure) | 197 | Cigna | $15,600 |
| Modifier mismatch (modifier 25) | 4 | Aetna | $2,800 |
| Non-covered service (psychotherapy) | 96 | United Healthcare | $950 |
| Documentation required | 252 | Blue Cross | $6,400 |
| Timely Filing (secondary, delay code 2) | 29 | Aetna | $7,100 |
| Prior Auth Missing (brain MRI) | 197 | Cigna | $4,200 |
| Medical Necessity (MRI spine, repeat) | 50 | Aetna | $3,900 |
| **11 paid claims** | — | Various | Historical baseline |

---

## Evaluation

### Unit tests
```bash
python -m pytest tests/ -v     # 30 tests, all pass, no API key needed
```

### Manually validated recoverability verdicts

| Claim | Scenario | Expected | System |
|-------|----------|----------|--------|
| CLM-2026-00142 | 278 days late, Commercial, no delay code | not_recoverable | ✅ 0.93 |
| CLM-2026-00287 | Medicare, missing NPI (RARC N20) | recoverable | ✅ |
| CLM-2026-00391 | Aetna medical necessity, no prior auth | needs_review | ✅ |
| CLM-2026-00455 | Duplicate claim | not_recoverable | ✅ |
| CLM-2026-00510 | Cigna, missing prior auth | needs_review / recoverable | ✅ |

---

## Cost Estimates

| Operation | Est. Cost |
|-----------|-----------|
| Single claim analysis (P1+P2) | ~$0.02–0.04 |
| Batch clustering report (11 denied claims) | ~$0.08–0.15 |
| Full run — all problems | ~$0.20–0.30 |
| Production (1,000 denials/day, GPT-4o-mini for simple cases) | ~$2–10/day |

---

## Design Decisions

| Decision | Chosen | Why |
|----------|--------|-----|
| LLM | GPT-4o | RCM reasoning needs frontier quality |
| Embeddings | text-embedding-3-small | Better than ada-002, 5× cheaper than large |
| Startup indexing | Pre-embed all claims at boot | First request must be fast |
| Primary clustering | Rule-based (payer+CARC) | Interpretability over precision — billing teams need "Aetna Prior Auth", not "Cluster #4" |
| Prompts | Separate .txt files | Easy to read, evaluate, and version independently of code |
| Validation | Pydantic + clamping | Prevents LLM hallucinations from reaching the UI silently |
| Async | `asyncio.to_thread()` | Keeps FastAPI event loop free during blocking LLM calls |
| P1+P2 concurrently | `asyncio.gather()` | Both LLM calls run in parallel — ~40% faster per request |

---

## Known Limitations

1. **In-memory vector store** — PatternMatcher stores embeddings in RAM. Production → ChromaDB or pgvector.
2. **22-claim dataset** — Demonstrates the system; real patterns need thousands of historical claims.
3. **Single service line per claim** — Real EDI data often has multiple lines per claim.
4. **No payer policy database** — Timely filing uses general rules (Medicare=365, Commercial=180 days). Production needs per-payer-per-plan data.

## What I Would Do With More Time

- Persistent vector DB (ChromaDB) so embeddings survive server restart
- Payer policy database (filing windows, auth requirements by CPT code)
- Streaming LLM responses for real-time UI feedback
- Evaluation framework: golden set of labelled claims → precision/recall of recoverability predictions
- Real EDI file parsing (`.edi` → structured data, using `pyx12`)
- Async concurrent batch processing (currently sequential per claim)

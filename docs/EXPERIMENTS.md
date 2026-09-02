# ADAPT-IDS — Experiment Log

All results in this document are from actual experiments.
Fields marked **NOT YET RUN** have not been executed.

---

## E1 — LightGBM Baseline (Random Split)

| Field | Value |
|-------|-------|
| Experiment ID | E1 |
| Objective | Establish baseline LightGBM performance under conventional evaluation |
| Dataset | CIC-IDS2017 |
| Data split | Random 80/20 stratified |
| Model | LightGBM |
| Parameters | See `configs/default.yaml` → models.lightgbm |
| Drift config | None |
| Evaluation metrics | F1, Precision, Recall, FPR, FNR, MCC, confusion matrix |
| Expected observation | Strong performance under IID conditions |
| Actual observation | **NOT YET RUN** — Requires CIC-IDS2017 data in `data/raw/` |

---

## E2 — Random Forest Baseline (Random Split)

| Field | Value |
|-------|-------|
| Experiment ID | E2 |
| Objective | Establish Random Forest baseline for comparison |
| Dataset | CIC-IDS2017 |
| Data split | Random 80/20 stratified |
| Model | Random Forest |
| Actual observation | **NOT YET RUN** |

---

## E3a — LightGBM Temporal Evaluation

| Field | Value |
|-------|-------|
| Experiment ID | E3a |
| Objective | Measure performance degradation under temporal evaluation (RQ1) |
| Dataset | CIC-IDS2017 |
| Data split | Temporal 70/10/20 |
| Model | LightGBM |
| Expected observation | Lower F1 than random split if distribution shifts exist between training and test periods |
| Actual observation | **NOT YET RUN** |

---

## E3b — ADWIN Drift Detection

| Field | Value |
|-------|-------|
| Experiment ID | E3b |
| Objective | Detect concept drift in error-rate stream (RQ2) |
| Dataset | CIC-IDS2017 (temporal test portion) |
| Model | LightGBM (trained on temporal train portion) |
| Drift detector | ADWIN (delta=0.002) |
| Signal | Per-sample error (0 = correct, 1 = incorrect) |
| Expected observation | Drift events correlate with regions of degraded performance |
| Actual observation | **NOT YET RUN** |

---

## E4–E9 — Future Experiments

These experiments belong to Phases 2–3 and are not implemented in Phase 1.
See `docs/RESEARCH_PLAN.md` for the full experiment matrix.

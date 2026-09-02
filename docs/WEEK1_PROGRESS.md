# ADAPT-IDS — Week 1 Progress Report

## Status: Phase 1 Infrastructure Complete

All Phase 1 code, tests, and documentation are implemented. The pipeline is
ready to execute experiments as soon as the CIC-IDS2017 dataset is placed in
`data/raw/`.

---

## Completed

- [x] Project structure (modular Python package under `src/adaptive_ids/`)
- [x] Python virtual environment with pinned dependencies
- [x] YAML-based hierarchical configuration system
- [x] Dataset loader with chunked reading, sampling, and hash tracking
- [x] Dataset inspection script (`inspect_dataset.py`)
- [x] Preprocessing pipeline:
  - Infinity handling, NaN filling (median), duplicate removal
  - ID/leakage column removal
  - Constant-feature removal
  - Binary and multiclass label mapping
  - Leakage-safe fit/transform separation
- [x] Feature selection module (CIC-IDS2017 flow features)
- [x] Baseline IDS models: LightGBM, Random Forest
- [x] Model persistence (joblib) and filesystem registry
- [x] Evaluation:
  - Metrics: Precision, Recall, F1, FPR, FNR, MCC, confusion matrix
  - Random (stratified) and temporal (chronological) splitting
  - Temporal integrity validation (`max(train_ts) < min(test_ts)`)
  - Windowed performance-over-time analysis
- [x] Streaming abstraction: CSVStream, BatchStream
- [x] ADWIN drift detector (via River) with modular interface
- [x] Drift event logger (CSV + JSON)
- [x] Synthetic drift generator (sudden, gradual, incremental, recurring)
- [x] Visualization:
  - Class distribution, feature distributions
  - Confusion matrix
  - F1-over-time with drift markers
  - Error-rate with drift detections
  - Traffic volume over time
- [x] Experiment tracker (JSON metadata)
- [x] Demo script (`run_demo.py`)
- [x] 51 unit tests — all passing
- [x] Research documentation (PROJECT_SPEC, RESEARCH_PLAN, DATASET, ARCHITECTURE, EXPERIMENTS)

## Dataset

**Status**: Infrastructure ready. CIC-IDS2017 CSV files must be downloaded
manually from [UNB/CIC](https://www.unb.ca/cic/datasets/ids-2017.html)
and placed in `data/raw/`.

The pipeline supports:
- Selective loading (per-file, max_rows, sample_fraction)
- All five CIC-IDS2017 day-files
- Automatic column-name normalization

## Models Implemented

| Model | Type | Parameters |
|-------|------|------------|
| LightGBM | Gradient boosting | 200 trees, depth 8, balanced class weights |
| Random Forest | Ensemble bagging | 200 trees, depth 20, balanced class weights |

Both models support:
- Binary (BENIGN/ATTACK) and multiclass classification
- Feature importance extraction
- Model save/load with metadata
- Automatic class encoding

## Experiments

**Actual results**: NOT YET RUN — requires CIC-IDS2017 data.

The pipeline is validated end-to-end using synthetic test data (51 unit tests).
The following experiments are ready to execute:

| ID | Description | Script |
|----|-------------|--------|
| E1 | LightGBM baseline (random split) | `train_baseline.py` |
| E2 | Random Forest baseline (random split) | `train_baseline.py` |
| E3a | Temporal evaluation (both models) | `evaluate_temporal.py` |
| E3b | ADWIN drift detection on error stream | `run_drift_experiment.py` |

## Drift Detection

**ADWIN** detector is implemented and tested:
- Successfully detects distribution shifts in synthetic error streams
- Drift events are logged with full metadata (position, detector state, metrics)
- Configurable delta, clock, window parameters
- Modular `DriftDetector` interface ready for DDM/EDDM/Page-Hinkley

## Tests

**51 tests, 51 passing**

| Module | Tests | Status |
|--------|-------|--------|
| Preprocessing | 10 | Pass |
| Streaming | 7 | Pass |
| Drift (ADWIN + synthetic) | 15 | Pass |
| Evaluation (metrics + split) | 12 | Pass |
| Models | 7 | Pass |

Key test coverage:
- Temporal split integrity: `max(train_ts) < min(test_ts)`
- Train/test index non-overlap
- NaN/infinity handling
- Label mapping (binary + multiclass)
- ADWIN drift detection on shifted distributions
- Synthetic drift generators
- Model save/load roundtrip

## Known Limitations

1. **No real data results yet** — CIC-IDS2017 must be downloaded separately.
2. **Error-rate drift detection requires labels** — In production, ground-truth
   labels are delayed or unavailable. This is documented as a research
   limitation. Unsupervised alternatives are planned for later phases.
3. **Single detector** — Only ADWIN is implemented. DDM/EDDM/Page-Hinkley
   are planned for Phase 2.
4. **No adaptation** — Phase 1 detects drift but does not retrain. Adaptation
   strategies are Phase 3.
5. **Single dataset** — Only CIC-IDS2017 is supported. Cross-dataset
   evaluation is Phase 6.
6. **No statistical significance** — Single-seed experiments. Multiple seeds
   and confidence intervals are planned.

## Problems Encountered

1. Python 3.12 `venv` required system package installation (`python3.12-venv`).
2. CIC-IDS2017 CSVs have inconsistent whitespace in column headers — handled
   by stripping in the loader.
3. Confusion matrix label ordering affected FPR/FNR calculation — fixed by
   explicit positive-class indexing rather than assuming `cm.ravel()` order.

## Next Steps

### Phase 2 — Multiple Drift Detectors
- Implement DDM, EDDM, Page-Hinkley detectors
- Comparative drift detection on same test stream

### Phase 3 — Adaptation Strategies
- Periodic retraining
- Drift-triggered retraining
- Incremental/online learning
- Proposed adaptive strategy with drift-magnitude classification

### Phase 4 — Controlled Drift Experiments
- Use synthetic drift generator on real data
- Compare detector responses to sudden/gradual/incremental/recurring drift

### Phase 5 — Feature Drift Analysis
- Statistical tests for feature distribution changes
- Feature importance shifts over time

## Research Significance

The current implementation establishes the **infrastructure** needed to
investigate RQ1 (static model degradation) and RQ2 (drift detection
effectiveness). The temporal evaluation pipeline with chronological splitting
ensures that performance comparisons between conventional and temporal
evaluation are methodologically sound. The ADWIN integration provides the
first mechanism for automatic drift detection in the error stream.

**What this does NOT prove yet**: No claims about model performance, drift
frequency, or adaptation effectiveness can be made until experiments are
run on real data. All hypothesis testing (H1–H4) awaits experimental results.

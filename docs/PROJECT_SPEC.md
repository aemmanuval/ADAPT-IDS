# ADAPT-IDS — Project Specification

## Project Title

Adaptive Intrusion Detection Under Concept and Feature Drift in Evolving Network Traffic

## Working Name

**ADAPT-IDS**

## Problem Statement

Traditional machine-learning intrusion detection systems (IDS) assume that the
statistical properties of network traffic remain sufficiently stable over time.
In real environments, traffic evolves due to new applications, protocols, user
behaviour changes, cloud migration, IoT adoption, and evolving attack
techniques. This non-stationarity produces **concept drift** (the relationship
between features and labels changes) and **feature drift** (the input feature
distributions shift), degrading model accuracy, increasing false positives and
false negatives, and leaving the network unprotected against novel threats.

## Motivation

1. Static IDS models degrade silently as traffic evolves.
2. Periodic retraining is wasteful when traffic is stable and inadequate
   when drift is rapid.
3. Labelling network traffic for retraining is expensive and often delayed.
4. Existing drift-aware IDS research is growing but lacks consistent
   comparative frameworks across drift types, detectors, and adaptation
   strategies.

## Primary Research Question

> Can drift-triggered adaptive learning maintain intrusion-detection
> performance under evolving network traffic while reducing unnecessary
> model retraining and associated computational/labeling costs?

## Secondary Research Questions

| ID  | Question |
|-----|----------|
| RQ1 | How much does a static IDS degrade when evaluated on temporally later traffic? |
| RQ2 | How effectively can drift-detection algorithms identify meaningful changes? |
| RQ3 | Does drift-triggered adaptation outperform fixed-period retraining? |
| RQ4 | What is the trade-off between detection performance and adaptation cost? |
| RQ5 | How do sudden, gradual, incremental, and recurring drift affect IDS? |
| RQ6 | Can the system adapt without requiring every sample to be manually labelled? |

## Objectives

1. Build a reproducible, modular IDS research pipeline.
2. Establish baseline IDS performance under conventional and temporal evaluation.
3. Demonstrate concept-drift detection using ADWIN on error-rate streams.
4. Compare static, periodic, and drift-triggered adaptation strategies.
5. Evaluate performance under controlled synthetic drift types.
6. Investigate label-efficiency through semi-supervised / active-learning
   mechanisms (later phases).

## Scope — Phase 1 (First Milestone)

- CIC-IDS2017 dataset
- LightGBM + Random Forest baselines
- Binary classification (BENIGN / ATTACK)
- Temporal train/test split with chronological integrity
- ADWIN drift detection on error-rate signal
- Drift event logging and visualization
- Performance-over-time analysis
- 51+ unit tests

## Non-Goals (Phase 1)

- Deep learning (LSTM, Transformers, LLM)
- Real-time PCAP processing
- Zeek / Suricata integration
- Web frontend / API server
- Cloud deployment
- Multi-dataset cross-evaluation

## Assumptions

1. CIC-IDS2017 flow-level CSV data is representative enough for initial
   drift experiments.
2. Timestamps in the dataset are reliable for chronological ordering.
3. Ground-truth labels are available in the benchmark (not in production).
4. A CPU-only environment (4+ cores, 16 GB RAM) is sufficient for Phase 1.

## Technologies

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| ML models | LightGBM, scikit-learn (Random Forest) |
| Drift detection | River (ADWIN) |
| Data | pandas, NumPy, PyArrow |
| Configuration | PyYAML |
| Visualization | matplotlib, seaborn |
| Testing | pytest |
| Environment | venv |

## Architecture

```
Dataset → Preprocessing → Temporal Split → Train Model
                                              ↓
                                     Ordered Test Stream
                                              ↓
                                     Predictions + Error
                                              ↓
                                     ADWIN Drift Detection
                                              ↓
                                  Drift Events + Metrics + Plots
```

See `docs/ARCHITECTURE.md` for the full eventual system diagram.

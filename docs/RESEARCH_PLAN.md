# ADAPT-IDS — Research Plan

## Literature Foundation

### Anchor Papers

1. **Shyaa et al. (2024)** — Survey on concept drift and feature dynamics
   in IDS. DOI: 10.1016/j.engappai.2024.109143

2. **Cai et al. (2025)** — CDDA-MD: Malicious traffic detection based on
   concept drift detection and adaptation. DOI: 10.1016/j.cose.2024.104121

3. **HOIDS (2026)** — Concept drift aware hybrid online intrusion detection
   system. DOI: 10.1016/j.jnca.2026.104556

### Key Observations from Literature

- IDS models face non-stationary traffic in real deployments.
- Concept drift degrades classification performance over time.
- Adaptive and online-learning IDS approaches are increasingly studied.
- Labelling cost is a practical concern in operational environments.
- CDDA-MD uses LSTM + sliding windows; HOIDS uses online/hybrid learning.

## Research Gap (Working Hypothesis)

> Existing research demonstrates drift-aware IDS and online adaptation,
> but there remains room for systematic comparative evaluation of drift
> types, detector behaviour, adaptation strategies, temporal generalization,
> and adaptation cost under a consistent experimental framework.

**This is a hypothesis, not a proven claim.** It requires validation through
literature review and experimental evidence.

## Differentiations

### From CDDA-MD

We start with lightweight classical models (LightGBM) rather than
LSTM + multi-head attention. Motivation: investigate whether a lightweight
adaptive architecture achieves useful drift resilience without deep-learning
computational complexity. **No superiority claim until measured.**

### From HOIDS

We do not present "online drift-aware IDS" as novel. We use HOIDS as
context. Potential differentiation: simpler reproducible baseline, controlled
comparison of drift detectors, temporal vs. synthetic vs. cross-dataset
drift, explicit adaptation-cost analysis.

## Hypotheses

All hypotheses are labelled as such. None are claimed as proven.

| ID | Hypothesis |
|----|------------|
| H1 | A static IDS experiences statistically meaningful performance degradation on temporally later traffic exhibiting distribution changes. |
| H2 | Drift-triggered adaptation can recover part of the performance lost after drift. |
| H3 | Drift-triggered retraining reduces unnecessary retraining vs. fixed-interval while maintaining comparable detection performance. |
| H4 | Different drift types produce different optimal adaptation behaviours. |

## Variables

### Independent Variables

- Model type (LightGBM, Random Forest)
- Drift detector (ADWIN, DDM, EDDM, Page-Hinkley)
- Adaptation strategy (static, periodic, drift-triggered, incremental)
- Drift type (sudden, gradual, incremental, recurring)
- Dataset / temporal period

### Dependent Variables

- F1, Precision, Recall
- FPR, FNR, MCC
- Drift detection delay
- False drift alarms
- Number of retrains
- Adaptation time / CPU cost
- Label requirements

### Control Variables

- Random seed
- Train/test split ratio
- Feature set
- Hyperparameters (fixed within experiment)

## Datasets

| Dataset | Phase | Status |
|---------|-------|--------|
| CIC-IDS2017 | Phase 1 | Primary |
| CSE-CIC-IDS2018 | Phase 6 | Planned |
| UNSW-NB15 | Phase 6 | Planned |

## Experiment Matrix

| ID | Model | Drift Detector | Adaptation | Phase |
|----|-------|----------------|------------|-------|
| E1 | LightGBM | None | None (static) | 1 |
| E2 | RF | None | None (static) | 1 |
| E3 | LightGBM | ADWIN | None (detect only) | 1 |
| E4 | LightGBM | ADWIN | Periodic retraining | 2-3 |
| E5 | LightGBM | ADWIN | Drift-triggered | 2-3 |
| E6 | LightGBM | DDM | Drift-triggered | 2 |
| E7 | LightGBM | EDDM | Drift-triggered | 2 |
| E8 | LightGBM | Page-Hinkley | Drift-triggered | 2 |
| E9 | LightGBM | Adaptive | Proposed strategy | 3+ |

## Evaluation Metrics

### Detection Quality
Precision, Recall, F1, Macro F1, Weighted F1, FPR, FNR, MCC

### Drift Detection
Detection delay, false alarms, missed events, stability

### Adaptation
F1 before/after drift, recovery time, recovery magnitude

### Resource Cost
Training time, number of retrains, processed records, CPU time

## Threats to Validity

1. **Internal**: Data leakage if temporal ordering is violated.
2. **Internal**: CIC-IDS2017 may have labelling errors.
3. **External**: Results on CIC-IDS2017 may not generalize to other networks.
4. **Construct**: Error-rate drift detection requires ground-truth labels.
5. **Statistical**: Single-run experiments without confidence intervals.

## Statistical Evaluation (Later Phases)

- Multiple seeds
- Bootstrap confidence intervals
- Wilcoxon signed-rank test (paired comparisons)
- Friedman/Nemenyi (multiple configurations)
- Document why each test is appropriate.

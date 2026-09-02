# ADAPT-IDS — Architecture

## Current Architecture (Phase 1)

```
configs/default.yaml
       │
       ▼
┌──────────────┐    ┌──────────────┐
│ Data Loader  │───▶│ Preprocessing│
│ (CIC-IDS2017)│    │   Pipeline   │
└──────────────┘    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │  Temporal    │
                    │  Splitter    │
                    └──┬───────┬──┘
                       │       │
                   Train     Test
                       │       │
                       ▼       ▼
              ┌────────────┐  ┌──────────────┐
              │  Baseline  │  │   CSVStream   │
              │    IDS     │  │  (ordered)    │
              │ (LightGBM) │  └──────┬───────┘
              └──────┬─────┘         │
                     │               │
                     └───────┬───────┘
                             │
                     ┌───────┴───────┐
                     │  Predictions  │
                     │   + Errors    │
                     └───────┬───────┘
                             │
                     ┌───────┴───────┐
                     │    ADWIN      │
                     │   Detector    │
                     └───────┬───────┘
                             │
                    ┌────────┴────────┐
                    │  Drift Event   │
                    │    Logger      │
                    └────────┬───────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
         drift_events   windowed_      performance
           .csv         metrics.json   plots (.png)
```

## Module Dependency Graph

```
config/settings ──────────────────────────────────►  (all modules)
utils/logging   ──────────────────────────────────►  (all modules)

data/loader ────► preprocessing/pipeline ────► features/selection
                                                    │
                              evaluation/temporal ◄──┘
                              evaluation/metrics
                                    │
                                    ▼
models/baseline ◄──── scripts ────► streaming/stream
models/registry                         │
                                        ▼
                                  drift/detectors
                                  drift/events
                                        │
                                        ▼
                              visualization/plots
                              experiments/tracker
```

## Eventual Target Architecture

```
                     NETWORK TRAFFIC
                           │
                           ▼
                ┌────────────────────┐
                │ Feature Extraction │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   Traffic Stream   │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   IDS Classifier   │
                └─────────┬──────────┘
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
             Prediction         Confidence
                 │                 │
                 └────────┬────────┘
                          ▼
                ┌────────────────────┐
                │ Monitoring / Drift │
                │ Detection Engine   │
                └─────────┬──────────┘
                          │
                     Drift?
                      /     \
                    No       Yes
                    │         │
                    ▼         ▼
                Continue   Drift Analysis
                              │
                              ▼
                      Adaptation Manager
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
              Update       Incremental   Retrain
              Window       Learning      Model
                 │            │            │
                 └────────────┼────────────┘
                              ▼
                        Model Registry
                              │
                              ▼
                         New Model
                              │
                              ▼
                         Monitoring
```

## Directory Structure

```
adaptive-ids/
├── configs/            YAML experiment configurations
├── data/               Dataset storage (gitignored)
│   ├── raw/            Original CIC-IDS2017 CSVs
│   ├── interim/        Intermediate processing
│   └── processed/      Cleaned Parquet files
├── docs/               Research documentation
├── notebooks/          Exploration and analysis
├── results/            Experiment outputs (gitignored)
│   ├── baseline/       Random-split results
│   ├── temporal/       Temporal-split results
│   ├── drift/          Drift events and logs
│   ├── figures/        Generated plots
│   └── models/         Model registry
├── scripts/            Runnable pipeline entry points
├── src/adaptive_ids/   Core library
│   ├── config/         Configuration management
│   ├── data/           Dataset loading
│   ├── preprocessing/  Cleaning pipeline
│   ├── features/       Feature selection
│   ├── models/         IDS classifiers
│   ├── evaluation/     Metrics and splitting
│   ├── streaming/      Ordered data streams
│   ├── drift/          Drift detection
│   ├── adaptation/     Adaptation strategies (Phase 2+)
│   ├── experiments/    Experiment tracking
│   ├── visualization/  Plot generation
│   └── utils/          Logging, reproducibility
└── tests/              Unit tests
```

## Team Scalability

| Member | Responsibility | Modules |
|--------|---------------|---------|
| 1 — Data & Network | Datasets, PCAP, feature extraction | data/, features/ |
| 2 — IDS/ML | Models, classification, evaluation | models/, evaluation/ |
| 3 — Drift & Adaptation | Drift algorithms, online learning | drift/, adaptation/ |
| 4 — Platform & Eval | API, visualization, experiment automation | visualization/, experiments/ |

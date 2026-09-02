# ADAPT-IDS

**Adaptive Intrusion Detection Under Concept and Feature Drift**

An MSc Cyber Security research prototype investigating whether drift-triggered
adaptive learning can maintain IDS performance under evolving network traffic
while reducing unnecessary retraining.

## Quick Start

### 1. Create environment

**Important:** Use Python 3.10, 3.11, or 3.12. Python 3.13+ and 3.14 lack
prebuilt wheels for `river` and `pyarrow`, causing build failures.

If your system default is 3.13+, create the venv with a specific version:

```bash
# Check your version first
python3 --version

# If 3.13+, use a specific version (install via pyenv, brew, or conda):
# Option A: pyenv
pyenv install 3.12.7
pyenv local 3.12.7
python3 -m venv .venv

# Option B: conda (create a Python 3.12 env, then use its venv)
conda create -n adaptids python=3.12 -y
conda activate adaptids

# Option C: If you already have Python 3.10-3.12 installed
python3.12 -m venv .venv
# or
python3.11 -m venv .venv
```

Then activate:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3. Download dataset

Download CIC-IDS2017 CSV files from
[UNB/CIC](https://www.unb.ca/cic/datasets/ids-2017.html) and place them in
`data/raw/`. See `data/README.md` for details.

### 4. Run the pipeline

```bash
# Inspect dataset
python scripts/inspect_dataset.py

# Preprocess
python scripts/preprocess_dataset.py

# Train baseline models (random split)
python scripts/train_baseline.py

# Temporal evaluation
python scripts/evaluate_temporal.py

# ADWIN drift detection experiment
python scripts/run_drift_experiment.py

# Full demo (all of the above in one command)
python scripts/run_demo.py
```

For a quick run with a smaller sample:

```bash
python scripts/run_demo.py --max-rows 50000
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
├── configs/              YAML experiment configuration
├── data/                 Dataset storage (download required)
├── docs/                 Research documentation
├── notebooks/            Exploration notebooks
├── results/              Experiment outputs (generated)
├── scripts/              Pipeline entry points
├── src/adaptive_ids/     Core library modules
│   ├── config/           Configuration management
│   ├── data/             Dataset loading (chunked, sampling)
│   ├── preprocessing/    Cleaning, encoding, leakage prevention
│   ├── features/         Feature selection & analysis
│   ├── models/           Baseline IDS (LightGBM, Random Forest)
│   ├── evaluation/       Metrics, temporal/random split
│   ├── streaming/        Ordered data stream abstraction
│   ├── drift/            ADWIN drift detection & events
│   ├── adaptation/       Adaptation strategies (Phase 2+)
│   ├── visualization/    Publication-quality plots
│   └── utils/            Logging, reproducibility
└── tests/                Unit tests (51 tests)
```

## Configuration

All experiments are driven by YAML configuration files in `configs/`.

```yaml
dataset:
  sample_fraction: 0.1   # Use 10% of data for development
  max_rows: 100000       # Or cap at 100K rows

evaluation:
  strategy: temporal
  train_fraction: 0.70
  test_fraction: 0.20

experiment:
  random_seed: 42
```

## Research Pipeline

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Dataset loading & preprocessing | Done |
| 1 | Baseline IDS (LightGBM, RF) | Done |
| 1 | Temporal evaluation | Done |
| 1 | Streaming abstraction | Done |
| 1 | ADWIN drift detection | Done |
| 1 | Drift event logging | Done |
| 1 | Performance-over-time visualization | Done |
| 1 | Unit tests (51 tests) | Done |
| 2 | Multiple drift detectors (DDM, EDDM, PH) | Planned |
| 3 | Adaptation strategies | Planned |
| 4 | Controlled synthetic drift experiments | Planned |
| 5 | Feature drift analysis | Planned |
| 6 | Cross-dataset evaluation | Planned |

## Research Questions

1. How much does a static IDS degrade on temporally later traffic?
2. How effectively can drift detectors identify meaningful changes?
3. Does drift-triggered adaptation outperform fixed-period retraining?
4. What is the detection-performance vs. adaptation-cost trade-off?

See `docs/RESEARCH_PLAN.md` for the full research methodology.

## Requirements

- Python 3.10+
- CPU only (no GPU required)
- 16 GB RAM recommended (configurable for less via sampling)
- CIC-IDS2017 dataset (~2 GB CSV files)

## Documentation

- [Project Specification](docs/PROJECT_SPEC.md)
- [Research Plan](docs/RESEARCH_PLAN.md)
- [Dataset Documentation](docs/DATASET.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Experiments](docs/EXPERIMENTS.md)

## License

MIT

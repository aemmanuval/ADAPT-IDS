#!/bin/bash
#
# ADAPT-IDS — Complete Setup & Run Script
# Runs everything from installation to training to results display.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh              # full run
#   ./run_all.sh --quick      # smaller dataset for faster testing
#
# Requirements:
#   - Python 3.10-3.12 (not 3.13+)
#   - CIC-IDS2017 CSVs in data/raw/
#   - UNSW-NB15 CSVs in data/raw_unsw/ (auto-downloads if missing)
#   - MongoDB running locally (optional — falls back gracefully)
#

set -e

# Fix LightGBM OpenMP segfault on macOS ARM
export OMP_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

QUICK=false
if [ "$1" = "--quick" ]; then
    QUICK=true
    echo -e "${YELLOW}Running in QUICK mode (smaller dataset)${NC}"
fi

print_header() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}[✓] $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}[!] $1${NC}"
}

print_error() {
    echo -e "${RED}[✗] $1${NC}"
}

# ── Step 0: Check Python version ─────────────────────────────
print_header "Step 0: Environment Check"

PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Python version: $PYTHON_VERSION"

if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 13 ]; then
    print_error "Python $PYTHON_VERSION detected. Need 3.10-3.12."
    echo "If you have pyenv:"
    echo "  ~/.pyenv/versions/3.12.7/bin/python3 -m venv .venv"
    exit 1
fi

print_step "Python version OK"

# ── Step 1: Virtual environment ──────────────────────────────
print_header "Step 1: Virtual Environment"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    print_step "Created .venv"
else
    print_step ".venv already exists"
fi

source .venv/bin/activate
print_step "Activated .venv ($(python3 --version))"

# ── Step 2: Install dependencies ─────────────────────────────
print_header "Step 2: Install Dependencies"

pip install --upgrade pip -q
pip install -r requirements.txt -q 2>&1 | tail -1
pip install -e . -q 2>&1 | tail -1

# Force single-threaded LightGBM on macOS ARM
export OMP_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1

print_step "All dependencies installed"

# ── Step 3: Check datasets ───────────────────────────────────
print_header "Step 3: Check Datasets"

if [ -d "data/raw" ] && ls data/raw/*.csv 1>/dev/null 2>&1; then
    CSV_COUNT=$(ls data/raw/*.csv | wc -l | tr -d ' ')
    print_step "CIC-IDS2017: $CSV_COUNT CSV files found"
else
    print_error "CIC-IDS2017 not found in data/raw/"
    echo "Download from: https://www.unb.ca/cic/datasets/ids-2017.html"
    echo "Place CSV files in data/raw/ and rerun this script."
    exit 1
fi

if [ -d "data/raw_unsw" ] && ls data/raw_unsw/UNSW-NB15_*.csv 1>/dev/null 2>&1; then
    print_step "UNSW-NB15: found"
else
    print_warn "UNSW-NB15 not found — downloading from Zenodo..."
    mkdir -p data/raw_unsw
    curl -L -o data/raw_unsw/UNSW-NB15_1.csv \
        "https://zenodo.org/api/records/10140548/files/UNSW-NB15_1.csv/content" 2>/dev/null
    curl -L -o data/raw_unsw/UNSW-NB15_2.csv \
        "https://zenodo.org/api/records/10140548/files/UNSW-NB15_2.csv/content" 2>/dev/null

    if head -1 data/raw_unsw/UNSW-NB15_1.csv | grep -q "^[0-9]"; then
        print_step "UNSW-NB15 downloaded"
    else
        print_warn "UNSW-NB15 download may have failed — skipping cross-dataset"
    fi
fi

# ── Step 4: Inspect dataset ──────────────────────────────────
print_header "Step 4: Inspect Dataset"

python scripts/inspect_dataset.py
print_step "Dataset inspection complete"

# ── Step 5: Preprocess ───────────────────────────────────────
print_header "Step 5: Preprocess Dataset"

python scripts/preprocess_dataset.py
print_step "Preprocessing complete"

# ── Step 6: Train baseline models (random split) ────────────
print_header "Step 6: Train Baseline Models (Random Split)"

python scripts/train_baseline.py
print_step "Baseline training complete"

# ── Step 7: Temporal evaluation ──────────────────────────────
print_header "Step 7: Temporal Evaluation"

python scripts/evaluate_temporal.py
print_step "Temporal evaluation complete"

# ── Step 8: ADWIN drift detection ────────────────────────────
print_header "Step 8: ADWIN Drift Detection"

python scripts/run_drift_experiment.py
print_step "Drift detection complete"

# ── Step 9: Adaptation strategies comparison ─────────────────
print_header "Step 9: Adaptation Strategies Comparison"

python scripts/run_adaptation_experiment.py
print_step "Adaptation comparison complete"

# ── Step 10: LSTM vs LightGBM ───────────────────────────────
print_header "Step 10: LSTM vs LightGBM Comparison"

python scripts/run_lstm_comparison.py
print_step "LSTM comparison complete"

# ── Step 11: Synthetic drift experiments ─────────────────────
print_header "Step 11: Synthetic Drift Experiments"

python scripts/run_synthetic_drift.py
print_step "Synthetic drift experiments complete"

# ── Step 12: Feature drift analysis ──────────────────────────
print_header "Step 12: Feature Drift Analysis"

python scripts/run_feature_drift_analysis.py
print_step "Feature drift analysis complete"

# ── Step 13: Cross-dataset training (UNSW-NB15) ─────────────
print_header "Step 13: Cross-Dataset Training (UNSW-NB15)"

if ls data/raw_unsw/UNSW-NB15_*.csv 1>/dev/null 2>&1; then
    python scripts/train_cross_dataset.py
    print_step "Cross-dataset training complete"
else
    print_warn "Skipping — UNSW-NB15 data not available"
fi

# ── Step 14: Run tests ───────────────────────────────────────
print_header "Step 14: Run Tests"

python -m pytest tests/ -v
print_step "All tests passed"

# ── Step 15: Display Results ─────────────────────────────────
print_header "Step 15: RESULTS SUMMARY"

echo ""
echo -e "${CYAN}─── Baseline (Random Split) ───${NC}"
if [ -f "results/baseline/lightgbm/metrics.json" ]; then
    python3 -c "
import json
m = json.load(open('results/baseline/lightgbm/metrics.json'))
print(f'  LightGBM:  F1={m[\"f1\"]:.4f}  Recall={m[\"recall\"]:.4f}  FPR={m.get(\"fpr\", \"N/A\")}')
"
fi
if [ -f "results/baseline/random_forest/metrics.json" ]; then
    python3 -c "
import json
m = json.load(open('results/baseline/random_forest/metrics.json'))
print(f'  RF:        F1={m[\"f1\"]:.4f}  Recall={m[\"recall\"]:.4f}  FPR={m.get(\"fpr\", \"N/A\")}')
"
fi

echo ""
echo -e "${CYAN}─── Temporal Split (The Real Test) ───${NC}"
if [ -f "results/temporal/lightgbm/metrics.json" ]; then
    python3 -c "
import json
m = json.load(open('results/temporal/lightgbm/metrics.json'))
print(f'  LightGBM:  F1={m[\"f1\"]:.4f}  Recall={m[\"recall\"]:.4f}  FNR={m.get(\"fnr\", \"N/A\")}')
print(f'  ⚠️  {m.get(\"fnr\", 0)*100:.1f}% of attacks MISSED without adaptation')
"
fi

echo ""
echo -e "${CYAN}─── Adaptation Strategies ───${NC}"
if [ -f "results/adaptation/comparison.json" ]; then
    python3 -c "
import json
data = json.load(open('results/adaptation/comparison.json'))
print(f'  {\"Strategy\":<25} {\"F1\":>8} {\"Recall\":>8} {\"Retrains\":>10} {\"Cost(s)\":>10}')
print('  ' + '-' * 65)
for name, r in data.items():
    print(f'  {name:<25} {r[\"f1\"]:>8.4f} {r[\"recall\"]:>8.4f} {r[\"n_retrains\"]:>10} {r[\"retrain_time_s\"]:>10.1f}')
"
fi

echo ""
echo -e "${CYAN}─── LSTM vs LightGBM ───${NC}"
if [ -f "results/lstm_comparison/comparison.json" ]; then
    python3 -c "
import json
data = json.load(open('results/lstm_comparison/comparison.json'))
print(f'  {\"Model\":<15} {\"Split\":<10} {\"F1\":>8} {\"Time(s)\":>10}')
print('  ' + '-' * 48)
for key, r in data.items():
    print(f'  {r[\"model\"]:<15} {r[\"split\"]:<10} {r[\"f1\"]:>8.4f} {r[\"training_time_s\"]:>10.1f}')
"
fi

echo ""
echo -e "${CYAN}─── Synthetic Drift ───${NC}"
if [ -f "results/synthetic/synthetic_drift_results.json" ]; then
    python3 -c "
import json
data = json.load(open('results/synthetic/synthetic_drift_results.json'))
print(f'  {\"Drift Type\":<16} {\"Static F1\":>10} {\"Adaptive F1\":>12} {\"Retrains\":>10}')
print('  ' + '-' * 52)
for name, r in data.items():
    print(f'  {name:<16} {r[\"static\"][\"f1\"]:>10.4f} {r[\"adaptive\"][\"f1\"]:>12.4f} {r[\"adaptive\"][\"n_retrains\"]:>10}')
"
fi

echo ""
echo -e "${CYAN}─── Feature Drift ───${NC}"
if [ -f "results/feature_drift/feature_drift_summary.json" ]; then
    python3 -c "
import json
s = json.load(open('results/feature_drift/feature_drift_summary.json'))
print(f'  Features analyzed: {s[\"n_features_analyzed\"]}')
print(f'  Significant drift: {s[\"n_significant_drift\"]} ({s[\"pct_features_drifted\"]}%)')
print(f'  Top shifted: {s[\"top_drifted_features\"][0][\"feature\"]} (KS={s[\"top_drifted_features\"][0][\"ks_statistic\"]:.3f})')
"
fi

echo ""
echo -e "${CYAN}─── Cross-Dataset (UNSW-NB15) ───${NC}"
if [ -f "results/cross_dataset/cross_dataset_summary.json" ]; then
    python3 -c "
import json
s = json.load(open('results/cross_dataset/cross_dataset_summary.json'))
u = s['unsw_internal']
print(f'  UNSW-NB15: F1={u[\"f1\"]:.4f}  Recall={u[\"recall\"]:.4f}')
"
fi

echo ""
echo -e "${CYAN}─── Generated Figures ───${NC}"
if [ -d "results/figures" ]; then
    echo "  $(ls results/figures/*.png 2>/dev/null | wc -l | tr -d ' ') plots saved in results/figures/"
    ls results/figures/*.png 2>/dev/null | head -10 | while read f; do echo "    $(basename $f)"; done
fi

# ── Final ────────────────────────────────────────────────────
print_header "ALL DONE"

echo -e "  ${GREEN}All experiments complete. Results saved.${NC}"
echo ""
echo "  Next steps:"
echo "    streamlit run app.py              # Launch dashboard"
echo "    uvicorn adaptive_ids.api.server:app --port 8000  # Start API"
echo "    sudo python scripts/live_monitor.py              # Live capture"
echo ""
echo "  Push results to GitHub:"
echo "    git add -f results/ data/processed/dataset_profile.json"
echo "    git commit -m 'results: complete experiment results'"
echo "    git pull --rebase && git push"
echo ""

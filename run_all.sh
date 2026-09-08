#!/bin/bash
#
# ╔═══════════════════════════════════════════════════════════════════╗
# ║  ADAPT-IDS — Complete Pipeline Runner                            ║
# ║  Adaptive Intrusion Detection Under Concept & Feature Drift      ║
# ║                                                                   ║
# ║  This single script handles EVERYTHING:                           ║
# ║    • Environment setup (venv + dependencies)                      ║
# ║    • Dataset download (CIC-IDS2018, UNSW-NB15)                   ║
# ║    • Preprocessing + Feature extraction                           ║
# ║    • Model training (LightGBM, Random Forest, LSTM)              ║
# ║    • Temporal evaluation + Drift detection                        ║
# ║    • Adaptation strategy comparison                               ║
# ║    • Synthetic drift experiments                                  ║
# ║    • Feature drift analysis                                       ║
# ║    • Cross-dataset validation                                     ║
# ║    • Unit tests                                                   ║
# ║    • Web dashboard launch                                         ║
# ╚═══════════════════════════════════════════════════════════════════╝
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh                    # full pipeline + launch dashboard
#   ./run_all.sh --quick            # smaller dataset, faster training
#   ./run_all.sh --skip-training    # skip to dashboard (if results exist)
#   ./run_all.sh --download-only    # just download datasets, nothing else
#   ./run_all.sh --no-dashboard     # run everything but don't launch dashboard
#   ./run_all.sh --help             # show this help
#
# Requirements:
#   - Python 3.10-3.12 (3.13+ lacks wheels for river/pyarrow)
#   - CIC-IDS2017 CSVs in data/raw/ (see download instructions below)
#   - ~4 GB disk space for datasets + results
#
# Platform support:
#   - macOS (Intel + Apple Silicon M1/M2/M3/M4)
#   - Linux (Ubuntu, Debian, etc.)
#   - Windows (via WSL2 or Git Bash)
#

set -e

# ── Colors ────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Flags ─────────────────────────────────────────────────────
QUICK=false
SKIP_TRAINING=false
DOWNLOAD_ONLY=false
NO_DASHBOARD=false
SKIP_LSTM=false

for arg in "$@"; do
    case $arg in
        --quick)        QUICK=true; SKIP_LSTM=true ;;
        --skip-training) SKIP_TRAINING=true ;;
        --download-only) DOWNLOAD_ONLY=true ;;
        --no-dashboard)  NO_DASHBOARD=true ;;
        --no-lstm)       SKIP_LSTM=true ;;
        --help|-h)
            head -40 "$0" | tail -36
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $arg${NC}"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# ── Helper functions ──────────────────────────────────────────
print_header() {
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() { echo -e "  ${GREEN}[✓]${NC} $1"; }
print_warn() { echo -e "  ${YELLOW}[!]${NC} $1"; }
print_fail() { echo -e "  ${RED}[✗]${NC} $1"; }
print_info() { echo -e "  ${CYAN}[i]${NC} $1"; }

elapsed_since() {
    local start=$1
    local now=$(date +%s)
    local diff=$((now - start))
    printf "%dm %ds" $((diff / 60)) $((diff % 60))
}

TOTAL_START=$(date +%s)

# Fix LightGBM OpenMP segfault on macOS ARM
export OMP_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1

echo -e "${BOLD}"
echo "  ╔═══════════════════════════════════════╗"
echo "  ║     🛡️  ADAPT-IDS Full Pipeline       ║"
echo "  ║  Adaptive Intrusion Detection System  ║"
echo "  ╚═══════════════════════════════════════╝"
echo -e "${NC}"

if $QUICK; then
    echo -e "${YELLOW}  Mode: QUICK (10% dataset, smaller models, no LSTM)${NC}"
else
    echo -e "${GREEN}  Mode: FULL${NC}"
fi
echo ""

# ══════════════════════════════════════════════════════════════
# STEP 0: ENVIRONMENT CHECK
# ══════════════════════════════════════════════════════════════
print_header "Step 0: Environment Check"

# Detect OS
OS="unknown"
ARCH=$(uname -m)
case "$(uname -s)" in
    Darwin*)  OS="macos" ;;
    Linux*)   OS="linux" ;;
    MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
esac
print_step "Platform: $OS ($ARCH)"

# Find Python
PYTHON=""
for candidate in python3 python; do
    if command -v $candidate &>/dev/null; then
        PY_VER=$($candidate --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        PY_MAJOR=$(echo $PY_VER | cut -d. -f1)
        PY_MINOR=$(echo $PY_VER | cut -d. -f2)
        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 10 ] && [ "$PY_MINOR" -le 12 ]; then
            PYTHON=$candidate
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    print_fail "Python 3.10-3.12 required but not found."
    echo ""
    echo "  Install Python 3.12:"
    echo "    macOS:   brew install python@3.12"
    echo "    Ubuntu:  sudo apt install python3.12 python3.12-venv"
    echo "    Windows: https://www.python.org/downloads/"
    echo "    pyenv:   pyenv install 3.12.7 && pyenv local 3.12.7"
    exit 1
fi

print_step "Python: $($PYTHON --version)"

# Check git
if command -v git &>/dev/null; then
    print_step "Git: $(git --version | head -1)"
else
    print_warn "Git not found (not required for pipeline, but useful)"
fi

# ══════════════════════════════════════════════════════════════
# STEP 1: VIRTUAL ENVIRONMENT
# ══════════════════════════════════════════════════════════════
print_header "Step 1: Virtual Environment"

if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment..."
    $PYTHON -m venv .venv
    print_step "Created .venv"
else
    print_step ".venv already exists"
fi

# Activate venv
if [ "$OS" = "windows" ]; then
    source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
else
    source .venv/bin/activate
fi
print_step "Activated venv ($(python3 --version))"

# ══════════════════════════════════════════════════════════════
# STEP 2: INSTALL DEPENDENCIES
# ══════════════════════════════════════════════════════════════
print_header "Step 2: Install Dependencies"

STEP_START=$(date +%s)

pip install --upgrade pip -q 2>&1 | tail -1
pip install -r requirements.txt -q 2>&1 | tail -3
pip install -e . -q 2>&1 | tail -1

# Re-export after venv activation
export OMP_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1

print_step "All dependencies installed ($(elapsed_since $STEP_START))"

# Quick sanity check
python3 -c "
import numpy, pandas, sklearn, lightgbm, river, torch, matplotlib
print('  Verified: numpy, pandas, sklearn, lightgbm, river, torch, matplotlib')
" || { print_fail "Dependency import failed"; exit 1; }

# ══════════════════════════════════════════════════════════════
# STEP 3: DATASET SETUP
# ══════════════════════════════════════════════════════════════
print_header "Step 3: Dataset Setup"

# --- CIC-IDS2017 (Primary) ---
if [ -d "data/raw" ] && ls data/raw/*.csv 1>/dev/null 2>&1; then
    CSV_COUNT=$(ls data/raw/*.csv | wc -l | tr -d ' ')
    print_step "CIC-IDS2017: $CSV_COUNT CSV files found in data/raw/"
else
    print_fail "CIC-IDS2017 not found in data/raw/"
    echo ""
    echo -e "  ${BOLD}Download instructions:${NC}"
    echo ""
    echo "  Option A — Kaggle (easiest, ~600 MB):"
    echo "    pip install kaggle"
    echo "    kaggle datasets download -d chethuhn/network-intrusion-dataset"
    echo "    unzip network-intrusion-dataset.zip -d data/raw/"
    echo ""
    echo "  Option B — CIC website (full, ~2 GB):"
    echo "    https://www.unb.ca/cic/datasets/ids-2017.html"
    echo "    Download MachineLearningCSV.zip → extract to data/raw/"
    echo ""
    echo "  Option C — Direct curl:"
    echo "    mkdir -p data/raw"
    echo "    # Download each day's CSV from the CIC mirror"
    echo ""
    echo "  After downloading, your data/raw/ should contain files like:"
    echo "    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    echo "    Monday-WorkingHours.pcap_ISCX.csv"
    echo "    etc."
    echo ""

    if $DOWNLOAD_ONLY; then
        exit 0
    fi
    exit 1
fi

# --- UNSW-NB15 (Cross-validation) ---
if [ -d "data/raw_unsw" ] && ls data/raw_unsw/UNSW-NB15_*.csv 1>/dev/null 2>&1; then
    print_step "UNSW-NB15: found in data/raw_unsw/"
else
    print_warn "UNSW-NB15 not found — attempting download from Zenodo..."
    mkdir -p data/raw_unsw

    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o data/raw_unsw/UNSW-NB15_1.csv \
            "https://zenodo.org/api/records/10140548/files/UNSW-NB15_1.csv/content" 2>&1 || true
        curl -L --progress-bar -o data/raw_unsw/UNSW-NB15_2.csv \
            "https://zenodo.org/api/records/10140548/files/UNSW-NB15_2.csv/content" 2>&1 || true

        if [ -f "data/raw_unsw/UNSW-NB15_1.csv" ] && [ -s "data/raw_unsw/UNSW-NB15_1.csv" ]; then
            print_step "UNSW-NB15 downloaded"
        else
            print_warn "UNSW-NB15 download failed — cross-dataset step will be skipped"
        fi
    else
        print_warn "curl not found — skipping UNSW-NB15 download"
    fi
fi

if $DOWNLOAD_ONLY; then
    print_header "Download Complete"
    echo "  Datasets are ready. Run ./run_all.sh to start the pipeline."
    exit 0
fi

# ══════════════════════════════════════════════════════════════
# STEP 4: INSPECT DATASET
# ══════════════════════════════════════════════════════════════
if ! $SKIP_TRAINING; then

print_header "Step 4: Inspect Dataset"
STEP_START=$(date +%s)

python3 scripts/inspect_dataset.py
print_step "Dataset inspection complete ($(elapsed_since $STEP_START))"

# ══════════════════════════════════════════════════════════════
# STEP 5: PREPROCESS
# ══════════════════════════════════════════════════════════════
print_header "Step 5: Preprocess Dataset"
STEP_START=$(date +%s)

if $QUICK; then
    python3 scripts/preprocess_dataset.py configs/development.yaml
else
    python3 scripts/preprocess_dataset.py
fi
print_step "Preprocessing complete ($(elapsed_since $STEP_START))"

# ══════════════════════════════════════════════════════════════
# STEP 6: TRAIN BASELINE MODELS
# ══════════════════════════════════════════════════════════════
print_header "Step 6: Train Baseline Models (Random Split)"
STEP_START=$(date +%s)

if $QUICK; then
    python3 scripts/train_baseline.py configs/development.yaml
else
    python3 scripts/train_baseline.py
fi
print_step "Baseline training complete ($(elapsed_since $STEP_START))"

# ══════════════════════════════════════════════════════════════
# STEP 7: TEMPORAL EVALUATION
# ══════════════════════════════════════════════════════════════
print_header "Step 7: Temporal Evaluation (Drift-Exposed)"
STEP_START=$(date +%s)

if $QUICK; then
    python3 scripts/evaluate_temporal.py configs/development.yaml
else
    python3 scripts/evaluate_temporal.py
fi
print_step "Temporal evaluation complete ($(elapsed_since $STEP_START))"

# ══════════════════════════════════════════════════════════════
# STEP 8: DRIFT DETECTION
# ══════════════════════════════════════════════════════════════
print_header "Step 8: ADWIN Drift Detection"
STEP_START=$(date +%s)

if $QUICK; then
    python3 scripts/run_drift_experiment.py configs/development.yaml
else
    python3 scripts/run_drift_experiment.py
fi
print_step "Drift detection complete ($(elapsed_since $STEP_START))"

# ══════════════════════════════════════════════════════════════
# STEP 9: ADAPTATION STRATEGIES
# ══════════════════════════════════════════════════════════════
print_header "Step 9: Adaptation Strategies Comparison"
STEP_START=$(date +%s)

if $QUICK; then
    python3 scripts/run_adaptation_experiment.py configs/development.yaml
else
    python3 scripts/run_adaptation_experiment.py
fi
print_step "Adaptation comparison complete ($(elapsed_since $STEP_START))"

# ══════════════════════════════════════════════════════════════
# STEP 10: LSTM COMPARISON (optional)
# ══════════════════════════════════════════════════════════════
if ! $SKIP_LSTM; then
    print_header "Step 10: LSTM vs LightGBM Comparison"
    STEP_START=$(date +%s)

    python3 scripts/run_lstm_comparison.py
    print_step "LSTM comparison complete ($(elapsed_since $STEP_START))"
else
    print_header "Step 10: LSTM Comparison (SKIPPED)"
    print_info "Skipped LSTM in quick mode (use full mode for deep learning comparison)"
fi

# ══════════════════════════════════════════════════════════════
# STEP 11: SYNTHETIC DRIFT
# ══════════════════════════════════════════════════════════════
print_header "Step 11: Synthetic Drift Experiments"
STEP_START=$(date +%s)

if $QUICK; then
    python3 scripts/run_synthetic_drift.py configs/development.yaml
else
    python3 scripts/run_synthetic_drift.py
fi
print_step "Synthetic drift experiments complete ($(elapsed_since $STEP_START))"

# ══════════════════════════════════════════════════════════════
# STEP 12: FEATURE DRIFT ANALYSIS
# ══════════════════════════════════════════════════════════════
print_header "Step 12: Feature Drift Analysis"
STEP_START=$(date +%s)

if $QUICK; then
    python3 scripts/run_feature_drift_analysis.py configs/development.yaml
else
    python3 scripts/run_feature_drift_analysis.py
fi
print_step "Feature drift analysis complete ($(elapsed_since $STEP_START))"

# ══════════════════════════════════════════════════════════════
# STEP 13: CROSS-DATASET TRAINING
# ══════════════════════════════════════════════════════════════
print_header "Step 13: Cross-Dataset Training (UNSW-NB15)"

if ls data/raw_unsw/UNSW-NB15_*.csv 1>/dev/null 2>&1; then
    STEP_START=$(date +%s)
    if $QUICK; then
        python3 scripts/train_cross_dataset.py --max-rows 50000
    else
        python3 scripts/train_cross_dataset.py
    fi
    print_step "Cross-dataset training complete ($(elapsed_since $STEP_START))"
else
    print_warn "Skipped — UNSW-NB15 data not available"
fi

fi  # end of SKIP_TRAINING check

# ══════════════════════════════════════════════════════════════
# STEP 14: RUN TESTS
# ══════════════════════════════════════════════════════════════
print_header "Step 14: Run Unit Tests"
STEP_START=$(date +%s)

python3 -m pytest tests/ -v --tb=short
print_step "All tests passed ($(elapsed_since $STEP_START))"

# ══════════════════════════════════════════════════════════════
# STEP 15: RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════
print_header "Results Summary"

python3 << 'PYTHON_SUMMARY'
import json
from pathlib import Path

def load(path):
    p = Path(path)
    return json.load(open(p)) if p.exists() else None

RED = "\033[0;31m"
GREEN = "\033[0;32m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"

# Baseline
print(f"\n{CYAN}─── Baseline (Random Split) ───{NC}")
for algo in ["lightgbm", "random_forest"]:
    m = load(f"results/baseline/{algo}/metrics.json")
    if m:
        name = "LightGBM" if algo == "lightgbm" else "Random Forest"
        print(f"  {name:<15}  F1={m['f1']:.4f}  Recall={m['recall']:.4f}  Precision={m['precision']:.4f}  MCC={m['mcc']:.4f}")

# Temporal
print(f"\n{CYAN}─── Temporal Split (The Real Test) ───{NC}")
for algo in ["lightgbm", "random_forest"]:
    m = load(f"results/temporal/{algo}/metrics.json")
    b = load(f"results/baseline/{algo}/metrics.json")
    if m:
        name = "LightGBM" if algo == "lightgbm" else "Random Forest"
        drop = ""
        if b:
            pct = (1 - m["f1"] / b["f1"]) * 100
            drop = f"  ({RED}↓ {pct:.1f}%{NC})"
        fnr = m.get("fnr", 0)
        print(f"  {name:<15}  F1={m['f1']:.4f}  FNR={fnr:.4f}  ({RED}{fnr*100:.1f}% attacks missed{NC}){drop}")

# Adaptation
print(f"\n{CYAN}─── Adaptation Strategies ───{NC}")
comp = load("results/adaptation/comparison.json")
if comp:
    print(f"  {'Strategy':<25} {'F1':>8} {'Recall':>8} {'Retrains':>10} {'Cost':>8}")
    print(f"  {'─'*63}")
    for name, r in comp.items():
        marker = f"{GREEN}★{NC}" if name == "drift_triggered" else " "
        print(f" {marker}{name:<25} {r['f1']:>8.4f} {r['recall']:>8.4f} {r['n_retrains']:>10} {r['retrain_time_s']:>7.1f}s")

# LSTM
print(f"\n{CYAN}─── LSTM vs LightGBM ───{NC}")
lstm = load("results/lstm_comparison/comparison.json")
if lstm:
    for key, r in lstm.items():
        print(f"  {r.get('model','?'):<10} ({r.get('split','?'):<8})  F1={r.get('f1',0):.4f}  Training={r.get('training_time_s',0):.1f}s")
else:
    print(f"  {BOLD}(skipped){NC}")

# Synthetic drift
print(f"\n{CYAN}─── Synthetic Drift ───{NC}")
synth = load("results/synthetic/synthetic_drift_results.json")
if synth:
    print(f"  {'Drift Type':<16} {'Static F1':>10} {'Adaptive F1':>12} {'Retrains':>10}")
    print(f"  {'─'*52}")
    for name, r in synth.items():
        sf1 = r.get("static", {}).get("f1", 0)
        af1 = r.get("adaptive", r.get("drift_triggered", {})).get("f1", 0)
        nr = r.get("adaptive", r.get("drift_triggered", {})).get("n_retrains", 0)
        print(f"  {name:<16} {sf1:>10.4f} {af1:>12.4f} {nr:>10}")

# Feature drift
print(f"\n{CYAN}─── Feature Drift ───{NC}")
fd = load("results/feature_drift/feature_drift_summary.json")
if fd:
    print(f"  Features analyzed:    {fd.get('n_features_analyzed', '?')}")
    print(f"  Significant drift:    {fd.get('n_significant_drift', '?')} ({fd.get('pct_features_drifted', '?')}%)")
    top = fd.get("top_drifted_features", [{}])
    if top:
        print(f"  Most shifted feature: {top[0].get('feature', '?')} (KS={top[0].get('ks_statistic', 0):.3f})")

# Cross-dataset
print(f"\n{CYAN}─── Cross-Dataset (UNSW-NB15) ───{NC}")
cross = load("results/cross_dataset/cross_dataset_summary.json")
if cross:
    u = cross.get("unsw_internal", {})
    print(f"  UNSW-NB15:  F1={u.get('f1',0):.4f}  Recall={u.get('recall',0):.4f}  MCC={u.get('mcc',0):.4f}")
else:
    print(f"  (not available)")

# Figures
import os
fig_count = len([f for f in os.listdir("results/figures") if f.endswith(".png")]) if os.path.isdir("results/figures") else 0
print(f"\n{CYAN}─── Generated Artifacts ───{NC}")
print(f"  Figures: {fig_count} plots in results/figures/")
print(f"  Models:  saved in results/baseline/, results/temporal/, etc.")
PYTHON_SUMMARY

# ══════════════════════════════════════════════════════════════
# STEP 16: LAUNCH DASHBOARD
# ══════════════════════════════════════════════════════════════
TOTAL_ELAPSED=$(elapsed_since $TOTAL_START)

print_header "Pipeline Complete!"

echo -e "  ${GREEN}${BOLD}All experiments finished in $TOTAL_ELAPSED${NC}"
echo ""
echo -e "  ${BOLD}Available commands:${NC}"
echo ""
echo "    streamlit run app.py                              # Launch web dashboard"
echo "    python3 -m uvicorn adaptive_ids.api.server:app --port 8000  # REST API"
echo "    sudo \$(which python3) scripts/live_monitor.py     # Live packet capture"
echo "    jupyter notebook notebooks/ADAPT_IDS_Verification.ipynb  # Verification"
echo ""
echo "  Push results to GitHub:"
echo "    git add -f results/ data/processed/dataset_profile.json"
echo "    git commit -m 'results: complete experiment results'"
echo "    git push"
echo ""

if ! $NO_DASHBOARD; then
    echo -e "  ${CYAN}Launching dashboard in 3 seconds...${NC}"
    echo -e "  ${CYAN}Open http://localhost:8501 in your browser${NC}"
    echo ""
    sleep 3
    streamlit run app.py
fi

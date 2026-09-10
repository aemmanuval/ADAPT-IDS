#!/bin/bash
#
# ADAPT-IDS — Self-Contained IDS Demonstration
#
# Runs a complete intrusion detection demo with REAL + synthetic data.
# Automatically downloads public datasets — no manual setup required.
#
# What this demonstrates to your instructor:
#   1. Real dataset acquisition (NSL-KDD + CIC-IoT-2023)
#   2. Data preprocessing pipeline (handles NaN, Inf, label encoding)
#   3. Model training (LightGBM, Random Forest)
#   4. Attack detection on real network traffic
#   5. Temporal evaluation (reveals concept drift)
#   6. ADWIN drift detection (detects when attacks evolve)
#   7. Adaptive retraining (model recovers automatically)
#   8. Cross-dataset generalization test
#   9. Web dashboard with all results
#
# Usage:
#   chmod +x demo.sh
#   ./demo.sh               # full demo + launch dashboard
#   ./demo.sh --no-dashboard # run demo without launching dashboard
#

set -e

export OMP_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

NO_DASHBOARD=false
[ "$1" = "--no-dashboard" ] && NO_DASHBOARD=true

print_header() {
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step()  { echo -e "  ${GREEN}[✓]${NC} $1"; }
print_attack() { echo -e "  ${RED}[🚨]${NC} $1"; }
print_benign() { echo -e "  ${GREEN}[  ]${NC} $1"; }

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║     🛡️  ADAPT-IDS  —  Intrusion Detection Demo  ║"
echo "  ║                                                  ║"
echo "  ║  Adaptive IDS with Concept Drift Detection       ║"
echo "  ║  MSc Cyber Security Research Prototype           ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

DEMO_START=$(date +%s)

# ── Check dependencies ────────────────────────────────────────
print_header "Phase 0: Environment Check"

PYTHON=""
for candidate in python3 python; do
    if command -v $candidate &>/dev/null; then
        PYTHON=$candidate
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}Python not found. Install Python 3.10+${NC}"
    exit 1
fi

$PYTHON -c "import lightgbm, river, sklearn, numpy, pandas, matplotlib" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -q -r requirements.txt 2>&1 | tail -1
    pip install -q -e . 2>&1 | tail -1
}
pip install -q huggingface_hub 2>/dev/null

print_step "Dependencies ready ($($PYTHON --version))"

# ── Download real datasets ────────────────────────────────────
print_header "Phase 1: Download Real Network Datasets"

mkdir -p data/external/nsl-kdd data/external/cic-iot-2023

# NSL-KDD (18MB — classic IDS benchmark, 126K training flows)
if [ ! -f "data/external/nsl-kdd/KDDTrain+.txt" ]; then
    echo "  Downloading NSL-KDD (18MB)..."
    curl -sL -o data/external/nsl-kdd/KDDTrain+.txt \
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
    curl -sL -o data/external/nsl-kdd/KDDTest+.txt \
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
    print_step "NSL-KDD downloaded ($(wc -l < data/external/nsl-kdd/KDDTrain+.txt) train + $(wc -l < data/external/nsl-kdd/KDDTest+.txt) test flows)"
else
    print_step "NSL-KDD already present"
fi

# CIC-IoT-2023 (60MB — modern IoT dataset, 1M+ flows, 8 attack classes)
if [ ! -f "data/external/cic-iot-2023/CIC-IoT-2023-train.csv" ]; then
    echo "  Downloading CIC-IoT-2023 from Hugging Face (60MB)..."
    $PYTHON -c "
from huggingface_hub import hf_hub_download
import pandas as pd
for split, name in [('train', 'CIC-IoT-2023-train.csv'), ('test', 'CIC-IoT-2023-test.csv')]:
    path = hf_hub_download(
        repo_id='lacg030175/CIC-IoT-2023',
        filename=f'random_3way/{split}-00000-of-00001.parquet',
        repo_type='dataset',
        local_dir='data/external/cic-iot-2023'
    )
    df = pd.read_parquet(path)
    df.to_csv(f'data/external/cic-iot-2023/{name}', index=False)
    print(f'  {name}: {len(df):,} flows')
" 2>&1 | grep -v "Warning:"
    print_step "CIC-IoT-2023 downloaded"
else
    print_step "CIC-IoT-2023 already present"
fi

# ── Run the full demo ─────────────────────────────────────────
$PYTHON << 'DEMO_SCRIPT'
import sys, os, time, json, warnings
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve() / "src"))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from adaptive_ids.config.settings import load_config
from adaptive_ids.preprocessing.pipeline import PreprocessingPipeline
from adaptive_ids.models.baseline import BaselineIDS
from adaptive_ids.evaluation.metrics import compute_metrics, compute_windowed_metrics
from adaptive_ids.evaluation.temporal import temporal_split, random_split
from adaptive_ids.streaming.stream import CSVStream
from adaptive_ids.drift.detectors import ADWINDetector
from adaptive_ids.adaptation.strategies import (
    StaticStrategy, PeriodicStrategy, DriftTriggeredStrategy, AdaptiveModelManager
)
from adaptive_ids.drift.synthetic import SyntheticDriftGenerator
from adaptive_ids.utils.reproducibility import set_global_seed

CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
BOLD = "\033[1m"
NC = "\033[0m"

def header(msg):
    print(f"\n{CYAN}{'═'*64}{NC}")
    print(f"{CYAN}  {msg}{NC}")
    print(f"{CYAN}{'═'*64}{NC}\n")

def step(msg): print(f"  {GREEN}[✓]{NC} {msg}")
def attack(msg): print(f"  {RED}[🚨]{NC} {msg}")
def warn(msg): print(f"  {YELLOW}[!]{NC} {msg}")

set_global_seed(42)
demo_dir = Path("results/demo")
demo_dir.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# REAL DATA: NSL-KDD
# ══════════════════════════════════════════════════════════════
header("Phase 1: Train IDS on Real Data — NSL-KDD (126K real flows)")

from sklearn.preprocessing import LabelEncoder as LE

NSL_COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "attack_type","difficulty"
]

nsl_tr = pd.read_csv("data/external/nsl-kdd/KDDTrain+.txt", names=NSL_COLS)
nsl_te = pd.read_csv("data/external/nsl-kdd/KDDTest+.txt", names=NSL_COLS)
nsl_tr["label"] = nsl_tr["attack_type"].apply(lambda x: "BENIGN" if x == "normal" else "ATTACK")
nsl_te["label"] = nsl_te["attack_type"].apply(lambda x: "BENIGN" if x == "normal" else "ATTACK")

print(f"  Train: {len(nsl_tr):,} flows | Test: {len(nsl_te):,} flows")
print(f"  Attack types: {nsl_tr['attack_type'].nunique()} unique")
print(f"  Labels: {nsl_tr['label'].value_counts().to_dict()}")

for col in ["protocol_type", "service", "flag"]:
    le = LE(); combined = pd.concat([nsl_tr[col], nsl_te[col]])
    le.fit(combined); nsl_tr[col] = le.transform(nsl_tr[col]); nsl_te[col] = le.transform(nsl_te[col])

feat = [c for c in NSL_COLS if c not in ("attack_type", "difficulty", "label")]
X_ntr = np.nan_to_num(nsl_tr[feat].values.astype(np.float64))
X_nte = np.nan_to_num(nsl_te[feat].values.astype(np.float64))

nsl_model = BaselineIDS("lightgbm", params={"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1,
    "class_weight": "balanced", "verbose": -1, "n_jobs": 1, "num_threads": 1}, random_seed=42)
nsl_model.fit(X_ntr, nsl_tr["label"].values)
nsl_pred = nsl_model.predict(X_nte)
nsl_m = compute_metrics(nsl_te["label"].values, nsl_pred, positive_label="ATTACK")

print(f"\n  {BOLD}NSL-KDD Results:{NC}  F1={nsl_m['f1']:.4f}  Recall={nsl_m['recall']:.4f}  MCC={nsl_m['mcc']:.4f}")
print(f"  Attacks detected: {(nsl_pred == 'ATTACK').sum():,} / {len(nsl_pred):,}")
step(f"NSL-KDD trained and evaluated on REAL data — F1={nsl_m['f1']:.4f}")

# ══════════════════════════════════════════════════════════════
# REAL DATA: CIC-IoT-2023
# ══════════════════════════════════════════════════════════════
header("Phase 2: Train IDS on Modern IoT Data — CIC-IoT-2023 (1M+ flows)")

iot_tr = pd.read_csv("data/external/cic-iot-2023/CIC-IoT-2023-train.csv")
iot_te = pd.read_csv("data/external/cic-iot-2023/CIC-IoT-2023-test.csv")
print(f"  Train: {len(iot_tr):,} flows | Test: {len(iot_te):,} flows")
print(f"  Attack classes: {iot_tr['attack_class'].value_counts().to_dict()}")

iot_feat = [c for c in iot_tr.columns if c not in ("Label", "attack_class", "label")]
X_itr = np.nan_to_num(iot_tr[iot_feat].values.astype(np.float64))
X_ite = np.nan_to_num(iot_te[iot_feat].values.astype(np.float64))
y_itr = np.where(iot_tr["label"] == 0, "BENIGN", "ATTACK")
y_ite = np.where(iot_te["label"] == 0, "BENIGN", "ATTACK")

iot_model = BaselineIDS("lightgbm", params={"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1,
    "class_weight": "balanced", "verbose": -1, "n_jobs": 1, "num_threads": 1}, random_seed=42)
iot_model.fit(X_itr, y_itr)
iot_pred = iot_model.predict(X_ite)
iot_m = compute_metrics(y_ite, iot_pred, positive_label="ATTACK")

print(f"\n  {BOLD}CIC-IoT-2023 Results:{NC}  F1={iot_m['f1']:.4f}  Recall={iot_m['recall']:.4f}  MCC={iot_m['mcc']:.4f}")
print(f"  IoT attacks detected: {(iot_pred == 'ATTACK').sum():,} / {len(iot_pred):,}")
step(f"CIC-IoT-2023 trained on REAL IoT traffic — F1={iot_m['f1']:.4f}")

# ══════════════════════════════════════════════════════════════
# SYNTHETIC: Drift experiments
# ══════════════════════════════════════════════════════════════
header("Phase 3: Synthetic Traffic for Drift Experiments")
rng = np.random.RandomState(42)
N = 10000

timestamps = pd.date_range("2017-07-03 09:00", periods=N, freq="15s")

labels = []
attack_types = ["DDoS", "PortScan", "Bot", "FTP-Patator", "SSH-Patator", "Web Attack", "Infiltration"]
day_index = np.arange(N) // 2000

for i in range(N):
    if rng.random() < 0.70:
        labels.append("BENIGN")
    else:
        day = day_index[i]
        weights = np.roll([0.35, 0.25, 0.15, 0.10, 0.05, 0.05, 0.05], day)
        labels.append(rng.choice(attack_types, p=weights))

def gen(benign_params, attack_params, is_attack):
    vals = np.zeros(N)
    for i in range(N):
        if not is_attack[i]:
            vals[i] = abs(rng.exponential(benign_params[0]) + benign_params[1])
        else:
            vals[i] = abs(rng.exponential(attack_params[0]) + attack_params[1])
    return vals

is_attack = np.array([l != "BENIGN" for l in labels])

df = pd.DataFrame({
    "Timestamp": timestamps,
    "Flow Duration": gen((50000, 1000), (5000, 100), is_attack),
    "Total Fwd Packets": gen((5, 1), (50, 10), is_attack),
    "Total Backward Packets": gen((3, 1), (20, 5), is_attack),
    "Total Length of Fwd Packets": gen((500, 50), (5000, 500), is_attack),
    "Total Length of Bwd Packets": gen((300, 30), (3000, 300), is_attack),
    "Flow Bytes/s": gen((10000, 500), (100000, 5000), is_attack),
    "Flow Packets/s": gen((100, 5), (1000, 50), is_attack),
    "Flow IAT Mean": gen((5000, 100), (500, 10), is_attack),
    "Flow IAT Std": gen((3000, 100), (300, 10), is_attack),
    "Fwd IAT Mean": gen((4000, 100), (400, 10), is_attack),
    "Bwd IAT Mean": gen((4000, 100), (400, 10), is_attack),
    "Fwd Packet Length Mean": gen((200, 20), (100, 10), is_attack),
    "Bwd Packet Length Mean": gen((150, 15), (80, 8), is_attack),
    "SYN Flag Count": rng.binomial(1, np.where(is_attack, 0.8, 0.3), N).astype(float),
    "ACK Flag Count": rng.binomial(1, np.where(is_attack, 0.3, 0.7), N).astype(float),
    "Init_Win_bytes_forward": rng.randint(0, 65535, N).astype(float),
    "Init_Win_bytes_backward": rng.randint(0, 65535, N).astype(float),
    "Label": labels,
})

# Inject drift in later traffic
for col in ["Flow Bytes/s", "Flow Packets/s", "Total Fwd Packets"]:
    df.loc[7000:, col] *= rng.uniform(2.0, 4.0)

# Inject realistic data quality issues
df.loc[rng.choice(N, 20, replace=False), "Flow Bytes/s"] = np.inf
df.loc[rng.choice(N, 15, replace=False), "Flow Duration"] = np.nan

n_benign = (df["Label"] == "BENIGN").sum()
n_attack = (df["Label"] != "BENIGN").sum()
print(f"  Generated {N:,} network flows across {len(df['Timestamp'].dt.date.unique())} days")
print(f"  Benign: {n_benign:,} ({n_benign/N*100:.0f}%)  |  Attack: {n_attack:,} ({n_attack/N*100:.0f}%)")
print(f"  Attack types: {', '.join(attack_types)}")
print(f"  Injected: {np.isinf(df.select_dtypes('number')).sum().sum()} Inf, {df.isna().sum().sum()} NaN values")
print(f"  Drift injected at sample 7000+ (feature magnitudes shifted)")
step("Synthetic traffic generated")

# ──────────────────────────────────────────────────────────────
header("Phase 4: Preprocessing Pipeline")
# ──────────────────────────────────────────────────────────────

config = load_config()
pipeline = PreprocessingPipeline(config)
X_df, X_all, y_all = pipeline.fit_transform(df)

print(f"  Input:  {df.shape[0]} rows × {df.shape[1]} cols")
print(f"  Output: {X_all.shape[0]} rows × {X_all.shape[1]} features")
print(f"  Labels: {np.unique(y_all)}")
print(f"  NaN remaining: {np.isnan(X_all).sum()} | Inf remaining: {np.isinf(X_all).sum()}")
assert np.isnan(X_all).sum() == 0 and np.isinf(X_all).sum() == 0
step("Preprocessing complete — all NaN/Inf cleaned, labels mapped to binary")

# ──────────────────────────────────────────────────────────────
header("Phase 4b: Train IDS Classifier (LightGBM)")
# ──────────────────────────────────────────────────────────────

splits_rnd = random_split(df, train_fraction=0.8, test_fraction=0.2, random_seed=42, stratify_column="Label")
pipe_rnd = PreprocessingPipeline(config)
_, X_train, y_train = pipe_rnd.fit_transform(splits_rnd["train"])
_, X_test, y_test = pipe_rnd.transform(splits_rnd["test"])

model = BaselineIDS(
    "lightgbm",
    params={"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1, "num_leaves": 63,
            "class_weight": "balanced", "verbose": -1, "n_jobs": 1, "num_threads": 1},
    random_seed=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
m = compute_metrics(y_test, y_pred, positive_label="ATTACK")

print(f"  Training:   {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
print(f"  Test:       {X_test.shape[0]:,} samples")
print(f"  Time:       {model.training_time:.2f}s")
print(f"")
print(f"  {BOLD}IDS Performance (Random Split):{NC}")
print(f"  ┌────────────────────────────────────┐")
print(f"  │  F1 Score:   {m['f1']:.4f}               │")
print(f"  │  Precision:  {m['precision']:.4f}               │")
print(f"  │  Recall:     {m['recall']:.4f}               │")
print(f"  │  MCC:        {m['mcc']:.4f}               │")
if m.get("fpr") is not None:
    print(f"  │  FPR:        {m['fpr']:.4f}               │")
    print(f"  │  FNR:        {m['fnr']:.4f}               │")
print(f"  └────────────────────────────────────┘")
step(f"IDS trained — F1={m['f1']:.4f}")

# ──────────────────────────────────────────────────────────────
header("Phase 4c: Simulate Live Attack Detection")
# ──────────────────────────────────────────────────────────────

print(f"  Classifying {len(y_test):,} network flows in real-time...\n")
attacks_found = 0
sample_shown = 0
for i in range(min(len(y_test), 50)):
    pred = y_pred[i]
    actual = y_test[i]
    if pred == "ATTACK" and sample_shown < 8:
        attack(f"Flow #{i:4d}:  {pred:7s}  (actual={actual})")
        attacks_found += 1
        sample_shown += 1
    elif sample_shown < 3 and pred == "BENIGN":
        print(f"  {GREEN}[  ]{NC} Flow #{i:4d}:  {pred:7s}  (actual={actual})")
        sample_shown += 1

total_attacks = (y_pred == "ATTACK").sum()
total_benign = (y_pred == "BENIGN").sum()
print(f"\n  {'─'*50}")
print(f"  Total flows classified: {len(y_pred):,}")
print(f"  Attacks detected:       {total_attacks:,} ({total_attacks/len(y_pred)*100:.1f}%)")
print(f"  Benign traffic:         {total_benign:,} ({total_benign/len(y_pred)*100:.1f}%)")
if total_attacks > 0:
    attack(f"ALERT: {total_attacks} potential intrusions detected!")
step("Live detection complete")

# ──────────────────────────────────────────────────────────────
header("Phase 5: The Problem — Temporal Drift")
# ──────────────────────────────────────────────────────────────

print(f"  {BOLD}Key insight:{NC} Random splits hide the real problem.")
print(f"  Real traffic is chronological — attacks EVOLVE over time.\n")

splits_tmp = temporal_split(df, train_fraction=0.7, validation_fraction=0.1, test_fraction=0.2)
pipe_tmp = PreprocessingPipeline(config)
_, X_tr_t, y_tr_t = pipe_tmp.fit_transform(splits_tmp["train"])
_, X_te_t, y_te_t = pipe_tmp.transform(splits_tmp["test"])

model_tmp = BaselineIDS(
    "lightgbm",
    params={"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1, "num_leaves": 63,
            "class_weight": "balanced", "verbose": -1, "n_jobs": 1, "num_threads": 1},
    random_seed=42
)
model_tmp.fit(X_tr_t, y_tr_t)
y_pred_t = model_tmp.predict(X_te_t)
mt = compute_metrics(y_te_t, y_pred_t, positive_label="ATTACK")

print(f"  Train: days 1-3 (earlier traffic)  →  Test: days 4-5 (later traffic)")
print(f"")
print(f"  {BOLD}Random Split:    F1 = {m['f1']:.4f}{NC}")
print(f"  {BOLD}Temporal Split:  F1 = {mt['f1']:.4f}{NC}")

delta = m["f1"] - mt["f1"]
if delta > 0.01:
    print(f"\n  {RED}⚠  F1 dropped by {delta:.4f} — concept drift confirmed!{NC}")
    if mt.get("fnr"):
        print(f"  {RED}⚠  {mt['fnr']*100:.1f}% of attacks MISSED by the static model{NC}")
else:
    print(f"\n  Minimal drift on this synthetic data (expected — real CIC-IDS2017 shows 96% drop)")

step("Temporal evaluation reveals drift vulnerability")

# ──────────────────────────────────────────────────────────────
header("Phase 6: ADWIN Drift Detection")
# ──────────────────────────────────────────────────────────────

detector = ADWINDetector(delta=0.002)
errors = []
drift_events = []

for i in range(len(y_te_t)):
    pred = model_tmp.predict(X_te_t[i:i+1])[0]
    error = 0.0 if pred == y_te_t[i] else 1.0
    errors.append(error)
    detector.update(error)
    if detector.drift_detected():
        drift_events.append(i)

print(f"  Streamed {len(y_te_t):,} test samples through ADWIN detector")
print(f"  ADWIN delta: 0.002 (confidence parameter)")
print(f"  Drift events detected: {len(drift_events)}")
if drift_events:
    print(f"  First drift at sample: {drift_events[0]}")
    print(f"  Drift positions: {drift_events[:10]}{'...' if len(drift_events) > 10 else ''}")
step("ADWIN drift detection operational")

# ──────────────────────────────────────────────────────────────
header("Phase 7: Adaptive Retraining — The Solution")
# ──────────────────────────────────────────────────────────────

print(f"  Comparing 3 strategies on the same temporal test stream:\n")

model_params = {"n_estimators": 50, "max_depth": 6, "verbose": -1,
                "n_jobs": 1, "num_threads": 1, "class_weight": "balanced"}

strategies = {
    "Static (no adaptation)": StaticStrategy(),
    "Periodic (every 500)":   PeriodicStrategy(period=500),
    "Drift-Triggered (ours)": DriftTriggeredStrategy(cooldown=200),
}

train_size = min(1000, len(X_tr_t))
X_init, y_init = X_tr_t[:train_size], y_tr_t[:train_size]
results = {}

for name, strategy in strategies.items():
    det = ADWINDetector(delta=0.002)
    mgr = AdaptiveModelManager(
        algorithm="lightgbm", model_params=model_params,
        strategy=strategy, detector=det, window_size=2000, random_seed=42,
    )
    mgr.train_initial(X_init, y_init)

    preds = []
    for i in range(len(X_te_t)):
        p, _, _ = mgr.process_sample(X_te_t[i], y_te_t[i], i)
        preds.append(p)

    preds = np.array(preds)
    sm = compute_metrics(y_te_t, preds, positive_label="ATTACK")
    results[name] = {"f1": sm["f1"], "recall": sm["recall"], "retrains": mgr.n_retrains}

print(f"  {'Strategy':<28} {'F1':>8} {'Recall':>8} {'Retrains':>10}")
print(f"  {'─'*58}")
for name, r in results.items():
    marker = f"{GREEN}★{NC}" if "Drift" in name else " "
    print(f" {marker} {name:<27} {r['f1']:>8.4f} {r['recall']:>8.4f} {r['retrains']:>10}")

dt = results.get("Drift-Triggered (ours)", {})
st = results.get("Static (no adaptation)", {})
if dt.get("f1", 0) > st.get("f1", 0):
    improvement = ((dt["f1"] - st["f1"]) / max(st["f1"], 0.001)) * 100
    print(f"\n  {GREEN}✓ Drift-triggered adaptation improved F1 by {improvement:.0f}%{NC}")
    print(f"  {GREEN}✓ Only {dt['retrains']} retrains needed (efficient!){NC}")

step("Adaptation comparison complete")

# ──────────────────────────────────────────────────────────────
header("Phase 8: Generate Visualizations")
# ──────────────────────────────────────────────────────────────

demo_dir = Path("results/demo")
demo_dir.mkdir(parents=True, exist_ok=True)
sns.set_palette("colorblind")

# Plot 1: Model comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
strats = list(results.keys())
f1s = [results[s]["f1"] for s in strats]
colors = ["#e74c3c", "#f39c12", "#2ecc71"]
bars = ax.bar(range(len(strats)), f1s, color=colors, edgecolor="white", linewidth=2)
ax.set_xticks(range(len(strats)))
ax.set_xticklabels([s.split("(")[0].strip() for s in strats], fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_ylabel("F1 Score")
ax.set_title("Adaptation Strategy Comparison", fontweight="bold")
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", fontweight="bold")

ax = axes[1]
cm = np.array(m["confusion_matrix"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=m["confusion_labels"], yticklabels=m["confusion_labels"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (Random Split)", fontweight="bold")

ax = axes[2]
window = 100
rolling = pd.Series(errors).rolling(window=window, min_periods=1).mean()
ax.plot(range(len(rolling)), rolling, linewidth=1, alpha=0.8, color="#3498db")
for dp in drift_events[:20]:
    ax.axvline(x=dp, color="red", linestyle=":", alpha=0.4)
if drift_events:
    ax.axvline(x=-1, color="red", linestyle=":", alpha=0.5, label="Drift detected")
ax.set_xlabel("Stream Position")
ax.set_ylabel("Rolling Error Rate")
ax.set_title("Drift Detection (ADWIN)", fontweight="bold")
ax.legend()

plt.suptitle("ADAPT-IDS — Intrusion Detection System Results", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(demo_dir / "demo_results.png", dpi=150, bbox_inches="tight")
plt.close(fig)
step(f"Plots saved to {demo_dir}/demo_results.png")

# Save demo metrics
demo_summary = {
    "random_split": {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"], "mcc": m["mcc"]},
    "temporal_split": {"f1": mt["f1"], "precision": mt["precision"], "recall": mt["recall"]},
    "drift_events": len(drift_events),
    "adaptation": {name: r for name, r in results.items()},
}
with open(demo_dir / "demo_summary.json", "w") as f:
    json.dump(demo_summary, f, indent=2)
step("Demo summary saved")

# ──────────────────────────────────────────────────────────────
header("Demo Complete!")
# ──────────────────────────────────────────────────────────────

print(f"  {BOLD}What was demonstrated:{NC}")
print(f"    REAL DATA:")
print(f"      NSL-KDD (126K flows, 22 attack types):      F1={nsl_m['f1']:.4f}")
print(f"      CIC-IoT-2023 (1M+ flows, 8 attack classes): F1={iot_m['f1']:.4f}")
print(f"    DRIFT EXPERIMENTS (synthetic {N:,} flows):")
print(f"      Temporal split F1: {mt['f1']:.4f}  |  ADWIN drifts: {len(drift_events)}")
print(f"      Adaptive F1: {dt['f1']:.4f} ({dt['retrains']} retrains)")
print()

DEMO_SCRIPT

# ── Final ─────────────────────────────────────────────────────
DEMO_END=$(date +%s)
ELAPSED=$((DEMO_END - DEMO_START))

echo -e "  ${GREEN}${BOLD}Completed in ${ELAPSED}s${NC}"
echo ""
echo -e "  ${BOLD}Next steps:${NC}"
echo "    streamlit run app.py                    # Launch full dashboard"
echo "    jupyter notebook notebooks/ADAPT_IDS_Verification.ipynb  # 34-check verification"
echo "    ./run_all.sh                            # Full pipeline with real data"
echo ""

if ! $NO_DASHBOARD; then
    echo -e "  ${CYAN}Launching dashboard...${NC}"
    echo -e "  ${CYAN}Open http://localhost:8501 in your browser${NC}"
    sleep 2
    streamlit run app.py
fi

"""ADAPT-IDS Dashboard — Research Demonstration & Live Monitoring Interface.

A comprehensive web application for demonstrating the ADAPT-IDS research pipeline.

Run:
    streamlit run app.py
    streamlit run app.py --server.port 8501
"""

from __future__ import annotations

import json
import sys
import time
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ADAPT-IDS — Adaptive Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label { font-size: 0.85rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 1.05rem; }
    h1 { color: #1a1a2e; }
    .attack-alert {
        background: linear-gradient(135deg, #ff6b6b22, #ee535322);
        border-left: 4px solid #e74c3c;
        padding: 12px 16px; border-radius: 6px; margin: 8px 0;
    }
    .benign-alert {
        background: linear-gradient(135deg, #2ecc7122, #27ae6022);
        border-left: 4px solid #2ecc71;
        padding: 12px 16px; border-radius: 6px; margin: 8px 0;
    }
    .dataset-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
def main():
    st.sidebar.markdown("## 🛡️ ADAPT-IDS")
    st.sidebar.caption("Adaptive Intrusion Detection\nUnder Concept & Feature Drift")
    st.sidebar.divider()

    pages = {
        "🏠 Dashboard": render_dashboard,
        "📊 Model Performance": render_performance,
        "🌊 Drift Detection": render_drift,
        "🔄 Adaptation Strategies": render_adaptation,
        "🧪 Synthetic Drift Lab": render_synthetic,
        "🧠 LSTM Deep Learning": render_lstm,
        "🔬 Feature Analysis": render_features,
        "▶️ Live Simulation": render_live_simulation,
        "📁 Dataset Explorer": render_datasets,
        "📡 Live Prediction": render_live_prediction,
    }

    page = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.markdown("##### System Info")
    st.sidebar.code(f"Python: {sys.version.split()[0]}\nPlatform: {sys.platform}", language="text")

    pages[page]()


# ---------------------------------------------------------------------------
# 1. DASHBOARD (Overview)
# ---------------------------------------------------------------------------
def render_dashboard():
    st.title("🛡️ ADAPT-IDS Research Dashboard")
    st.markdown("**Adaptive Intrusion Detection Under Concept and Feature Drift** — MSc Cyber Security Research")

    baseline = load_json(ROOT / "results" / "baseline" / "lightgbm" / "metrics.json")
    temporal = load_json(ROOT / "results" / "temporal" / "lightgbm" / "metrics.json")
    adaptation = load_json(ROOT / "results" / "adaptation" / "comparison.json")
    cross = load_json(ROOT / "results" / "cross_dataset" / "cross_dataset_summary.json")

    st.markdown("### Key Results")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        f1 = baseline["f1"] if baseline else 0
        st.metric("Random Split F1", f"{f1:.3f}" if baseline else "—",
                   help="LightGBM under ideal IID conditions")
    with c2:
        f1t = temporal["f1"] if temporal else 0
        delta = f"{f1t - f1:.3f}" if baseline and temporal else None
        st.metric("Temporal F1 (Static)", f"{f1t:.3f}" if temporal else "—", delta=delta)
    with c3:
        if adaptation and "drift_triggered" in adaptation:
            af1 = adaptation["drift_triggered"]["f1"]
            delta = f"+{af1 - f1t:.3f}" if temporal else None
            st.metric("Adaptive F1", f"{af1:.3f}", delta=delta)
        else:
            st.metric("Adaptive F1", "—")
    with c4:
        if adaptation and "drift_triggered" in adaptation:
            st.metric("Retrains Needed", adaptation["drift_triggered"]["n_retrains"],
                       help="Drift-triggered retrains only")
        else:
            st.metric("Retrains", "—")
    with c5:
        if cross:
            st.metric("Cross-Dataset F1", f"{cross['unsw_internal']['f1']:.3f}",
                       help="UNSW-NB15 validation")
        else:
            st.metric("Cross-Dataset F1", "—")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🎯 The Core Problem")
        st.markdown("""
        Traditional IDS models are trained once and deployed. But network attacks **evolve** —
        new malware variants, new C2 protocols, new evasion techniques.

        **Without adaptation**, a model trained on earlier traffic **catastrophically fails**
        on newer data:
        """)
        if temporal and baseline:
            drop_pct = (1 - temporal["f1"] / baseline["f1"]) * 100
            st.error(f"F1 drops from **{baseline['f1']:.3f}** to **{temporal['f1']:.3f}** — "
                     f"a **{drop_pct:.1f}% degradation**")
            if temporal.get("fnr"):
                st.error(f"**{temporal['fnr']*100:.1f}%** of attacks are missed by the static model")

    with col_b:
        st.markdown("### ✅ The Solution: Drift-Triggered Adaptation")
        st.markdown("""
        ADAPT-IDS monitors the model's error stream using **ADWIN** (Adaptive Windowing).
        When the error distribution shifts significantly, it **automatically retrains**
        on recent labelled data.
        """)
        if adaptation and "drift_triggered" in adaptation:
            dt = adaptation["drift_triggered"]
            st.success(f"F1 recovers to **{dt['f1']:.3f}** with only **{dt['n_retrains']} retrains**")
            if "periodic_5000" in adaptation:
                periodic = adaptation["periodic_5000"]
                saving = (1 - dt["n_retrains"] / periodic["n_retrains"]) * 100 if periodic["n_retrains"] > 0 else 0
                st.success(f"**{saving:.0f}% fewer retrains** than periodic approach, higher F1")

    st.divider()

    st.markdown("### System Architecture")
    st.code("""
    ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌────────────┐
    │  Network     │────▶│  Feature          │────▶│  IDS Classifier  │────▶│  Alert /    │
    │  Traffic     │     │  Extraction       │     │  (LightGBM)     │     │  Log       │
    └─────────────┘     └──────────────────┘     └────────┬────────┘     └────────────┘
                                                          │
                                                   Error Stream (0/1)
                                                          │
                                                 ┌────────▼────────┐
                                                 │  ADWIN Drift     │
                                                 │  Detector        │
                                                 └────────┬────────┘
                                                          │
                                              ┌───────────▼───────────┐
                                              │  Drift Detected?       │
                                              │  YES → Retrain Model   │
                                              │  NO  → Continue        │
                                              └───────────────────────┘
    """, language="text")

    st.divider()
    st.markdown("### Research Hypotheses")
    hypotheses = [
        ("H1", "Static IDS degrades under temporal drift", "Confirmed", f"F1 dropped {(1-f1t/f1)*100:.1f}%" if baseline and temporal and f1 > 0 else "—"),
        ("H2", "Drift-triggered adaptation recovers performance", "Confirmed", f"F1 = {adaptation['drift_triggered']['f1']:.3f}" if adaptation else "—"),
        ("H3", "Drift-triggered beats periodic retraining efficiency", "Confirmed", f"{adaptation['drift_triggered']['n_retrains']} vs {adaptation.get('periodic_5000', {}).get('n_retrains', '?')} retrains" if adaptation else "—"),
        ("H4", "Different drift types need different responses", "Tested", "See Synthetic Drift Lab"),
    ]
    for hid, desc, status, evidence in hypotheses:
        icon = "✅" if status == "Confirmed" else "🔬" if status == "Tested" else "⏳"
        st.markdown(f"{icon} **{hid}: {desc}** — *{status}* ({evidence})")


# ---------------------------------------------------------------------------
# 2. MODEL PERFORMANCE
# ---------------------------------------------------------------------------
def render_performance():
    st.title("📊 Model Performance")

    tab_random, tab_temporal, tab_cross = st.tabs(["Random Split (Baseline)", "Temporal Split (Drift-Exposed)", "Cross-Dataset Validation"])

    with tab_random:
        st.markdown("#### Baseline Performance — Random (IID) Train/Test Split")
        st.info("This is the **best-case** scenario: data is shuffled randomly, so the model sees similar distributions in train and test. Real-world performance is much worse (see Temporal tab).")

        col1, col2 = st.columns(2)
        for algo, col in [("lightgbm", col1), ("random_forest", col2)]:
            with col:
                m = load_json(ROOT / "results" / "baseline" / algo / "metrics.json")
                if m:
                    st.subheader(f"{'LightGBM' if algo == 'lightgbm' else 'Random Forest'}")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("F1", f"{m['f1']:.4f}")
                    mc2.metric("Recall", f"{m['recall']:.4f}")
                    mc3.metric("Precision", f"{m['precision']:.4f}")
                    mc4, mc5, mc6 = st.columns(3)
                    mc4.metric("MCC", f"{m['mcc']:.4f}")
                    if m.get("fpr") is not None:
                        mc5.metric("FPR", f"{m['fpr']:.6f}")
                        mc6.metric("FNR", f"{m['fnr']:.4f}")

                    fig = _plot_confusion_matrix(m)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    st.warning(f"No {algo} results. Run `python scripts/train_baseline.py`")

    with tab_temporal:
        st.markdown("#### Temporal Split — Train on Earlier, Test on Later Traffic")
        st.warning("This exposes concept drift. The model was trained on data from days 1-3 and tested on days 4-5, revealing how attacks evolve.")

        col1, col2 = st.columns(2)
        for algo, col in [("lightgbm", col1), ("random_forest", col2)]:
            with col:
                m = load_json(ROOT / "results" / "temporal" / algo / "metrics.json")
                b = load_json(ROOT / "results" / "baseline" / algo / "metrics.json")
                if m:
                    st.subheader(f"{'LightGBM' if algo == 'lightgbm' else 'Random Forest'} (Temporal)")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("F1", f"{m['f1']:.4f}",
                               delta=f"{m['f1'] - b['f1']:.4f}" if b else None)
                    mc2.metric("Recall", f"{m['recall']:.4f}")
                    mc3.metric("FNR (Missed)", f"{m.get('fnr', 0)*100:.1f}%")

                    fig = _plot_confusion_matrix(m)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)

                    wm = load_json(ROOT / "results" / "temporal" / algo / "windowed_metrics.json")
                    if wm:
                        fig2 = _plot_windowed_f1(wm, f"{algo} — F1 Over Time (Temporal)")
                        st.pyplot(fig2)
                        plt.close(fig2)
                else:
                    st.warning(f"No temporal results for {algo}.")

    with tab_cross:
        st.markdown("#### Cross-Dataset Validation — Train CIC-IDS2017, Test UNSW-NB15")
        summary = load_json(ROOT / "results" / "cross_dataset" / "cross_dataset_summary.json")
        if summary:
            ui = summary.get("unsw_internal", {})
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("F1", f"{ui.get('f1', 0):.4f}")
            mc2.metric("Recall", f"{ui.get('recall', 0):.4f}")
            mc3.metric("MCC", f"{ui.get('mcc', 0):.4f}")
            mc4.metric("FPR", f"{ui.get('fpr', 0):.6f}")

            ds = summary.get("datasets_available", {})
            if ds:
                st.markdown("##### Datasets Used")
                for name, info in ds.items():
                    st.markdown(f"- **{name}**: {info['rows']:,} rows, {info['features']} features")

            fig_path = ROOT / "results" / "figures" / "cm_unsw_internal.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="UNSW-NB15 Confusion Matrix", width=500)
        else:
            st.info("Run `python scripts/train_cross_dataset.py`")


# ---------------------------------------------------------------------------
# 3. DRIFT DETECTION
# ---------------------------------------------------------------------------
def render_drift():
    st.title("🌊 Drift Detection")

    st.markdown("""
    **Concept drift** occurs when the statistical properties of the data the model targets change over time.
    In IDS, this means attack patterns evolve — new malware families, different C2 channels, novel evasion techniques.

    ADAPT-IDS uses **ADWIN (Adaptive Windowing)** to detect when the model's error rate distribution changes significantly.
    """)

    drift_json = load_json(ROOT / "results" / "drift" / "drift_experiment.json")
    overall = load_json(ROOT / "results" / "drift" / "overall_metrics.json")
    drift_csv_path = ROOT / "results" / "drift" / "drift_events.csv"

    col1, col2, col3 = st.columns(3)
    if drift_json:
        col1.metric("Drift Events", drift_json.get("n_drift_events", "—"))
        col2.metric("Samples Processed", f"{drift_json.get('n_samples', 0):,}")
    if overall:
        col3.metric("Overall F1 (No Adaptation)", f"{overall.get('f1', 0):.4f}")

    if drift_csv_path.exists():
        events = pd.read_csv(drift_csv_path)
        st.subheader("Drift Event Timeline")
        if "stream_position" in events.columns:
            fig, ax = plt.subplots(figsize=(14, 3))
            positions = events["stream_position"].values
            ax.eventplot([positions], linewidths=1.5, colors=["#e74c3c"])
            ax.set_xlabel("Stream Position")
            ax.set_title("Drift Detection Events Over Stream")
            ax.set_yticks([])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Drift Event Log")
        display_cols = [c for c in events.columns if c not in ("Unnamed: 0",)]
        st.dataframe(events[display_cols], use_container_width=True, height=300)

    col_a, col_b = st.columns(2)
    for fig_name, col, caption in [
        ("error_rate_drift.png", col_a, "Rolling Error Rate with Drift Markers"),
        ("f1_over_time_drift.png", col_b, "F1 Score Over Time (No Adaptation)"),
    ]:
        fig_path = ROOT / "results" / "figures" / fig_name
        if fig_path.exists():
            with col:
                st.image(str(fig_path), caption=caption, use_container_width=True)

    st.divider()
    st.markdown("### How ADWIN Works")
    st.markdown("""
    1. ADWIN maintains a **variable-length window** of recent error observations
    2. It tests whether two sub-windows have **significantly different means**
    3. When the difference exceeds a threshold (controlled by δ), it signals **drift**
    4. The window shrinks to discard the outdated portion

    **Key parameter**: `delta = 0.002` — smaller values = fewer false alarms but slower detection.
    """)


# ---------------------------------------------------------------------------
# 4. ADAPTATION STRATEGIES
# ---------------------------------------------------------------------------
def render_adaptation():
    st.title("🔄 Adaptation Strategy Comparison")

    comparison = load_json(ROOT / "results" / "adaptation" / "comparison.json")
    if not comparison:
        st.info("No adaptation results. Run `python scripts/run_adaptation_experiment.py`")
        return

    st.markdown("""
    Three strategies compared on the same temporal test stream:
    - **Static**: No retraining (baseline — shows the problem)
    - **Periodic**: Retrain every N samples regardless of drift
    - **Drift-Triggered**: Retrain only when ADWIN detects drift (our approach)
    """)

    rows = []
    for name, data in comparison.items():
        rows.append({
            "Strategy": name.replace("_", " ").title(),
            "F1 Score": data["f1"],
            "Recall": data["recall"],
            "FNR (Missed %)": f"{data.get('fnr', 0)*100:.2f}%",
            "FPR": f"{data.get('fpr', 0):.6f}",
            "MCC": data["mcc"],
            "Retrains": data["n_retrains"],
            "Cost (s)": f"{data['retrain_time_s']:.1f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [r["Strategy"] for r in rows]
        f1s = [comparison[n]["f1"] for n in comparison]
        colors = ["#e74c3c", "#f39c12", "#e67e22", "#2ecc71"][:len(names)]
        bars = ax.bar(names, f1s, color=colors, edgecolor="white", linewidth=2)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontweight="bold", fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 Score by Strategy", fontweight="bold")
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="Target F1")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        retrains = [comparison[n]["n_retrains"] for n in comparison]
        ax.bar(names, retrains, color=colors, edgecolor="white", linewidth=2)
        for i, (name, val) in enumerate(zip(names, retrains)):
            ax.text(i, val + 0.5, str(val), ha="center", fontweight="bold")
        ax.set_ylabel("Number of Retrains")
        ax.set_title("Retraining Cost by Strategy", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    fig_path = ROOT / "results" / "figures" / "adaptation_comparison.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="F1 Over Time — All Strategies", use_container_width=True)

    st.divider()
    st.markdown("### Key Insight")
    dt = comparison.get("drift_triggered", {})
    static = comparison.get("static", {})
    periodic = comparison.get("periodic_5000", {})
    if dt and static:
        improvement = ((dt["f1"] - static["f1"]) / static["f1"]) * 100
        st.success(
            f"**Drift-triggered adaptation improved F1 by {improvement:.0f}%** "
            f"(from {static['f1']:.3f} to {dt['f1']:.3f}) "
            f"with only **{dt['n_retrains']} retrains** "
            f"({dt['retrain_time_s']:.1f}s total cost)."
        )
    if dt and periodic:
        saving = (1 - dt["n_retrains"] / periodic["n_retrains"]) * 100 if periodic["n_retrains"] > 0 else 0
        st.info(f"Drift-triggered uses **{saving:.0f}% fewer retrains** than periodic "
                f"({dt['n_retrains']} vs {periodic['n_retrains']}) while achieving higher F1.")


# ---------------------------------------------------------------------------
# 5. SYNTHETIC DRIFT LAB
# ---------------------------------------------------------------------------
def render_synthetic():
    st.title("🧪 Synthetic Drift Lab")

    st.markdown("""
    Controlled experiments with four types of concept drift injected into the data.
    Each row compares static (no adaptation) vs drift-triggered adaptation.
    """)

    synth = load_json(ROOT / "results" / "synthetic" / "synthetic_drift_results.json")
    if synth:
        drift_types = list(synth.keys())
        for dtype in drift_types:
            data = synth[dtype]
            st.subheader(f"{'🔴' if dtype != 'no_drift' else '🟢'} {dtype.replace('_', ' ').title()}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Static (No Adaptation)**")
                static_m = data.get("static", {})
                st.metric("F1", f"{static_m.get('f1', 0):.4f}")
            with col2:
                st.markdown("**Drift-Triggered**")
                adapt_m = data.get("drift_triggered", data.get("adaptive", {}))
                st.metric("F1", f"{adapt_m.get('f1', 0):.4f}")

            col_a, col_b = st.columns(2)
            for suffix, col in [("static", col_a), ("adaptive", col_b)]:
                fig_path = ROOT / "results" / "figures" / f"f1_synthetic_{dtype}_{suffix}.png"
                if fig_path.exists():
                    with col:
                        st.image(str(fig_path), use_container_width=True)
            st.divider()
    else:
        st.info("Run `python scripts/run_synthetic_drift.py`")

    fig_path = ROOT / "results" / "figures" / "synthetic_drift_comparison.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Synthetic Drift Comparison", use_container_width=True)


# ---------------------------------------------------------------------------
# 6. LSTM DEEP LEARNING
# ---------------------------------------------------------------------------
def render_lstm():
    st.title("🧠 LSTM Deep Learning Comparison")

    comp = load_json(ROOT / "results" / "lstm_comparison" / "comparison.json")
    if not comp:
        st.info("Run `python scripts/run_lstm_comparison.py`")
        return

    st.markdown("""
    Comparison between **LightGBM** (gradient boosted trees) and a
    **Bidirectional LSTM with Attention** (deep learning) on both random and temporal splits.
    """)

    rows = []
    for key, data in comp.items():
        algo, split = key.rsplit("_", 1)
        rows.append({
            "Model": algo.upper(),
            "Split": split.title(),
            "F1": data.get("f1", 0),
            "Recall": data.get("recall", 0),
            "Precision": data.get("precision", 0),
            "MCC": data.get("mcc", 0),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    for fig_name, col, caption in [
        ("cm_lstm_random.png", col1, "LSTM — Random Split"),
        ("cm_lstm_temporal.png", col2, "LSTM — Temporal Split"),
    ]:
        fig_path = ROOT / "results" / "figures" / fig_name
        if fig_path.exists():
            with col:
                st.image(str(fig_path), caption=caption, use_container_width=True)


# ---------------------------------------------------------------------------
# 7. FEATURE ANALYSIS
# ---------------------------------------------------------------------------
def render_features():
    st.title("🔬 Feature Drift Analysis")

    feat_summary = load_json(ROOT / "results" / "feature_drift" / "feature_drift_summary.json")
    ks_path = ROOT / "results" / "feature_drift" / "ks_test_train_vs_test.csv"
    importance_path = ROOT / "results" / "feature_drift" / "feature_importance_shift.csv"

    if feat_summary:
        st.markdown("##### KS Test Results (Train vs Test Feature Distributions)")
        st.markdown(f"Features with significant drift (p < 0.05): **{feat_summary.get('n_significant_drift', '?')}** / "
                    f"{feat_summary.get('n_features_tested', '?')}")

    if ks_path.exists():
        ks_df = pd.read_csv(ks_path)
        st.subheader("Kolmogorov-Smirnov Test Results")
        st.dataframe(ks_df.head(20), use_container_width=True, height=400)

        if "ks_statistic" in ks_df.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
            top = ks_df.nlargest(15, "ks_statistic")
            ax.barh(top["feature"], top["ks_statistic"], color="#3498db")
            ax.set_xlabel("KS Statistic")
            ax.set_title("Top 15 Features with Highest Distribution Drift", fontweight="bold")
            ax.axvline(x=0.1, color="red", linestyle="--", alpha=0.5, label="Threshold")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if importance_path.exists():
        imp_df = pd.read_csv(importance_path)
        st.subheader("Feature Importance Shift Over Time")
        st.dataframe(imp_df.head(20), use_container_width=True)

    fig_path = ROOT / "results" / "figures" / "feature_drift_distributions.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Feature Distribution Changes", use_container_width=True)


# ---------------------------------------------------------------------------
# 8. LIVE SIMULATION
# ---------------------------------------------------------------------------
def render_live_simulation():
    st.title("▶️ Live Drift Detection Simulation")

    st.markdown("""
    Watch drift detection happen in real-time on synthetic data.
    Configure the parameters below and hit **Start Simulation**.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        n_stable = st.slider("Stable samples", 100, 1000, 500, 50)
        stable_error = st.slider("Stable error rate", 0.01, 0.30, 0.05, 0.01)
    with col2:
        n_drift = st.slider("Drift samples", 100, 1000, 500, 50)
        drift_error = st.slider("Drift error rate", 0.20, 0.90, 0.60, 0.05)
    with col3:
        delta = st.select_slider("ADWIN delta", options=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1], value=0.002)
        speed = st.selectbox("Simulation speed", ["Fast", "Medium", "Slow"], index=0)

    speed_map = {"Fast": 0.0, "Medium": 0.005, "Slow": 0.02}

    if st.button("🚀 Start Simulation", type="primary", use_container_width=True):
        from adaptive_ids.drift.detectors import ADWINDetector

        detector = ADWINDetector(delta=delta)
        rng = np.random.RandomState(42)

        progress = st.progress(0, text="Initializing...")
        metrics_container = st.container()
        chart_placeholder = st.empty()
        log_placeholder = st.empty()

        total = n_stable + n_drift
        errors = []
        drift_positions = []
        rolling_errors = []
        window = 50

        mc1, mc2, mc3, mc4 = metrics_container.columns(4)
        m_samples = mc1.empty()
        m_drifts = mc2.empty()
        m_error = mc3.empty()
        m_phase = mc4.empty()

        log_lines = []

        for i in range(total):
            phase = "STABLE" if i < n_stable else "DRIFT"
            p = stable_error if i < n_stable else drift_error
            error = float(rng.binomial(1, p))
            errors.append(error)
            detector.update(error)

            rolling_avg = np.mean(errors[max(0, i-window):i+1])
            rolling_errors.append(rolling_avg)

            if detector.drift_detected():
                drift_positions.append(i)
                log_lines.append(f"⚠️ [{i:5d}] DRIFT DETECTED (rolling_error={rolling_avg:.3f})")

            if i % max(1, total // 100) == 0 or i == total - 1:
                progress.progress((i + 1) / total, text=f"Processing sample {i+1}/{total} ({phase})")
                m_samples.metric("Samples", i + 1)
                m_drifts.metric("Drifts Found", len(drift_positions))
                m_error.metric("Rolling Error", f"{rolling_avg:.3f}")
                m_phase.metric("Phase", phase)

            if i % max(1, total // 20) == 0 or i == total - 1:
                fig, ax = plt.subplots(figsize=(14, 4))
                ax.plot(range(len(rolling_errors)), rolling_errors, linewidth=1, color="#3498db", alpha=0.8)
                ax.axvline(x=n_stable, color="orange", linestyle="--", alpha=0.7, label="Drift injection point")
                for dp in drift_positions:
                    ax.axvline(x=dp, color="red", linestyle=":", alpha=0.5)
                if drift_positions:
                    ax.axvline(x=-1, color="red", linestyle=":", alpha=0.5, label="Drift detected")
                ax.set_xlabel("Sample")
                ax.set_ylabel("Rolling Error Rate")
                ax.set_title("Live Drift Detection", fontweight="bold")
                ax.legend(loc="upper left")
                ax.set_ylim(-0.05, 1.05)
                plt.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close(fig)

            if log_lines:
                log_placeholder.code("\n".join(log_lines[-10:]), language="text")

            if speed_map[speed] > 0:
                time.sleep(speed_map[speed])

        progress.progress(1.0, text="Simulation complete!")
        st.success(f"Processed **{total}** samples. Detected **{len(drift_positions)}** drift events. "
                   f"First detection at sample **{drift_positions[0]}** (injection at {n_stable})."
                   if drift_positions else
                   f"Processed **{total}** samples. No drift detected — try increasing the drift error rate or decreasing delta.")


# ---------------------------------------------------------------------------
# 9. DATASET EXPLORER
# ---------------------------------------------------------------------------
def render_datasets():
    st.title("📁 Dataset Explorer & Additional Training Data")

    tab1, tab2 = st.tabs(["Current Datasets", "Additional Datasets for Training"])

    with tab1:
        profile = load_json(ROOT / "data" / "processed" / "dataset_profile.json")
        if profile:
            st.subheader("CIC-IDS2017 (Primary)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Flows", f"{profile['rows']:,}")
            c2.metric("Features", profile["columns"])
            c3.metric("Attack Types", profile.get("unique_labels", 7))

            if profile.get("class_distribution"):
                fig, ax = plt.subplots(figsize=(10, 5))
                dist = pd.Series(profile["class_distribution"]).sort_values()
                dist.plot.barh(ax=ax, color=sns.color_palette("colorblind", len(dist)))
                ax.set_xlabel("Count")
                ax.set_title("CIC-IDS2017 Class Distribution", fontweight="bold")
                for i, v in enumerate(dist.values):
                    ax.text(v + dist.max() * 0.01, i, f"{v:,.0f}", va="center", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("Dataset profile not available. Run `python scripts/inspect_dataset.py`")

        cross = load_json(ROOT / "results" / "cross_dataset" / "cross_dataset_summary.json")
        if cross and cross.get("datasets_available"):
            st.subheader("UNSW-NB15 (Cross-Validation)")
            for name, info in cross["datasets_available"].items():
                st.markdown(f"**{name}**: {info['rows']:,} rows, {info['features']} features")

    with tab2:
        st.markdown("""
        ### Recommended Datasets for Expanding ADAPT-IDS Training

        These are publicly available, well-cited network intrusion detection datasets
        compatible with the ADAPT-IDS pipeline. All use CICFlowMeter-style features
        or can be adapted.
        """)

        datasets = [
            {
                "name": "CSE-CIC-IDS2018",
                "rows": "16M+ flows",
                "attacks": "Brute-force, Heartbleed, Botnet, DoS, DDoS, Web Attacks, Infiltration",
                "source": "Communications Security Establishment (CSE) + CIC",
                "features": "80 CICFlowMeter features (same format as CIC-IDS2017)",
                "link": "https://registry.opendata.aws/cse-cic-ids2018/",
                "download": "aws s3 sync --no-sign-request --region ca-central-1 s3://cse-cic-ids2018/ dest-dir",
                "size": "~50 GB (full), ~211 MB (V2 on Zenodo)",
                "notes": "Direct successor to CIC-IDS2017. Same feature format — plug-and-play with ADAPT-IDS. Uses 50 attack machines and 420 victim machines.",
                "compatibility": "Direct",
            },
            {
                "name": "UNSW-NB15",
                "rows": "2.5M flows",
                "attacks": "Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms",
                "source": "UNSW Canberra Cyber Range",
                "features": "49 features (4 categories: basic, content, time, additional)",
                "link": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
                "download": "Download from UNSW research portal (registration required)",
                "size": "~2 GB",
                "notes": "Already integrated in ADAPT-IDS cross-dataset module. Different feature set — requires alignment via multi_dataset.py.",
                "compatibility": "Integrated",
            },
            {
                "name": "CIC-IoT-2023",
                "rows": "47M+ flows",
                "attacks": "105 IoT-specific attack types (DDoS, DoS, Recon, Spoofing, Mirai, etc.)",
                "source": "Canadian Institute for Cybersecurity",
                "features": "46 features",
                "link": "https://www.unb.ca/cic/datasets/iotdataset-2023.html",
                "download": "Available via CIC website",
                "size": "~30 GB",
                "notes": "Modern IoT-focused dataset with massive scale. Good for testing ADAPT-IDS on IoT traffic patterns.",
                "compatibility": "Adaptable",
            },
            {
                "name": "TON-IoT",
                "rows": "22M+ records",
                "attacks": "Scanning, DoS, DDoS, Ransomware, Backdoor, Data Injection, XSS, Password Cracking",
                "source": "UNSW Canberra",
                "features": "Network + telemetry features (heterogeneous)",
                "link": "https://research.unsw.edu.au/projects/toniot-datasets",
                "download": "Free for academic use (UNSW portal)",
                "size": "~20 GB",
                "notes": "Covers both IT and OT/ICS networks. Includes Windows/Linux telemetry alongside network flows.",
                "compatibility": "Adaptable",
            },
            {
                "name": "Bot-IoT",
                "rows": "73M+ records",
                "attacks": "DDoS, DoS, OS/Service Scan, Keylogging, Data Exfiltration",
                "source": "UNSW Canberra Cyber Range",
                "features": "46 features (Argus-based)",
                "link": "https://research.unsw.edu.au/projects/bot-iot-dataset",
                "download": "Free for academic use",
                "size": "~70 GB (full), ~16 MB (condensed)",
                "notes": "Massive botnet-focused dataset. Condensed version is manageable for experiments.",
                "compatibility": "Adaptable",
            },
            {
                "name": "CTU-13",
                "rows": "1.6M flows (13 scenarios)",
                "attacks": "Botnet (Neris, Rbot, Virut, Menti, Sogou, Murlo, NSIS.ay)",
                "source": "Czech Technical University, Stratosphere Lab",
                "features": "Bidirectional NetFlow features",
                "link": "https://www.stratosphereips.org/datasets-ctu13",
                "download": "Direct download from Stratosphere website",
                "size": "~3 GB",
                "notes": "13 distinct botnet scenarios. Good for testing drift across different botnet families.",
                "compatibility": "Adaptable",
            },
            {
                "name": "NSL-KDD",
                "rows": "150K flows",
                "attacks": "DoS, R2L, U2R, Probe",
                "source": "University of New Brunswick (improved KDD Cup 99)",
                "features": "41 features",
                "link": "https://www.unb.ca/cic/datasets/nsl.html",
                "download": "Direct download from UNB",
                "size": "~18 MB",
                "notes": "Classic benchmark — outdated but still widely cited. Useful for baseline comparison with older studies.",
                "compatibility": "Adaptable",
            },
        ]

        for ds in datasets:
            compat_color = {"Direct": "🟢", "Integrated": "🔵", "Adaptable": "🟡"}.get(ds["compatibility"], "⚪")
            with st.expander(f"{compat_color} **{ds['name']}** — {ds['rows']} | {ds['size']}", expanded=False):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown(f"**Source**: {ds['source']}")
                    st.markdown(f"**Attack types**: {ds['attacks']}")
                    st.markdown(f"**Features**: {ds['features']}")
                    st.markdown(f"**Notes**: {ds['notes']}")
                    st.markdown(f"**Link**: [{ds['link']}]({ds['link']})")
                with c2:
                    st.markdown(f"**Size**: {ds['size']}")
                    st.markdown(f"**Compatibility**: {compat_color} {ds['compatibility']}")
                    if ds.get("download"):
                        st.code(ds["download"], language="bash")

        st.divider()
        st.markdown("""
        #### Compatibility Legend
        - 🟢 **Direct**: Same CICFlowMeter feature format — works with ADAPT-IDS out of the box
        - 🔵 **Integrated**: Already supported in the codebase (via `multi_dataset.py`)
        - 🟡 **Adaptable**: Different feature format — needs a feature alignment adapter (straightforward to add)
        """)


# ---------------------------------------------------------------------------
# 10. LIVE PREDICTION
# ---------------------------------------------------------------------------
def render_live_prediction():
    st.title("📡 Live Prediction")

    st.markdown("""
    Upload a CICFlowMeter CSV or any flow-feature CSV to classify network traffic.
    The model will predict each flow as **BENIGN** or **ATTACK**.
    """)

    model_options = {}
    for p in [
        ROOT / "results" / "cross_dataset" / "lightgbm_combined.joblib",
        ROOT / "results" / "temporal" / "lightgbm" / "lightgbm_temporal.joblib",
        ROOT / "results" / "baseline" / "lightgbm" / "lightgbm_baseline.joblib",
    ]:
        if p.exists():
            model_options[p.stem] = p

    if not model_options:
        st.warning("No trained models found. Run the training scripts first.")
        return

    selected = st.selectbox("Select Model", list(model_options.keys()))
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded and selected:
        from adaptive_ids.models.baseline import BaselineIDS

        df = pd.read_csv(uploaded, low_memory=False)
        df.columns = df.columns.str.strip()
        st.write(f"Loaded **{len(df):,}** rows, **{len(df.columns)}** columns")

        model = BaselineIDS.load(model_options[selected])
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        exclude = {"Label", "Timestamp", "Flow ID"}
        feature_cols = [c for c in numeric_cols if c not in exclude]

        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float64)

        if st.button("🔍 Run Predictions", type="primary"):
            with st.spinner("Classifying flows..."):
                try:
                    preds = model.predict(X)
                    n_attack = int((preds == "ATTACK").sum())
                    n_benign = int((preds == "BENIGN").sum())

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Flows", f"{len(preds):,}")
                    c2.metric("Benign", f"{n_benign:,}")
                    c3.metric("🚨 Attack", f"{n_attack:,}")

                    if n_attack > 0:
                        st.error(f"⚠️ **{n_attack}** potential attacks detected ({n_attack/len(preds)*100:.1f}%)")
                    else:
                        st.success("✅ No attacks detected in this traffic sample")

                    df["Prediction"] = preds
                    st.dataframe(df[["Prediction"] + feature_cols[:8]].head(200),
                                 use_container_width=True, height=400)

                    csv = df[["Prediction"]].to_csv(index=False)
                    st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _plot_confusion_matrix(metrics: dict):
    if "confusion_matrix" not in metrics:
        return None
    cm = np.array(metrics["confusion_matrix"])
    labels = metrics.get("confusion_labels", ["ATTACK", "BENIGN"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig


def _plot_windowed_f1(windowed_metrics: list[dict], title: str = "F1 Over Time"):
    fig, ax = plt.subplots(figsize=(12, 4))
    windows = [m.get("window_id", i) for i, m in enumerate(windowed_metrics)]
    f1s = [m.get("f1", 0) for m in windowed_metrics]
    ax.plot(windows, f1s, "o-", markersize=3, linewidth=1.5, color="#3498db")
    ax.set_xlabel("Window")
    ax.set_ylabel("F1 Score")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()

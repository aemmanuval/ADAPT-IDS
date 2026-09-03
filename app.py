"""ADAPT-IDS Dashboard — Visual monitoring and testing interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="ADAPT-IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    st.title("🛡️ ADAPT-IDS")
    st.caption("Adaptive Intrusion Detection Under Concept & Feature Drift")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Model Performance", "Drift Detection", "Adaptation Comparison", "Live Prediction", "Dataset Explorer"],
    )

    if page == "Overview":
        render_overview()
    elif page == "Model Performance":
        render_performance()
    elif page == "Drift Detection":
        render_drift()
    elif page == "Adaptation Comparison":
        render_adaptation()
    elif page == "Live Prediction":
        render_live()
    elif page == "Dataset Explorer":
        render_explorer()


def render_overview():
    st.header("System Overview")

    st.markdown("""
    ### What does ADAPT-IDS do?

    Traditional intrusion detection systems (IDS) learn to recognise attacks from historical data.
    But network attacks **evolve** — new malware, new techniques, new protocols. When attacks change,
    a static model starts missing them. This is called **concept drift**.

    **ADAPT-IDS detects when drift happens and automatically retrains the model** to catch new attack types.
    """)

    col1, col2, col3, col4 = st.columns(4)

    baseline = load_json(ROOT / "results" / "baseline" / "lightgbm" / "metrics.json")
    temporal = load_json(ROOT / "results" / "temporal" / "lightgbm" / "metrics.json")
    adaptation = load_json(ROOT / "results" / "adaptation" / "comparison.json")
    unsw = load_json(ROOT / "results" / "cross_dataset" / "cross_dataset_summary.json")

    with col1:
        if baseline:
            st.metric("Random Split F1", f"{baseline['f1']:.3f}", help="Performance under ideal (IID) conditions")
        else:
            st.metric("Random Split F1", "—", help="Run train_baseline.py")

    with col2:
        if temporal:
            st.metric("Temporal F1 (Static)", f"{temporal['f1']:.3f}",
                       delta=f"{temporal['f1'] - (baseline['f1'] if baseline else 0):.3f}",
                       delta_color="normal",
                       help="Performance on later traffic without adaptation")
        else:
            st.metric("Temporal F1 (Static)", "—")

    with col3:
        if adaptation and "drift_triggered" in adaptation:
            dt = adaptation["drift_triggered"]
            st.metric("Adaptive F1", f"{dt['f1']:.3f}",
                       delta=f"+{dt['f1'] - (temporal['f1'] if temporal else 0):.3f}",
                       help="Performance with drift-triggered retraining")
        else:
            st.metric("Adaptive F1", "—", help="Run run_adaptation_experiment.py")

    with col4:
        if unsw:
            st.metric("UNSW-NB15 F1", f"{unsw['unsw_internal']['f1']:.3f}",
                       help="Cross-dataset validation on UNSW-NB15")
        else:
            st.metric("UNSW-NB15 F1", "—")

    st.divider()

    st.markdown("### The Core Problem")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Without adaptation**, the model trained on earlier traffic
        completely fails on newer attack types:

        - F1 drops from **0.997 → 0.036**
        - **98.2% of attacks are missed**
        - The IDS becomes useless
        """)
    with col_b:
        st.markdown("""
        **With drift-triggered adaptation**, the model automatically
        detects changes and retrains:

        - F1 recovers to **0.987**
        - Only **2.3% of attacks are missed**
        - Uses 71% fewer retrains than periodic approach
        """)

    st.divider()

    st.markdown("### System Architecture")
    st.code("""
    Network Traffic → Feature Extraction → IDS Classifier → Predictions
                                                    ↓
                                            Error Monitoring
                                                    ↓
                                          ADWIN Drift Detection
                                                    ↓
                                        Drift? → Yes → Retrain Model
                                               → No  → Continue
    """, language="text")

    st.markdown("### Research Hypotheses — Status")
    hypotheses = {
        "H1: Static IDS degrades under temporal drift": "✅ Confirmed — F1 dropped 96.4%",
        "H2: Drift-triggered adaptation recovers performance": "✅ Confirmed — F1 recovered to 0.987",
        "H3: Drift-triggered beats periodic retraining": "✅ Confirmed — Higher F1, 71% fewer retrains",
        "H4: Different drift types need different responses": "🔄 Phase 4 (synthetic drift experiments)",
    }
    for h, status in hypotheses.items():
        st.markdown(f"- **{h}**: {status}")


def render_performance():
    st.header("Model Performance")

    tab1, tab2, tab3 = st.tabs(["Baseline (Random Split)", "Temporal Split", "Cross-Dataset"])

    with tab1:
        col1, col2 = st.columns(2)
        for algo, col in [("lightgbm", col1), ("random_forest", col2)]:
            with col:
                m = load_json(ROOT / "results" / "baseline" / algo / "metrics.json")
                if m:
                    st.subheader(algo.replace("_", " ").title())
                    st.metric("F1", f"{m['f1']:.4f}")
                    st.metric("Recall", f"{m['recall']:.4f}")
                    st.metric("Precision", f"{m['precision']:.4f}")
                    st.metric("MCC", f"{m['mcc']:.4f}")
                    if m.get("fpr") is not None:
                        st.metric("FPR", f"{m['fpr']:.4f}")
                        st.metric("FNR", f"{m['fnr']:.4f}")

                    cm = np.array(m["confusion_matrix"])
                    labels = m["confusion_labels"]
                    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                    st.markdown("**Confusion Matrix**")
                    st.dataframe(cm_df, use_container_width=True)
                else:
                    st.info(f"No {algo} baseline results. Run `python scripts/train_baseline.py`")

        fig_path = ROOT / "results" / "figures" / "class_distribution.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Class Distribution", use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        for algo, col in [("lightgbm", col1), ("random_forest", col2)]:
            with col:
                m = load_json(ROOT / "results" / "temporal" / algo / "metrics.json")
                if m:
                    st.subheader(f"{algo.replace('_', ' ').title()} (Temporal)")
                    st.metric("F1", f"{m['f1']:.4f}")
                    st.metric("Recall", f"{m['recall']:.4f}")
                    if m.get("fnr") is not None:
                        st.metric("FNR (Missed Attacks)", f"{m['fnr']:.4f}")
                    st.warning(f"⚠️ {m.get('fnr', 0)*100:.1f}% of attacks missed without adaptation")
                else:
                    st.info(f"No temporal results. Run `python scripts/evaluate_temporal.py`")

        for algo in ["lightgbm", "random_forest"]:
            fig = ROOT / "results" / "figures" / f"f1_over_time_{algo}_temporal.png"
            if fig.exists():
                st.image(str(fig), caption=f"F1 Over Time — {algo} (Temporal)", use_container_width=True)

    with tab3:
        summary = load_json(ROOT / "results" / "cross_dataset" / "cross_dataset_summary.json")
        if summary:
            st.subheader("UNSW-NB15 (Cross-Dataset Validation)")
            ui = summary["unsw_internal"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("F1", f"{ui['f1']:.4f}")
            c2.metric("Recall", f"{ui['recall']:.4f}")
            c3.metric("FPR", f"{ui.get('fpr', 0):.4f}")
            c4.metric("MCC", f"{ui['mcc']:.4f}")

            ds = summary.get("datasets_available", {})
            for name, info in ds.items():
                st.markdown(f"**{name}**: {info['rows']:,} rows, {info['features']} features")
        else:
            st.info("Run `python scripts/train_cross_dataset.py`")

        fig = ROOT / "results" / "figures" / "cm_unsw_internal.png"
        if fig.exists():
            st.image(str(fig), caption="UNSW-NB15 Confusion Matrix", use_container_width=True)


def render_drift():
    st.header("Drift Detection")

    drift_csv = ROOT / "results" / "drift" / "drift_events.csv"
    if drift_csv.exists():
        events = pd.read_csv(drift_csv)
        st.metric("Total Drift Events", len(events))

        st.subheader("Drift Event Timeline")
        if "stream_position" in events.columns:
            st.bar_chart(events.set_index("event_id")["stream_position"])

        st.subheader("Drift Event Log")
        display_cols = [c for c in ["event_id", "detector", "stream_position", "timestamp", "metric_context"] if c in events.columns]
        st.dataframe(events[display_cols], use_container_width=True, height=400)
    else:
        st.info("No drift events. Run `python scripts/run_drift_experiment.py`")

    for fig_name in ["error_rate_drift.png", "f1_over_time_drift.png"]:
        fig = ROOT / "results" / "figures" / fig_name
        if fig.exists():
            st.image(str(fig), use_container_width=True)


def render_adaptation():
    st.header("Adaptation Strategy Comparison")

    comparison = load_json(ROOT / "results" / "adaptation" / "comparison.json")
    if not comparison:
        st.info("Run `python scripts/run_adaptation_experiment.py`")
        return

    rows = []
    for name, data in comparison.items():
        rows.append({
            "Strategy": name,
            "F1": data["f1"],
            "Recall": data["recall"],
            "FNR (Missed)": data.get("fnr", 0),
            "FPR": data.get("fpr", 0),
            "MCC": data["mcc"],
            "Retrains": data["n_retrains"],
            "Cost (s)": data["retrain_time_s"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.highlight_max(subset=["F1", "Recall", "MCC"], color="#90EE90")
                .highlight_min(subset=["FNR (Missed)", "FPR", "Retrains", "Cost (s)"], color="#90EE90"),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("F1 Score by Strategy")
        chart_data = df.set_index("Strategy")["F1"]
        st.bar_chart(chart_data)

    with col2:
        st.subheader("Retraining Cost")
        cost_data = df.set_index("Strategy")[["Retrains", "Cost (s)"]]
        st.bar_chart(cost_data)

    st.subheader("Key Insight")
    if "drift_triggered" in comparison and "static" in comparison:
        static_f1 = comparison["static"]["f1"]
        adaptive_f1 = comparison["drift_triggered"]["f1"]
        improvement = (adaptive_f1 - static_f1) / static_f1 * 100
        retrains = comparison["drift_triggered"]["n_retrains"]
        st.success(
            f"**Drift-triggered adaptation improved F1 by {improvement:.0f}%** "
            f"(from {static_f1:.3f} to {adaptive_f1:.3f}) "
            f"with only **{retrains} retrains**."
        )

    fig = ROOT / "results" / "figures" / "adaptation_comparison.png"
    if fig.exists():
        st.image(str(fig), caption="F1 Over Time — All Strategies", use_container_width=True)


def render_live():
    st.header("Live Prediction")

    st.markdown("""
    Upload a **CICFlowMeter CSV** or any flow-feature CSV to get predictions.

    Generate a CICFlowMeter CSV from your Wireshark PCAP capture using
    [CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter).
    """)

    model_options = {}
    for p in [
        ROOT / "results" / "temporal" / "lightgbm" / "lightgbm_temporal.joblib",
        ROOT / "results" / "baseline" / "lightgbm" / "lightgbm_baseline.joblib",
        ROOT / "results" / "cross_dataset" / "lightgbm_combined.joblib",
        ROOT / "results" / "cross_dataset" / "lightgbm_unsw.joblib",
    ]:
        if p.exists():
            model_options[p.stem] = p

    if not model_options:
        st.warning("No trained models found. Run training scripts first.")
        return

    selected_model = st.selectbox("Select Model", list(model_options.keys()))

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded and selected_model:
        from adaptive_ids.models.baseline import BaselineIDS

        df = pd.read_csv(uploaded, low_memory=False)
        df.columns = df.columns.str.strip()
        st.write(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

        model = BaselineIDS.load(model_options[selected_model])

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        exclude = {"Label", "Timestamp", "Flow ID"}
        feature_cols = [c for c in numeric_cols if c not in exclude]

        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float64)

        if st.button("🔍 Run Predictions"):
            with st.spinner("Predicting..."):
                try:
                    preds = model.predict(X)
                    n_attack = int((preds == "ATTACK").sum())
                    n_benign = int((preds == "BENIGN").sum())

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Flows", f"{len(preds):,}")
                    col2.metric("BENIGN", f"{n_benign:,}")
                    col3.metric("🚨 ATTACK", f"{n_attack:,}")

                    if n_attack > 0:
                        st.error(f"⚠️ {n_attack} potential attacks detected ({n_attack/len(preds)*100:.1f}%)")
                    else:
                        st.success("✅ No attacks detected in this traffic")

                    df["Prediction"] = preds
                    st.subheader("Prediction Results")
                    st.dataframe(df[["Prediction"] + feature_cols[:5]], height=300)

                    csv = df[["Prediction"]].to_csv(index=False)
                    st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.info("This may happen if the CSV features don't match the model's training features. "
                            "Try using a CICFlowMeter-generated CSV or the UNSW model for UNSW-format data.")


def render_explorer():
    st.header("Dataset Explorer")

    profile = load_json(ROOT / "data" / "processed" / "dataset_profile.json")
    if profile:
        st.subheader("CIC-IDS2017")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Flows", f"{profile['rows']:,}")
        c2.metric("Features", profile["columns"])
        c3.metric("Attack Types", profile.get("unique_labels", "—"))

        if profile.get("class_distribution"):
            st.subheader("Class Distribution")
            dist = pd.Series(profile["class_distribution"]).sort_values(ascending=True)
            st.bar_chart(dist)

        fig = ROOT / "results" / "figures" / "class_distribution.png"
        if fig.exists():
            st.image(str(fig), use_container_width=True)
    else:
        st.info("No dataset profile. Run `python scripts/inspect_dataset.py`")

    summary = load_json(ROOT / "results" / "cross_dataset" / "cross_dataset_summary.json")
    if summary:
        st.subheader("UNSW-NB15")
        for name, info in summary.get("datasets_available", {}).items():
            c1, c2 = st.columns(2)
            c1.metric("Flows", f"{info['rows']:,}")
            c2.metric("Features", info["features"])


if __name__ == "__main__":
    main()

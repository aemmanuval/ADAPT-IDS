#!/usr/bin/env python3
import os; os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
"""Phase 5: Feature drift analysis — statistical tests on distribution changes.

Compares feature distributions between temporal windows using:
  - Kolmogorov-Smirnov test (distribution shape)
  - Mean/std shift magnitude
  - Feature importance changes across time
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_ids.config.settings import load_config, get_project_root
from adaptive_ids.preprocessing.pipeline import PreprocessingPipeline
from adaptive_ids.evaluation.temporal import temporal_split
from adaptive_ids.models.baseline import BaselineIDS
from adaptive_ids.utils.logging import setup_logging, get_logger
from adaptive_ids.utils.reproducibility import set_global_seed


def ks_test_features(df_a: pd.DataFrame, df_b: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Run KS test on each feature between two time windows."""
    results = []
    for col in feature_cols:
        a = df_a[col].dropna().values
        b = df_b[col].dropna().values
        if len(a) < 10 or len(b) < 10:
            continue

        ks_stat, p_value = stats.ks_2samp(a, b)
        mean_shift = abs(b.mean() - a.mean())
        relative_shift = mean_shift / (abs(a.mean()) + 1e-10)

        results.append({
            "feature": col,
            "ks_statistic": round(ks_stat, 6),
            "p_value": p_value,
            "significant": p_value < 0.05,
            "mean_a": round(a.mean(), 4),
            "mean_b": round(b.mean(), 4),
            "std_a": round(a.std(), 4),
            "std_b": round(b.std(), 4),
            "mean_shift": round(mean_shift, 4),
            "relative_shift": round(relative_shift, 4),
        })

    df = pd.DataFrame(results).sort_values("ks_statistic", ascending=False)
    return df


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    setup_logging(config["logging"]["level"])
    logger = get_logger("feature_drift")
    seed = config["experiment"]["random_seed"]
    set_global_seed(seed)

    root = get_project_root()
    proc_dir = root / config["dataset"]["processed_dir"]
    results_dir = root / "results" / "feature_drift"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = proc_dir / "cleaned.parquet"
    if not data_path.exists():
        logger.error("Run preprocess_dataset.py first.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ADAPT-IDS  —  Feature Drift Analysis (Phase 5)")
    print("=" * 60 + "\n")

    df = pd.read_parquet(data_path)
    ts_col = config["dataset"]["timestamp_column"]
    eval_cfg = config["evaluation"]

    splits = temporal_split(
        df, timestamp_column=ts_col,
        train_fraction=eval_cfg["train_fraction"],
        validation_fraction=eval_cfg["validation_fraction"],
        test_fraction=eval_cfg["test_fraction"],
    )

    pipeline = PreprocessingPipeline(config)
    train_df, _, _ = pipeline.fit_transform(splits["train"])
    test_df, _, _ = pipeline.transform(splits["test"])

    feature_cols = [c for c in train_df.columns if train_df[c].dtype in ["float64", "int64"]]
    print(f"Analyzing {len(feature_cols)} features across train/test split")

    # KS test: train vs test
    print("\n[1] Kolmogorov-Smirnov test: training vs test distributions...")
    ks_results = ks_test_features(train_df, test_df, feature_cols)
    ks_results.to_csv(results_dir / "ks_test_train_vs_test.csv", index=False)

    n_significant = ks_results["significant"].sum()
    print(f"    Features with significant drift (p<0.05): {n_significant}/{len(ks_results)}")

    print(f"\n    Top 10 most-shifted features:")
    print(f"    {'Feature':<35} {'KS Stat':>10} {'Rel Shift':>12} {'Significant':>12}")
    print("    " + "-" * 72)
    for _, row in ks_results.head(10).iterrows():
        sig = "YES" if row["significant"] else "no"
        print(f"    {row['feature']:<35} {row['ks_statistic']:>10.4f} {row['relative_shift']:>12.4f} {sig:>12}")

    # Feature importance comparison
    print("\n[2] Feature importance analysis...")
    model_params = config["models"].get("lightgbm", {})

    _, X_train, y_train = pipeline.fit_transform(splits["train"])
    model_train = BaselineIDS("lightgbm", params=model_params, random_seed=seed)
    model_train.fit(X_train, y_train)
    imp_train = model_train.feature_importances()

    # Train on first half of test to see how importances shift
    test_half = splits["test"].iloc[:len(splits["test"])//2]
    _, X_test_half, y_test_half = pipeline.transform(test_half)
    if len(set(y_test_half)) >= 2:
        model_test = BaselineIDS("lightgbm", params=model_params, random_seed=seed)
        model_test.fit(X_test_half, y_test_half)
        imp_test = model_test.feature_importances()

        importance_df = pd.DataFrame({
            "feature": pipeline.feature_columns[:len(imp_train)],
            "importance_train": imp_train,
            "importance_test": imp_test[:len(imp_train)] if len(imp_test) >= len(imp_train) else np.pad(imp_test, (0, len(imp_train) - len(imp_test))),
        })
        importance_df["importance_shift"] = abs(importance_df["importance_test"] - importance_df["importance_train"])
        importance_df = importance_df.sort_values("importance_shift", ascending=False)
        importance_df.to_csv(results_dir / "feature_importance_shift.csv", index=False)

        print(f"\n    Top 10 features with largest importance shift:")
        print(f"    {'Feature':<35} {'Train Imp':>10} {'Test Imp':>10} {'Shift':>10}")
        print("    " + "-" * 68)
        for _, row in importance_df.head(10).iterrows():
            print(f"    {row['feature']:<35} {row['importance_train']:>10.0f} {row['importance_test']:>10.0f} {row['importance_shift']:>10.0f}")

    # Windowed drift analysis
    print("\n[3] Windowed distribution analysis across test stream...")
    n_windows = 5
    window_size = len(test_df) // n_windows
    window_stats = []

    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, len(test_df))
        window = test_df.iloc[start:end]

        for col in feature_cols[:20]:
            vals = window[col].dropna().values
            if len(vals) == 0:
                continue
            window_stats.append({
                "window": i,
                "feature": col,
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "median": round(float(np.median(vals)), 4),
            })

    window_df = pd.DataFrame(window_stats)
    window_df.to_csv(results_dir / "windowed_feature_stats.csv", index=False)

    # Visualization
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_features = ks_results.head(6)["feature"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for idx, feat in enumerate(top_features):
        if idx >= 6:
            break
        ax = axes[idx]
        train_vals = train_df[feat].dropna().values
        test_vals = test_df[feat].dropna().values

        if len(train_vals) > 50000:
            train_vals = np.random.choice(train_vals, 50000, replace=False)
        if len(test_vals) > 50000:
            test_vals = np.random.choice(test_vals, 50000, replace=False)

        ax.hist(train_vals, bins=50, alpha=0.5, label="Train", density=True, color="#1f77b4")
        ax.hist(test_vals, bins=50, alpha=0.5, label="Test", density=True, color="#d62728")
        ks_row = ks_results[ks_results["feature"] == feat].iloc[0]
        ax.set_title(f"{feat}\nKS={ks_row['ks_statistic']:.3f}", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    plt.suptitle("Feature Distribution Drift: Training vs Test", fontsize=12)
    plt.tight_layout()
    fig_path = root / "results" / "figures" / "feature_drift_distributions.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary
    summary = {
        "n_features_analyzed": len(ks_results),
        "n_significant_drift": int(n_significant),
        "pct_features_drifted": round(n_significant / len(ks_results) * 100, 1),
        "top_drifted_features": ks_results.head(10)[["feature", "ks_statistic", "relative_shift"]].to_dict("records"),
    }
    with open(results_dir / "feature_drift_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  FEATURE DRIFT SUMMARY")
    print(f"{'='*60}")
    print(f"  Features analyzed:      {len(ks_results)}")
    print(f"  Significant drift:      {n_significant} ({summary['pct_features_drifted']}%)")
    print(f"  Results:                {results_dir}")
    print(f"  Distribution plot:      {fig_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)

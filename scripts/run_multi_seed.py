#!/usr/bin/env python3
"""Multi-seed experiment runner with confidence intervals and statistical tests.

Runs the core experiments across multiple seeds and reports:
  - Mean ± std for all metrics
  - 95% bootstrap confidence intervals
  - Wilcoxon signed-rank test (drift-triggered vs periodic)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_ids.config.settings import load_config, get_project_root
from adaptive_ids.preprocessing.pipeline import PreprocessingPipeline
from adaptive_ids.evaluation.temporal import temporal_split
from adaptive_ids.evaluation.metrics import compute_metrics
from adaptive_ids.models.baseline import BaselineIDS
from adaptive_ids.streaming.stream import CSVStream
from adaptive_ids.drift.detectors import create_detector
from adaptive_ids.adaptation.strategies import (
    StaticStrategy, PeriodicStrategy, DriftTriggeredStrategy,
    AdaptiveModelManager,
)
from adaptive_ids.utils.logging import setup_logging, get_logger
from adaptive_ids.utils.reproducibility import set_global_seed

SEEDS = [42, 123, 256, 512, 1024]


def run_one_seed(seed, X_train, y_train, X_test, y_test, feature_names, config):
    set_global_seed(seed)
    algo = "lightgbm"
    model_params = {**config["models"].get(algo, {}), "n_jobs": 1, "num_threads": 1}
    drift_cfg = config["drift"]

    strategies = {
        "static": StaticStrategy(),
        "periodic_5000": PeriodicStrategy(period=5000),
        "drift_triggered": DriftTriggeredStrategy(cooldown=2000),
    }

    results = {}
    for name, strategy in strategies.items():
        model = BaselineIDS(algo, params=model_params, random_seed=seed)
        model.fit(X_train, y_train)
        detector = create_detector(drift_cfg)

        manager = AdaptiveModelManager(
            algorithm=algo, model_params=model_params,
            strategy=strategy, detector=detector,
            window_size=20000, random_seed=seed,
        )
        manager.set_initial_model(model)

        stream = CSVStream(X_test, y_test, feature_names=feature_names)
        preds = []
        for event in stream:
            y_pred, _, _ = manager.process_sample(event["features"], event["label"], event["index"])
            preds.append(y_pred)

        metrics = compute_metrics(y_test, np.array(preds), positive_label="ATTACK")
        results[name] = {
            "f1": metrics["f1"],
            "recall": metrics["recall"],
            "fnr": metrics.get("fnr", 0),
            "mcc": metrics["mcc"],
            "n_retrains": manager.n_retrains,
        }
    return results


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    values = np.array(values)
    bootstraps = np.array([np.mean(np.random.choice(values, size=len(values), replace=True))
                           for _ in range(n_bootstrap)])
    lower = np.percentile(bootstraps, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstraps, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


def main():
    config = load_config()
    setup_logging(config["logging"]["level"])
    logger = get_logger("multi_seed")

    root = get_project_root()
    proc_dir = root / config["dataset"]["processed_dir"]
    results_dir = root / "results" / "multi_seed"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = proc_dir / "cleaned.parquet"
    if not data_path.exists():
        logger.error("Run preprocess_dataset.py first.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ADAPT-IDS  —  Multi-Seed Statistical Validation")
    print(f"  Seeds: {SEEDS}")
    print("=" * 60 + "\n")

    df = pd.read_parquet(data_path)
    eval_cfg = config["evaluation"]
    ts_col = config["dataset"]["timestamp_column"]

    splits = temporal_split(
        df, timestamp_column=ts_col,
        train_fraction=eval_cfg["train_fraction"],
        validation_fraction=eval_cfg["validation_fraction"],
        test_fraction=eval_cfg["test_fraction"],
    )

    pipeline = PreprocessingPipeline(config)
    _, X_train, y_train = pipeline.fit_transform(splits["train"])
    _, X_test, y_test = pipeline.transform(splits["test"])

    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}\n")

    all_seed_results = {}
    for seed in SEEDS:
        print(f"--- Seed {seed} ---")
        result = run_one_seed(seed, X_train, y_train, X_test, y_test,
                              pipeline.feature_columns, config)
        all_seed_results[seed] = result
        for name, r in result.items():
            print(f"  {name:<20} F1={r['f1']:.4f}  Recall={r['recall']:.4f}  Retrains={r['n_retrains']}")

    # Aggregate
    strategies = ["static", "periodic_5000", "drift_triggered"]
    metrics_names = ["f1", "recall", "fnr", "mcc"]

    print("\n" + "=" * 60)
    print("  AGGREGATED RESULTS (mean ± std, 95% CI)")
    print("=" * 60)

    summary = {}
    for strat in strategies:
        strat_summary = {}
        for metric in metrics_names:
            values = [all_seed_results[s][strat][metric] for s in SEEDS]
            ci_low, ci_high = bootstrap_ci(values)
            strat_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,
                "values": values,
            }
        retrains = [all_seed_results[s][strat]["n_retrains"] for s in SEEDS]
        strat_summary["n_retrains"] = {"mean": float(np.mean(retrains)), "std": float(np.std(retrains)), "values": retrains}
        summary[strat] = strat_summary

        f1 = strat_summary["f1"]
        print(f"\n  {strat}:")
        print(f"    F1:     {f1['mean']:.4f} ± {f1['std']:.4f}  (95% CI: [{f1['ci_95_lower']:.4f}, {f1['ci_95_upper']:.4f}])")
        recall = strat_summary["recall"]
        print(f"    Recall: {recall['mean']:.4f} ± {recall['std']:.4f}")

    # Wilcoxon signed-rank test: drift-triggered vs periodic
    dt_f1 = [all_seed_results[s]["drift_triggered"]["f1"] for s in SEEDS]
    p5_f1 = [all_seed_results[s]["periodic_5000"]["f1"] for s in SEEDS]

    if len(SEEDS) >= 5:
        stat, p_value = scipy_stats.wilcoxon(dt_f1, p5_f1)
        print(f"\n  Wilcoxon signed-rank (drift_triggered vs periodic_5000):")
        print(f"    statistic={stat:.4f}, p-value={p_value:.4f}")
        print(f"    {'Significant (p<0.05)' if p_value < 0.05 else 'Not significant'}")
        summary["statistical_test"] = {
            "test": "Wilcoxon signed-rank",
            "comparison": "drift_triggered vs periodic_5000",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    with open(results_dir / "multi_seed_results.json", "w") as fh:
        json.dump({"seeds": SEEDS, "results_per_seed": {str(k): v for k, v in all_seed_results.items()}, "summary": summary}, fh, indent=2, default=str)

    print(f"\nResults saved to {results_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

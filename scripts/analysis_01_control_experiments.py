from __future__ import annotations

import itertools
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


SNAPSHOT_LABEL = "snapshot_45"
METRICS = ["f1_full_test", "mae_full_test", "daf_full_test", "r2_full_test"]
METRIC_LABELS = {
    "f1_full_test": "F1",
    "mae_full_test": "MAE",
    "daf_full_test": "DAF",
    "r2_full_test": "R2",
}


def build_same_data_pairs(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for combo, group in metadata.groupby("training_combo"):
        if len(group) < 2:
            continue
        records = group.to_dict("records")
        for left, right in itertools.combinations(records, 2):
            if left["family"] == right["family"]:
                continue
            row = {
                "experiment_type": "same_data_diff_architecture",
                "group_key": combo,
                "model_a": left["model_key"],
                "model_b": right["model_key"],
                "family_a": left["family"],
                "family_b": right["family"],
                "architecture_a": left["architecture_group"],
                "architecture_b": right["architecture_group"],
            }
            for metric in METRICS:
                row[f"delta_{metric}"] = abs(left[metric] - right[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def build_same_family_pairs(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for family, group in metadata.groupby("family"):
        if group["training_combo"].nunique() < 2:
            continue
        records = group.to_dict("records")
        for left, right in itertools.combinations(records, 2):
            if left["training_combo"] == right["training_combo"]:
                continue
            row = {
                "experiment_type": "same_family_diff_data",
                "group_key": family,
                "model_a": left["model_key"],
                "model_b": right["model_key"],
                "family": family,
                "training_combo_a": left["training_combo"],
                "training_combo_b": right["training_combo"],
                "architecture_group": left["architecture_group"],
            }
            for metric in METRICS:
                row[f"delta_{metric}"] = abs(left[metric] - right[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_pairs(same_data_pairs: pd.DataFrame, same_family_pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    comparison_rows: list[dict] = []
    summary: dict[str, dict] = {}

    for label, frame in {
        "same_data_diff_architecture": same_data_pairs,
        "same_family_diff_data": same_family_pairs,
    }.items():
        summary[label] = {
            "n_pairs": int(len(frame)),
        }
        for metric in METRICS:
            metric_key = f"delta_{metric}"
            summary[label][metric_key] = {
                "mean": float(frame[metric_key].mean()),
                "median": float(frame[metric_key].median()),
                "max": float(frame[metric_key].max()),
            }
            comparison_rows.append(
                {
                    "experiment_type": label,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "mean_abs_delta": frame[metric_key].mean(),
                    "median_abs_delta": frame[metric_key].median(),
                    "max_abs_delta": frame[metric_key].max(),
                    "n_pairs": len(frame),
                }
            )

    return pd.DataFrame(comparison_rows), summary


def make_plot(comparison: pd.DataFrame, output_path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    label_map = {
        "same_data_diff_architecture": "Same data,\nchange model",
        "same_family_diff_data": "Same family,\nchange data",
    }
    palette = {
        "same_data_diff_architecture": "#4c78a8",
        "same_family_diff_data": "#e45756",
    }

    for ax, metric in zip(axes.flat, METRICS):
        subset = comparison[comparison["metric"] == metric].copy()
        subset["experiment_label"] = subset["experiment_type"].map(label_map)
        ax.bar(
            subset["experiment_label"],
            subset["mean_abs_delta"],
            color=[palette[k] for k in subset["experiment_type"]],
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel("Mean absolute delta")
        ax.set_xlabel("")

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_table_dir = RESULTS_DIR / "tables" / "analysis_01_control"
    output_figure_dir = RESULTS_DIR / "figures" / "analysis_01_control"
    ensure_dir(output_table_dir)
    ensure_dir(output_figure_dir)

    metadata = pd.read_csv(PROCESSED_DIR / f"model_metadata_{SNAPSHOT_LABEL}.csv")
    same_data_pairs = build_same_data_pairs(metadata)
    same_family_pairs = build_same_family_pairs(metadata)
    comparison, summary = summarize_pairs(same_data_pairs, same_family_pairs)

    same_data_pairs.to_csv(output_table_dir / "same_data_diff_architecture_pairs.csv", index=False)
    same_family_pairs.to_csv(output_table_dir / "same_family_diff_data_pairs.csv", index=False)
    comparison.to_csv(output_table_dir / "pair_delta_summary.csv", index=False)
    (output_table_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_plot(comparison, output_figure_dir / "pair_delta_comparison.png")


if __name__ == "__main__":
    main()

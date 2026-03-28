from __future__ import annotations

import json
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dva_project.settings import RESULTS_DIR
from dva_project.utils import ensure_dir


SNAPSHOT_LABEL = "snapshot_45"
TRAINING_GROUP_LABELS = {
    "A_MPtrj_only": "MPtrj",
    "B_MPtrj_sAlex": "MPtrj+sAlex",
    "C_OMat24_MPtrj": "OMat24+MPtrj",
    "D_OMat24_sAlex_MPtrj": "OMat24+sAlex+MPtrj",
    "E_other": "Other",
}
ARCH_MARKERS = {
    "equivariant_gnn": "o",
    "invariant_gnn": "s",
    "transformer": "^",
    "hybrid_ensemble": "D",
    "non_gnn": "X",
}


def extract_gpu_hours(raw) -> float:
    if pd.isna(raw) or raw == "missing":
        return np.nan
    if isinstance(raw, str) and raw.startswith("{"):
        obj = json.loads(raw)

        def rec(value):
            if isinstance(value, dict):
                if "amount" in value and "hours" in value:
                    return float(value["amount"]) * float(value["hours"])
                numbers = [rec(v) for v in value.values()]
                numbers = [n for n in numbers if not math.isnan(n)]
                return sum(numbers) if numbers else np.nan
            return np.nan

        return rec(obj)
    match = re.search(r"(\d+)\s*[x×]\s*[^x×]+[x×]\s*(\d+(?:\.\d+)?)\s*h", str(raw))
    if match:
        return float(match.group(1)) * float(match.group(2))
    return np.nan


def compute_proxy_costs(metadata: pd.DataFrame) -> pd.DataFrame:
    frame = metadata.copy()
    frame["proxy_cost"] = frame["effective_training_structures"] * frame["model_params"]
    frame["log_proxy_cost"] = np.log10(frame["proxy_cost"])
    frame["gpu_hours"] = frame["training_cost_raw"].apply(extract_gpu_hours)
    frame["training_group_label"] = frame["training_group"].map(TRAINING_GROUP_LABELS).fillna(frame["training_group"])
    return frame


def pareto_frontier(frame: pd.DataFrame, cost_column: str, score_column: str) -> pd.DataFrame:
    ordered = frame.sort_values([cost_column, score_column], ascending=[True, False]).reset_index(drop=True)
    rows = []
    best_score = -np.inf
    for row in ordered.itertuples(index=False):
        score = getattr(row, score_column)
        if score > best_score:
            rows.append(row._asdict())
            best_score = score
    return pd.DataFrame(rows)


def build_budget_recommendations(frame: pd.DataFrame) -> pd.DataFrame:
    q1, q2 = frame["log_proxy_cost"].quantile([0.33, 0.66]).tolist()
    budget_labels = pd.cut(
        frame["log_proxy_cost"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["low", "mid", "high"],
    )
    frame = frame.assign(budget_tier=budget_labels)
    rows: list[dict] = []
    for tier in ["low", "mid", "high"]:
        subset = frame.loc[frame["budget_tier"] == tier].copy()
        top = subset.sort_values(["f1_full_test", "mae_full_test"], ascending=[False, True]).iloc[0]
        cheapest = subset.sort_values(["log_proxy_cost", "f1_full_test"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "budget_tier": tier,
                "best_model": top["model_key"],
                "best_family": top["family"],
                "best_training_combo": top["training_combo"],
                "best_f1": top["f1_full_test"],
                "best_mae": top["mae_full_test"],
                "best_log_proxy_cost": top["log_proxy_cost"],
                "cheapest_model": cheapest["model_key"],
                "cheapest_f1": cheapest["f1_full_test"],
                "cheapest_log_proxy_cost": cheapest["log_proxy_cost"],
            }
        )
    return pd.DataFrame(rows)


def build_strategy_comparison(
    metadata: pd.DataFrame,
    parameter_fits: pd.DataFrame,
    data_fits: pd.DataFrame,
    ensemble_curve: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    parameter_main = parameter_fits[
        (parameter_fits["subset"] == "MPtrj__OMat24__sAlex")
        & (parameter_fits["metric"] == "f1_full_test")
    ].iloc[0]
    rows.append(
        {
            "strategy": "A_fixed_data_scale_params",
            "view": "train_budget",
            "summary": "Increase parameter count within the strongest shared data regime.",
            "f1_gain_per_log10_step": parameter_main["slope_per_log10"],
            "reference_subset": parameter_main["subset"],
            "best_k_or_note": "single model",
        }
    )

    data_main = data_fits[
        (data_fits["analysis"] == "data_scaling")
        & (data_fits["subset"] == "pooled_family_combo_means")
        & (data_fits["metric"] == "f1_full_test")
    ].iloc[0]
    rows.append(
        {
            "strategy": "B_fixed_family_scale_data",
            "view": "train_budget",
            "summary": "Increase training-set size across repeated model families.",
            "f1_gain_per_log10_step": data_main["slope_per_log10"],
            "reference_subset": data_main["subset"],
            "best_k_or_note": "single model",
        }
    )

    ranked = metadata.sort_values(["f1_full_test", "r2_full_test"], ascending=[False, False]).copy()
    ranked["cumulative_proxy_cost"] = ranked["proxy_cost"].cumsum()
    ranked["ensemble_size"] = range(1, len(ranked) + 1)
    ensemble = ensemble_curve.merge(
        ranked[["ensemble_size", "cumulative_proxy_cost"]],
        on="ensemble_size",
        how="left",
    )
    base = ensemble.iloc[0]
    best = ensemble.loc[ensemble["f1"].idxmax()]
    rows.append(
        {
            "strategy": "C_prefix_ensemble",
            "view": "train_budget",
            "summary": "Prefix ensemble using cumulative training-equivalent cost.",
            "f1_gain_per_log10_step": (best["f1"] - base["f1"])
            / (np.log10(best["cumulative_proxy_cost"]) - np.log10(base["cumulative_proxy_cost"])),
            "reference_subset": "top-ranked prefix ensemble",
            "best_k_or_note": int(best["ensemble_size"]),
        }
    )
    rows.append(
        {
            "strategy": "C_prefix_ensemble",
            "view": "checkpoint_reuse",
            "summary": "If checkpoints are already available, ensemble a few strong models.",
            "f1_gain_per_log10_step": best["f1"] - base["f1"],
            "reference_subset": "top-ranked prefix ensemble",
            "best_k_or_note": int(best["ensemble_size"]),
        }
    )

    return pd.DataFrame(rows)


def make_figure(
    metadata: pd.DataFrame,
    frontier: pd.DataFrame,
    budget_recommendations: pd.DataFrame,
    output_path,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))
    palette = sns.color_palette("Set2", n_colors=len(TRAINING_GROUP_LABELS))
    training_color_map = dict(zip(TRAINING_GROUP_LABELS.values(), palette))

    for architecture_group, marker in ARCH_MARKERS.items():
        subset = metadata.loc[metadata["architecture_group"] == architecture_group].copy()
        ax.scatter(
            subset["proxy_cost"],
            subset["f1_full_test"],
            s=80,
            marker=marker,
            c=subset["training_group_label"].map(training_color_map),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
            label=architecture_group,
        )

    frontier = frontier.sort_values("proxy_cost")
    ax.plot(frontier["proxy_cost"], frontier["f1_full_test"], color="black", linewidth=2.2, zorder=4)

    annotate_models = budget_recommendations["best_model"].tolist() + ["orb-v3"]
    annotate_models = [m for m in annotate_models if m in frontier["model_key"].tolist()]
    for model_key in annotate_models:
        row = frontier.loc[frontier["model_key"] == model_key].iloc[0]
        ax.annotate(
            model_key,
            (row["proxy_cost"], row["f1_full_test"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Proxy training cost (training structures x parameters)")
    ax.set_ylabel("F1")
    ax.set_title("Performance vs proxy cost with Pareto frontier")

    training_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="black", markersize=8, label=label)
        for label, color in training_color_map.items()
    ]
    arch_handles = [
        plt.Line2D([0], [0], marker=marker, color="black", linestyle="none", markersize=8, label=label)
        for label, marker in ARCH_MARKERS.items()
    ]
    first_legend = ax.legend(
        handles=training_handles,
        title="Training data",
        loc="lower right",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=arch_handles,
        title="Architecture",
        loc="lower center",
        bbox_to_anchor=(0.72, 0.18),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metadata = pd.read_csv(f"data/processed/model_metadata_{SNAPSHOT_LABEL}.csv")
    parameter_fits = pd.read_csv("results/tables/analysis_03/parameter_scaling_fits.csv")
    data_fits = pd.read_csv("results/tables/analysis_03/data_scaling_fits.csv")
    ensemble_curve = pd.read_csv("results/tables/analysis_03/ensemble_scaling_curve.csv")

    output_table_dir = RESULTS_DIR / "tables" / "analysis_05"
    output_figure_dir = RESULTS_DIR / "figures" / "analysis_05"
    ensure_dir(output_table_dir)
    ensure_dir(output_figure_dir)

    metadata = compute_proxy_costs(metadata)
    metadata.to_csv(output_table_dir / "model_cost_summary.csv", index=False)

    proxy_frontier = pareto_frontier(metadata, "proxy_cost", "f1_full_test")
    proxy_frontier.to_csv(output_table_dir / "pareto_frontier_proxy.csv", index=False)

    gpu_subset = metadata.dropna(subset=["gpu_hours"]).copy()
    gpu_frontier = pareto_frontier(gpu_subset, "gpu_hours", "f1_full_test")
    gpu_frontier.to_csv(output_table_dir / "pareto_frontier_gpu_hours.csv", index=False)

    budget_recommendations = build_budget_recommendations(metadata)
    budget_recommendations.to_csv(output_table_dir / "budget_recommendations.csv", index=False)

    strategy_comparison = build_strategy_comparison(metadata, parameter_fits, data_fits, ensemble_curve)
    strategy_comparison.to_csv(output_table_dir / "strategy_comparison.csv", index=False)

    summary = {
        "proxy_frontier_models": proxy_frontier["model_key"].tolist(),
        "gpu_frontier_models": gpu_frontier["model_key"].tolist(),
        "best_low_budget_model": budget_recommendations.loc[
            budget_recommendations["budget_tier"] == "low", "best_model"
        ].iloc[0],
        "best_mid_budget_model": budget_recommendations.loc[
            budget_recommendations["budget_tier"] == "mid", "best_model"
        ].iloc[0],
        "best_high_budget_model": budget_recommendations.loc[
            budget_recommendations["budget_tier"] == "high", "best_model"
        ].iloc[0],
    }
    (output_table_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_figure(metadata, proxy_frontier, budget_recommendations, output_figure_dir / "pareto_frontier.png")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


SNAPSHOT_LABEL = "snapshot_45"
METRICS = {
    "f1_full_test": "F1",
    "mae_full_test": "MAE",
    "daf_full_test": "DAF",
    "r2_full_test": "R2",
}
TRAINING_FACTOR_VARIANTS = {
    "coarse": ("C(training_group)", "Training data (coarse)"),
    "exact": ("C(training_combo)", "Training data"),
}
BASE_TERMS = [
    ("C(architecture_group)", "Architecture"),
    ("log_model_params", "Parameters"),
]
COST_TERM = ("log_gpu_hours", "Training cost")
PLOT_COLORS = {
    "Training data": "#0b6e4f",
    "Architecture": "#f4a259",
    "Parameters": "#5b8e7d",
}


def parse_gpu_hours(raw_value) -> float | float("nan"):
    if pd.isna(raw_value):
        return np.nan
    if raw_value == "missing":
        return np.nan
    if isinstance(raw_value, str) and raw_value.startswith("{"):
        return extract_gpu_hours(json.loads(raw_value))
    if isinstance(raw_value, str):
        match = re.search(r"(\d+)\s*[x×]\s*[^x×]+[x×]\s*(\d+(?:\.\d+)?)\s*h", raw_value)
        if match:
            return float(match.group(1)) * float(match.group(2))
    return np.nan


def extract_gpu_hours(payload) -> float | float("nan"):
    if isinstance(payload, dict):
        if {"amount", "hours"} <= set(payload):
            amount = payload.get("amount")
            hours = payload.get("hours")
            if amount is not None and hours is not None:
                return float(amount) * float(hours)
        total = 0.0
        found = False
        for value in payload.values():
            extracted = extract_gpu_hours(value)
            if not math.isnan(extracted):
                total += extracted
                found = True
        return total if found else np.nan
    if isinstance(payload, list):
        total = 0.0
        found = False
        for value in payload:
            extracted = extract_gpu_hours(value)
            if not math.isnan(extracted):
                total += extracted
                found = True
        return total if found else np.nan
    return np.nan


def incremental_r2(
    frame: pd.DataFrame,
    metric: str,
    ordered_terms: list[tuple[str, str]],
    variant: str,
) -> list[dict]:
    rows: list[dict] = []
    previous_r2 = 0.0
    active_terms: list[str] = []

    for step_index, (term, label) in enumerate(ordered_terms, start=1):
        active_terms.append(term)
        formula = f"{metric} ~ " + " + ".join(active_terms)
        model = smf.ols(formula, data=frame).fit()
        rows.append(
            {
                "metric": metric,
                "metric_label": METRICS[metric],
                "step": step_index,
                "factor": label,
                "formula": formula,
                "r2": model.rsquared,
                "delta_r2": model.rsquared - previous_r2,
                "n_models": int(model.nobs),
                "variant": variant,
            }
        )
        previous_r2 = model.rsquared

    return rows


def fit_anova(
    frame: pd.DataFrame,
    metric: str,
    training_term: tuple[str, str],
    include_cost: bool,
    variant: str,
) -> tuple[pd.DataFrame, object]:
    terms = [training_term, *BASE_TERMS]
    if include_cost:
        terms.append(COST_TERM)
    formula = f"{metric} ~ " + " + ".join(term for term, _ in terms)
    model = smf.ols(formula, data=frame).fit()
    anova = anova_lm(model, typ=3).reset_index().rename(columns={"index": "term"})
    factor_labels = {term: label for term, label in terms}
    anova["factor"] = anova["term"].map(factor_labels).fillna(anova["term"])
    residual_ss = float(anova.loc[anova["term"] == "Residual", "sum_sq"].iloc[0])
    anova["partial_eta_sq"] = anova["sum_sq"] / (anova["sum_sq"] + residual_ss)
    anova["metric"] = metric
    anova["metric_label"] = METRICS[metric]
    anova["n_models"] = int(model.nobs)
    anova["include_cost"] = include_cost
    anova["variant"] = variant
    return anova, model


def make_plot(primary_anova: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ordered_factors = ["Training data", "Architecture", "Parameters"]

    for ax, metric in zip(axes.flat, METRICS):
        subset = primary_anova[
            (primary_anova["metric"] == metric)
            & (primary_anova["factor"].isin(ordered_factors))
        ].copy()
        subset["factor"] = pd.Categorical(subset["factor"], categories=ordered_factors, ordered=True)
        subset = subset.sort_values("factor")
        ax.bar(
            subset["factor"],
            subset["partial_eta_sq"],
            color=[PLOT_COLORS[factor] for factor in subset["factor"]],
        )
        ax.set_title(METRICS[metric])
        ax.set_ylabel("Partial eta squared")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=22)
        ax.set_ylim(0, max(0.05, subset["partial_eta_sq"].max() * 1.2))

    fig.suptitle("Variance decomposition on the frozen 45-model snapshot", fontsize=14)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_table_dir = RESULTS_DIR / "tables" / "analysis_01"
    output_figure_dir = RESULTS_DIR / "figures" / "analysis_01"
    ensure_dir(output_table_dir)
    ensure_dir(output_figure_dir)

    metadata = pd.read_csv(PROCESSED_DIR / f"model_metadata_{SNAPSHOT_LABEL}.csv")
    metadata["log_model_params"] = np.log10(metadata["model_params"].astype(float))
    metadata["gpu_hours"] = metadata["training_cost_raw"].apply(parse_gpu_hours)
    metadata["log_gpu_hours"] = np.log10(metadata["gpu_hours"])
    metadata["log_train_size"] = np.log10(metadata["effective_training_structures"].astype(float))

    primary_anova_rows: list[pd.DataFrame] = []
    cost_anova_rows: list[pd.DataFrame] = []
    size_sensitivity_rows: list[pd.DataFrame] = []
    forward_r2_rows: list[dict] = []
    reverse_r2_rows: list[dict] = []
    summary: dict[str, dict] = {}

    for metric in METRICS:
        summary[metric] = {}

        for variant, training_term in TRAINING_FACTOR_VARIANTS.items():
            training_column = "training_group" if variant == "coarse" else "training_combo"
            primary_frame = metadata.dropna(
                subset=[metric, training_column, "architecture_group", "log_model_params"]
            ).copy()
            anova_frame, primary_model = fit_anova(
                primary_frame,
                metric,
                training_term=training_term,
                include_cost=False,
                variant=variant,
            )
            primary_anova_rows.append(anova_frame)
            ordered_terms = [training_term, *BASE_TERMS]
            forward_r2_rows.extend(incremental_r2(primary_frame, metric, ordered_terms, variant))
            reverse_r2_rows.extend(
                incremental_r2(primary_frame, metric, list(reversed(ordered_terms)), variant)
            )

            summary[metric][variant] = {
                "n_primary_models": int(primary_model.nobs),
                "primary_r2": primary_model.rsquared,
                "primary_adj_r2": primary_model.rsquared_adj,
            }

            cost_frame = metadata.dropna(
                subset=[
                    metric,
                    training_column,
                    "architecture_group",
                    "log_model_params",
                    "log_gpu_hours",
                ]
            ).copy()
            if len(cost_frame) >= 8:
                cost_anova, cost_model = fit_anova(
                    cost_frame,
                    metric,
                    training_term=training_term,
                    include_cost=True,
                    variant=variant,
                )
                cost_anova_rows.append(cost_anova)
                summary[metric][variant]["n_cost_models"] = int(cost_model.nobs)
                summary[metric][variant]["cost_r2"] = cost_model.rsquared
                summary[metric][variant]["cost_adj_r2"] = cost_model.rsquared_adj
            else:
                summary[metric][variant]["n_cost_models"] = int(len(cost_frame))

        size_frame = metadata.dropna(
            subset=[metric, "log_train_size", "architecture_group", "log_model_params"]
        ).copy()
        size_model = smf.ols(
            f"{metric} ~ log_train_size + C(architecture_group) + log_model_params",
            data=size_frame,
        ).fit()
        size_anova = anova_lm(size_model, typ=3).reset_index().rename(columns={"index": "term"})
        residual_ss = float(size_anova.loc[size_anova["term"] == "Residual", "sum_sq"].iloc[0])
        size_anova["factor"] = size_anova["term"].map(
            {
                "log_train_size": "Training data size",
                "C(architecture_group)": "Architecture",
                "log_model_params": "Parameters",
            }
        ).fillna(size_anova["term"])
        size_anova["partial_eta_sq"] = size_anova["sum_sq"] / (size_anova["sum_sq"] + residual_ss)
        size_anova["metric"] = metric
        size_anova["metric_label"] = METRICS[metric]
        size_anova["n_models"] = int(size_model.nobs)
        size_anova["include_cost"] = False
        size_anova["variant"] = "size"
        size_sensitivity_rows.append(size_anova)
        summary[metric]["size"] = {
            "n_primary_models": int(size_model.nobs),
            "primary_r2": size_model.rsquared,
            "primary_adj_r2": size_model.rsquared_adj,
        }

    primary_anova = pd.concat(primary_anova_rows, ignore_index=True)
    cost_anova = pd.concat(cost_anova_rows, ignore_index=True) if cost_anova_rows else pd.DataFrame()
    size_sensitivity = pd.concat(size_sensitivity_rows, ignore_index=True)
    forward_r2 = pd.DataFrame(forward_r2_rows)
    reverse_r2 = pd.DataFrame(reverse_r2_rows)

    primary_anova.to_csv(output_table_dir / "anova_primary.csv", index=False)
    if not cost_anova.empty:
        cost_anova.to_csv(output_table_dir / "anova_cost_subset.csv", index=False)
    size_sensitivity.to_csv(output_table_dir / "anova_train_size_sensitivity.csv", index=False)
    forward_r2.to_csv(output_table_dir / "r2_forward.csv", index=False)
    reverse_r2.to_csv(output_table_dir / "r2_reverse.csv", index=False)
    (output_table_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_plot(
        primary_anova[
            (primary_anova["variant"] == "exact")
            & (primary_anova["term"] != "Residual")
        ].copy(),
        output_figure_dir / "variance_decomposition_primary.png",
    )


if __name__ == "__main__":
    main()

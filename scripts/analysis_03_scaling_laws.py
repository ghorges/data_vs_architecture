from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from sklearn.metrics import f1_score

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


SNAPSHOT_LABEL = "snapshot_45"
PARAMETER_SCALING_COMBOS = ["MPtrj__OMat24__sAlex", "MPtrj"]
METRICS = {
    "f1_full_test": "F1",
    "mae_full_test": "MAE",
    "daf_full_test": "DAF",
    "r2_full_test": "R2",
}
OUTLIER_ABS_THRESHOLD = 10.0


@dataclass
class FitSummary:
    analysis: str
    subset: str
    metric: str
    metric_label: str
    n_points: int
    slope_per_log10: float
    intercept: float
    r_value: float
    r_squared: float
    p_value: float
    stderr: float


def fit_log_scaling(x: pd.Series, y: pd.Series, analysis: str, subset: str, metric: str) -> FitSummary:
    log_x = np.log10(x.astype(float))
    result = linregress(log_x, y.astype(float))
    return FitSummary(
        analysis=analysis,
        subset=subset,
        metric=metric,
        metric_label=METRICS[metric],
        n_points=len(x),
        slope_per_log10=result.slope,
        intercept=result.intercept,
        r_value=result.rvalue,
        r_squared=result.rvalue**2,
        p_value=result.pvalue,
        stderr=result.stderr,
    )


def evaluate_parameter_scaling(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows: list[dict] = []
    fit_rows: list[dict] = []

    for combo in PARAMETER_SCALING_COMBOS:
        subset = metadata.loc[metadata["training_combo"] == combo].copy()
        if len(subset) < 4:
            continue
        subset["log_model_params"] = np.log10(subset["model_params"].astype(float))
        for row in subset.itertuples(index=False):
            point_rows.append(
                {
                    "analysis": "parameter_scaling",
                    "subset": combo,
                    "model_key": row.model_key,
                    "family": row.family,
                    "architecture_group": row.architecture_group,
                    "model_params": row.model_params,
                    "log_model_params": row.log_model_params,
                    **{metric: getattr(row, metric) for metric in METRICS},
                }
            )
        for metric in METRICS:
            fit_rows.append(
                asdict(
                    fit_log_scaling(
                        subset["model_params"],
                        subset[metric],
                        analysis="parameter_scaling",
                        subset=combo,
                        metric=metric,
                    )
                )
            )

    return pd.DataFrame(point_rows), pd.DataFrame(fit_rows)


def evaluate_data_scaling(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    fit_rows: list[dict] = []

    grouped = (
        metadata.groupby(["family", "training_combo"], as_index=False)
        .agg(
            architecture_group=("architecture_group", "first"),
            effective_training_structures=("effective_training_structures", "mean"),
            n_models=("model_key", "size"),
            f1_full_test=("f1_full_test", "mean"),
            mae_full_test=("mae_full_test", "mean"),
            daf_full_test=("daf_full_test", "mean"),
            r2_full_test=("r2_full_test", "mean"),
        )
        .sort_values(["family", "effective_training_structures"])
        .reset_index(drop=True)
    )
    grouped["log_train_size"] = np.log10(grouped["effective_training_structures"].astype(float))

    repeated = grouped.groupby("family").filter(lambda frame: frame["training_combo"].nunique() >= 2).copy()
    for row in repeated.itertuples(index=False):
        rows.append(
            {
                "analysis": "data_scaling",
                "family": row.family,
                "training_combo": row.training_combo,
                "architecture_group": row.architecture_group,
                "effective_training_structures": row.effective_training_structures,
                "log_train_size": row.log_train_size,
                "n_models": row.n_models,
                **{metric: getattr(row, metric) for metric in METRICS},
            }
        )

    for metric in METRICS:
        fit_rows.append(
            asdict(
                fit_log_scaling(
                    repeated["effective_training_structures"],
                    repeated[metric],
                    analysis="data_scaling",
                    subset="pooled_family_combo_means",
                    metric=metric,
                )
            )
        )

    for family, family_frame in repeated.groupby("family"):
        if len(family_frame) < 2:
            continue
        for metric in METRICS:
            fit_rows.append(
                asdict(
                    fit_log_scaling(
                        family_frame["effective_training_structures"],
                        family_frame[metric],
                        analysis="data_scaling_family",
                        subset=family,
                        metric=metric,
                    )
                )
            )

    return pd.DataFrame(rows), pd.DataFrame(fit_rows)


def evaluate_ensemble_scaling(
    metadata: pd.DataFrame,
    prediction_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ranked_models = metadata.sort_values(
        ["f1_full_test", "r2_full_test"],
        ascending=[False, False],
    )["model_key"].tolist()
    reference_hull_energy = (
        prediction_matrix["e_form_per_atom_mp2020_corrected"]
        - prediction_matrix["e_above_hull_mp2020_corrected_ppd_mp"]
    )
    stable_labels = prediction_matrix["e_above_hull_mp2020_corrected_ppd_mp"] <= 0

    outlier_rows: list[dict] = []
    cleaned = prediction_matrix.copy()
    for model in ranked_models:
        mask = cleaned[model].abs() > OUTLIER_ABS_THRESHOLD
        outlier_rows.append(
            {
                "model_key": model,
                "n_outliers_abs_gt_10": int(mask.sum()),
                "max_abs_prediction": float(cleaned[model].abs().max()),
            }
        )
        cleaned.loc[mask, model] = np.nan

    curve_rows: list[dict] = []
    best_f1 = -np.inf
    best_f1_k = None
    best_mae = np.inf
    best_mae_k = None

    for k in range(1, len(ranked_models) + 1):
        subset = ranked_models[:k]
        ensemble_prediction = cleaned[subset].mean(axis=1)
        predicted_ehull = ensemble_prediction - reference_hull_energy
        f1 = f1_score(stable_labels, predicted_ehull <= 0)
        mae = (ensemble_prediction - cleaned["e_form_per_atom_wbm"]).abs().mean()
        centered = cleaned["e_form_per_atom_wbm"] - cleaned["e_form_per_atom_wbm"].mean()
        r2 = 1 - ((ensemble_prediction - cleaned["e_form_per_atom_wbm"]) ** 2).sum() / (centered**2).sum()
        curve_rows.append(
            {
                "ensemble_size": k,
                "f1": f1,
                "mae": mae,
                "r2": r2,
                "top_model_1": subset[0],
                "top_model_3": ",".join(subset[:3]),
            }
        )
        if f1 > best_f1:
            best_f1 = f1
            best_f1_k = k
        if mae < best_mae:
            best_mae = mae
            best_mae_k = k

    summary = {
        "ranking_rule": "Sort models by descending published F1, then descending published R2.",
        "outlier_abs_threshold": OUTLIER_ABS_THRESHOLD,
        "best_f1": best_f1,
        "best_f1_k": best_f1_k,
        "best_mae": best_mae,
        "best_mae_k": best_mae_k,
        "top_3_models": ranked_models[:3],
        "top_10_models": ranked_models[:10],
    }
    return pd.DataFrame(curve_rows), pd.DataFrame(outlier_rows), summary


def make_figure(
    parameter_points: pd.DataFrame,
    parameter_fits: pd.DataFrame,
    data_points: pd.DataFrame,
    data_fits: pd.DataFrame,
    ensemble_curve: pd.DataFrame,
    output_path,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    # Panel A: parameter scaling on the highest-data combo
    combo = "MPtrj__OMat24__sAlex"
    ax = axes[0]
    panel = parameter_points[parameter_points["subset"] == combo].copy().sort_values("model_params")
    sns.scatterplot(
        data=panel,
        x="model_params",
        y="f1_full_test",
        hue="architecture_group",
        palette="Set2",
        s=80,
        ax=ax,
    )
    fit = parameter_fits[
        (parameter_fits["subset"] == combo) & (parameter_fits["metric"] == "f1_full_test")
    ].iloc[0]
    x_grid = np.geomspace(panel["model_params"].min(), panel["model_params"].max(), 200)
    y_grid = fit["intercept"] + fit["slope_per_log10"] * np.log10(x_grid)
    ax.plot(x_grid, y_grid, color="black", linewidth=2, label="Log-linear fit")
    ax.set_xscale("log")
    ax.set_title("Parameter scaling\n(fixed training data)")
    ax.set_xlabel("Model parameters")
    ax.set_ylabel("F1")
    ax.legend(frameon=False, fontsize=8)

    # Panel B: data scaling across repeated families
    ax = axes[1]
    palette = sns.color_palette("tab20", n_colors=data_points["family"].nunique())
    family_colors = dict(zip(sorted(data_points["family"].unique()), palette))
    for family, family_frame in data_points.groupby("family"):
        family_frame = family_frame.sort_values("effective_training_structures")
        ax.plot(
            family_frame["effective_training_structures"],
            family_frame["f1_full_test"],
            marker="o",
            linewidth=1.5,
            color=family_colors[family],
            alpha=0.9,
            label=family,
        )
    pooled_fit = data_fits[
        (data_fits["analysis"] == "data_scaling")
        & (data_fits["subset"] == "pooled_family_combo_means")
        & (data_fits["metric"] == "f1_full_test")
    ].iloc[0]
    x_grid = np.geomspace(
        data_points["effective_training_structures"].min(),
        data_points["effective_training_structures"].max(),
        200,
    )
    y_grid = pooled_fit["intercept"] + pooled_fit["slope_per_log10"] * np.log10(x_grid)
    ax.plot(x_grid, y_grid, color="black", linestyle="--", linewidth=2.5, label="Pooled fit")
    ax.set_xscale("log")
    ax.set_title("Data scaling\n(fixed family, pooled fit)")
    ax.set_xlabel("Effective training structures")
    ax.set_ylabel("F1")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    # Panel C: ensemble scaling
    ax = axes[2]
    ax.plot(ensemble_curve["ensemble_size"], ensemble_curve["f1"], color="#1f77b4", linewidth=2.5)
    ax.set_title("Ensemble scaling")
    ax.set_xlabel("Ensemble size")
    ax.set_ylabel("F1", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    best_row = ensemble_curve.loc[ensemble_curve["f1"].idxmax()]
    ax.axvline(best_row["ensemble_size"], color="#1f77b4", linestyle=":", linewidth=1.5)
    ax.text(
        best_row["ensemble_size"] + 0.5,
        best_row["f1"],
        f"best F1 @ k={int(best_row['ensemble_size'])}",
        color="#1f77b4",
        fontsize=9,
        va="bottom",
    )
    ax2 = ax.twinx()
    ax2.plot(ensemble_curve["ensemble_size"], ensemble_curve["mae"], color="#e45756", linewidth=2, alpha=0.8)
    ax2.set_ylabel("MAE", color="#e45756")
    ax2.tick_params(axis="y", labelcolor="#e45756")

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_table_dir = RESULTS_DIR / "tables" / "analysis_03"
    output_figure_dir = RESULTS_DIR / "figures" / "analysis_03"
    ensure_dir(output_table_dir)
    ensure_dir(output_figure_dir)

    metadata = pd.read_csv(PROCESSED_DIR / f"model_metadata_{SNAPSHOT_LABEL}.csv")
    prediction_matrix = pd.read_parquet(PROCESSED_DIR / f"discovery_prediction_matrix_{SNAPSHOT_LABEL}.parquet")

    parameter_points, parameter_fits = evaluate_parameter_scaling(metadata)
    data_points, data_fits = evaluate_data_scaling(metadata)
    ensemble_curve, ensemble_outliers, ensemble_summary = evaluate_ensemble_scaling(
        metadata,
        prediction_matrix,
    )

    parameter_points.to_csv(output_table_dir / "parameter_scaling_points.csv", index=False)
    parameter_fits.to_csv(output_table_dir / "parameter_scaling_fits.csv", index=False)
    data_points.to_csv(output_table_dir / "data_scaling_points.csv", index=False)
    data_fits.to_csv(output_table_dir / "data_scaling_fits.csv", index=False)
    ensemble_curve.to_csv(output_table_dir / "ensemble_scaling_curve.csv", index=False)
    ensemble_outliers.to_csv(output_table_dir / "ensemble_outlier_audit.csv", index=False)
    (output_table_dir / "ensemble_summary.json").write_text(
        json.dumps(ensemble_summary, indent=2),
        encoding="utf-8",
    )

    make_figure(
        parameter_points,
        parameter_fits,
        data_points,
        data_fits,
        ensemble_curve,
        output_figure_dir / "scaling_laws.png",
    )


if __name__ == "__main__":
    main()

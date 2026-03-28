from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
from sklearn.metrics import adjusted_rand_score
from statsmodels.stats.anova import anova_lm

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


SNAPSHOT_LABEL = "snapshot_45"
RANDOM_SEED = 42
N_RESAMPLES = 2000
METRICS = {
    "f1_full_test": "F1",
    "mae_full_test": "MAE",
    "daf_full_test": "DAF",
    "r2_full_test": "R2",
}
PARAMETER_SCALING_COMBOS = ["MPtrj__OMat24__sAlex", "MPtrj"]


@dataclass
class SlopeRecord:
    analysis: str
    subset: str
    metric: str
    metric_label: str
    n_points: int
    slope_per_log10: float
    intercept: float
    r_squared: float
    p_value: float
    stderr: float


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def fit_exact_anova(metadata: pd.DataFrame, metric: str) -> dict:
    frame = metadata.copy()
    frame["log_model_params"] = np.log10(frame["model_params"].astype(float))
    model = smf.ols(
        f"{metric} ~ C(training_combo) + C(architecture_group) + log_model_params",
        data=frame,
    ).fit()
    anova = anova_lm(model, typ=3).reset_index().rename(columns={"index": "term"})
    residual_ss = float(anova.loc[anova["term"] == "Residual", "sum_sq"].iloc[0])
    eta_lookup = {
        row.term: row.sum_sq / (row.sum_sq + residual_ss)
        for row in anova.itertuples(index=False)
    }
    return {
        "metric": metric,
        "metric_label": METRICS[metric],
        "n_models": int(model.nobs),
        "train_eta_sq": float(eta_lookup["C(training_combo)"]),
        "arch_eta_sq": float(eta_lookup["C(architecture_group)"]),
        "param_eta_sq": float(eta_lookup["log_model_params"]),
        "train_minus_arch": float(eta_lookup["C(training_combo)"] - eta_lookup["C(architecture_group)"]),
        "model_r2": float(model.rsquared),
    }


def cluster_alignment(metadata: pd.DataFrame, model_order: list[str], full_distance_square: np.ndarray) -> dict:
    subset_indices = [model_order.index(model_key) for model_key in metadata["model_key"]]
    subset_square = full_distance_square[np.ix_(subset_indices, subset_indices)]
    condensed = squareform(subset_square, checks=False)
    linkage_matrix = linkage(condensed, method="ward")

    train_clusters = fcluster(
        linkage_matrix,
        t=metadata["training_combo"].nunique(),
        criterion="maxclust",
    )
    arch_clusters = fcluster(
        linkage_matrix,
        t=metadata["architecture_group"].nunique(),
        criterion="maxclust",
    )

    ari_training = float(adjusted_rand_score(metadata["training_combo"], train_clusters))
    ari_architecture = float(adjusted_rand_score(metadata["architecture_group"], arch_clusters))
    return {
        "n_models": int(len(metadata)),
        "ari_training_exact": ari_training,
        "ari_architecture": ari_architecture,
        "ari_train_minus_arch": float(ari_training - ari_architecture),
    }


def one_per_family_resampling(
    metadata: pd.DataFrame,
    model_order: list[str],
    full_distance_square: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = sorted(metadata["family"].unique())
    anova_rows: list[dict] = []
    cluster_rows: list[dict] = []

    for draw_idx in range(N_RESAMPLES):
        chosen_keys: list[str] = []
        for family in families:
            family_keys = metadata.loc[metadata["family"] == family, "model_key"].tolist()
            chosen_keys.append(rng.choice(family_keys))

        subset = metadata.loc[metadata["model_key"].isin(chosen_keys)].copy()
        subset = subset.set_index("model_key").loc[chosen_keys].reset_index()

        for metric in METRICS:
            row = fit_exact_anova(subset, metric)
            row["draw"] = draw_idx
            anova_rows.append(row)

        cluster_row = cluster_alignment(subset, model_order, full_distance_square)
        cluster_row["draw"] = draw_idx
        cluster_rows.append(cluster_row)

    return pd.DataFrame(anova_rows), pd.DataFrame(cluster_rows)


def fit_slope(x: pd.Series, y: pd.Series, analysis: str, subset: str, metric: str) -> SlopeRecord:
    log_x = np.log10(x.astype(float))
    result = linregress(log_x, y.astype(float))
    return SlopeRecord(
        analysis=analysis,
        subset=subset,
        metric=metric,
        metric_label=METRICS[metric],
        n_points=int(len(x)),
        slope_per_log10=float(result.slope),
        intercept=float(result.intercept),
        r_squared=float(result.rvalue**2),
        p_value=float(result.pvalue),
        stderr=float(result.stderr),
    )


def bootstrap_parameter_scaling(
    metadata: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_rows: list[dict] = []
    bootstrap_rows: list[dict] = []

    for combo in PARAMETER_SCALING_COMBOS:
        subset = metadata.loc[metadata["training_combo"] == combo].copy().reset_index(drop=True)
        if len(subset) < 4:
            continue

        for metric in METRICS:
            observed_rows.append(asdict(fit_slope(subset["model_params"], subset[metric], "parameter_scaling", combo, metric)))

        for draw_idx in range(N_RESAMPLES):
            sampled = subset.iloc[rng.integers(0, len(subset), size=len(subset))].copy()
            if sampled["model_params"].nunique() < 2:
                continue
            for metric in METRICS:
                record = asdict(fit_slope(sampled["model_params"], sampled[metric], "parameter_scaling_bootstrap", combo, metric))
                record["draw"] = draw_idx
                bootstrap_rows.append(record)

    return pd.DataFrame(observed_rows), pd.DataFrame(bootstrap_rows)


def prepare_data_scaling_points(metadata: pd.DataFrame) -> pd.DataFrame:
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
    repeated = grouped.groupby("family").filter(lambda frame: frame["training_combo"].nunique() >= 2).copy()
    return repeated.reset_index(drop=True)


def bootstrap_data_scaling(
    metadata: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = prepare_data_scaling_points(metadata)
    observed_rows = [
        asdict(
            fit_slope(
                points["effective_training_structures"],
                points[metric],
                "data_scaling",
                "pooled_family_combo_means",
                metric,
            )
        )
        for metric in METRICS
    ]

    families = sorted(points["family"].unique())
    bootstrap_rows: list[dict] = []
    for draw_idx in range(N_RESAMPLES):
        sampled_families = rng.choice(families, size=len(families), replace=True)
        sampled = pd.concat(
            [points.loc[points["family"] == family].copy() for family in sampled_families],
            ignore_index=True,
        )
        if sampled["effective_training_structures"].nunique() < 2:
            continue
        for metric in METRICS:
            record = asdict(
                fit_slope(
                    sampled["effective_training_structures"],
                    sampled[metric],
                    "data_scaling_bootstrap",
                    "pooled_family_combo_means",
                    metric,
                )
            )
            record["draw"] = draw_idx
            bootstrap_rows.append(record)

    return pd.DataFrame(observed_rows), pd.DataFrame(bootstrap_rows)


def summarize_interval(values: pd.Series) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "q025": float(values.quantile(0.025)),
        "q05": float(values.quantile(0.05)),
        "q95": float(values.quantile(0.95)),
        "q975": float(values.quantile(0.975)),
    }


def summarize_anova_resampling(full_anova: pd.DataFrame, resample_anova: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric, metric_label in METRICS.items():
        full_row = full_anova.loc[full_anova["metric"] == metric].iloc[0]
        subset = resample_anova.loc[resample_anova["metric"] == metric]
        row = {
            "metric": metric,
            "metric_label": metric_label,
            "full_train_eta_sq": float(full_row["train_eta_sq"]),
            "full_arch_eta_sq": float(full_row["arch_eta_sq"]),
            "full_train_minus_arch": float(full_row["train_minus_arch"]),
            "fraction_train_gt_arch": float((subset["train_eta_sq"] > subset["arch_eta_sq"]).mean()),
            "fraction_delta_gt_zero": float((subset["train_minus_arch"] > 0).mean()),
        }
        for value_column in ["train_eta_sq", "arch_eta_sq", "train_minus_arch", "model_r2"]:
            stats = summarize_interval(subset[value_column])
            row.update({f"{value_column}_{key}": value for key, value in stats.items()})
        row["full_delta_below_q025"] = bool(row["full_train_minus_arch"] < row["train_minus_arch_q025"])
        row["full_delta_above_q975"] = bool(row["full_train_minus_arch"] > row["train_minus_arch_q975"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_cluster_resampling(full_cluster: pd.DataFrame, resample_cluster: pd.DataFrame) -> pd.DataFrame:
    full_row = full_cluster.iloc[0]
    row = {
        "full_ari_training_exact": float(full_row["ari_training_exact"]),
        "full_ari_architecture": float(full_row["ari_architecture"]),
        "full_ari_train_minus_arch": float(full_row["ari_train_minus_arch"]),
        "fraction_train_gt_arch": float(
            (resample_cluster["ari_training_exact"] > resample_cluster["ari_architecture"]).mean()
        ),
        "fraction_delta_gt_zero": float((resample_cluster["ari_train_minus_arch"] > 0).mean()),
    }
    for value_column in ["ari_training_exact", "ari_architecture", "ari_train_minus_arch"]:
        stats = summarize_interval(resample_cluster[value_column])
        row.update({f"{value_column}_{key}": value for key, value in stats.items()})
    row["full_delta_below_q025"] = bool(row["full_ari_train_minus_arch"] < row["ari_train_minus_arch_q025"])
    row["full_delta_above_q975"] = bool(row["full_ari_train_minus_arch"] > row["ari_train_minus_arch_q975"])
    return pd.DataFrame([row])


def summarize_scaling_bootstrap(observed: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (analysis, subset, metric, metric_label), observed_subset in observed.groupby(
        ["analysis", "subset", "metric", "metric_label"]
    ):
        full_row = observed_subset.iloc[0]
        draw_subset = bootstrap.loc[
            (bootstrap["subset"] == subset)
            & (bootstrap["metric"] == metric)
            & (bootstrap["analysis"].str.contains("bootstrap"))
        ]
        if draw_subset.empty:
            continue
        row = {
            "analysis": analysis,
            "subset": subset,
            "metric": metric,
            "metric_label": metric_label,
            "observed_slope_per_log10": float(full_row["slope_per_log10"]),
            "observed_r_squared": float(full_row["r_squared"]),
            "fraction_slope_gt_zero": float((draw_subset["slope_per_log10"] > 0).mean()),
            "fraction_slope_lt_zero": float((draw_subset["slope_per_log10"] < 0).mean()),
        }
        for value_column in ["slope_per_log10", "r_squared"]:
            stats = summarize_interval(draw_subset[value_column])
            row.update({f"{value_column}_{key}": value for key, value in stats.items()})
        row["observed_slope_below_q025"] = bool(row["observed_slope_per_log10"] < row["slope_per_log10_q025"])
        row["observed_slope_above_q975"] = bool(row["observed_slope_per_log10"] > row["slope_per_log10_q975"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(
    anova_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    scaling_summary: pd.DataFrame,
) -> dict:
    return {
        "n_resamples": N_RESAMPLES,
        "anova": {
            row.metric_label: {
                "full_train_minus_arch": row.full_train_minus_arch,
                "train_minus_arch_q025": row.train_minus_arch_q025,
                "train_minus_arch_q975": row.train_minus_arch_q975,
                "fraction_delta_gt_zero": row.fraction_delta_gt_zero,
            }
            for row in anova_summary.itertuples(index=False)
        },
        "cluster_alignment": {
            "full_ari_train_minus_arch": float(cluster_summary.iloc[0]["full_ari_train_minus_arch"]),
            "ari_train_minus_arch_q025": float(cluster_summary.iloc[0]["ari_train_minus_arch_q025"]),
            "ari_train_minus_arch_q975": float(cluster_summary.iloc[0]["ari_train_minus_arch_q975"]),
            "fraction_delta_gt_zero": float(cluster_summary.iloc[0]["fraction_delta_gt_zero"]),
        },
        "scaling": {
            f"{row.analysis}::{row.subset}::{row.metric_label}": {
                "observed_slope_per_log10": row.observed_slope_per_log10,
                "slope_per_log10_q025": row.slope_per_log10_q025,
                "slope_per_log10_q975": row.slope_per_log10_q975,
                "fraction_slope_gt_zero": row.fraction_slope_gt_zero,
                "fraction_slope_lt_zero": row.fraction_slope_lt_zero,
            }
            for row in scaling_summary.itertuples(index=False)
        },
    }


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_08"
    ensure_dir(output_dir)

    rng = np.random.default_rng(RANDOM_SEED)

    metadata = pd.read_csv(PROCESSED_DIR / f"model_metadata_{SNAPSHOT_LABEL}.csv")
    errors = pd.read_parquet(PROCESSED_DIR / f"discovery_error_matrix_{SNAPSHOT_LABEL}.parquet")

    model_order = metadata["model_key"].tolist()
    error_matrix = errors[model_order].copy()
    filled = error_matrix.apply(lambda col: col.fillna(col.mean()), axis=0)
    model_vectors = zscore_rows(filled.to_numpy(dtype=float).T)
    full_distance_square = squareform(pdist(model_vectors, metric="euclidean"))

    full_anova = pd.DataFrame([fit_exact_anova(metadata, metric) for metric in METRICS])
    full_cluster = pd.DataFrame([cluster_alignment(metadata, model_order, full_distance_square)])
    resample_anova, resample_cluster = one_per_family_resampling(
        metadata,
        model_order,
        full_distance_square,
        rng,
    )

    parameter_observed, parameter_bootstrap = bootstrap_parameter_scaling(metadata, rng)
    data_observed, data_bootstrap = bootstrap_data_scaling(metadata, rng)
    scaling_observed = pd.concat([parameter_observed, data_observed], ignore_index=True)
    scaling_bootstrap = pd.concat([parameter_bootstrap, data_bootstrap], ignore_index=True)

    anova_summary = summarize_anova_resampling(full_anova, resample_anova)
    cluster_summary = summarize_cluster_resampling(full_cluster, resample_cluster)
    scaling_summary = summarize_scaling_bootstrap(scaling_observed, scaling_bootstrap)
    summary = build_summary(anova_summary, cluster_summary, scaling_summary)

    full_anova.to_csv(output_dir / "anova_full_snapshot.csv", index=False)
    resample_anova.to_csv(output_dir / "anova_one_per_family_resampling.csv", index=False)
    anova_summary.to_csv(output_dir / "anova_uncertainty_summary.csv", index=False)

    full_cluster.to_csv(output_dir / "cluster_full_snapshot.csv", index=False)
    resample_cluster.to_csv(output_dir / "cluster_one_per_family_resampling.csv", index=False)
    cluster_summary.to_csv(output_dir / "cluster_uncertainty_summary.csv", index=False)

    scaling_observed.to_csv(output_dir / "scaling_observed_fits.csv", index=False)
    scaling_bootstrap.to_csv(output_dir / "scaling_bootstrap_distribution.csv", index=False)
    scaling_summary.to_csv(output_dir / "scaling_uncertainty_summary.csv", index=False)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
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
N_RESAMPLES = 1000
RANDOM_SEED = 42


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

    return {
        "n_models": int(len(metadata)),
        "ari_training_exact": float(adjusted_rand_score(metadata["training_combo"], train_clusters)),
        "ari_architecture": float(adjusted_rand_score(metadata["architecture_group"], arch_clusters)),
        "ari_train_minus_arch": float(
            adjusted_rand_score(metadata["training_combo"], train_clusters)
            - adjusted_rand_score(metadata["architecture_group"], arch_clusters)
        ),
    }


def summarize_quantiles(frame: pd.DataFrame, value_columns: list[str], grouping_columns: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = frame.groupby(grouping_columns)
    for group_key, subset in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: value for column, value in zip(grouping_columns, group_key)}
        for value_column in value_columns:
            row[f"{value_column}_mean"] = subset[value_column].mean()
            row[f"{value_column}_median"] = subset[value_column].median()
            row[f"{value_column}_q05"] = subset[value_column].quantile(0.05)
            row[f"{value_column}_q95"] = subset[value_column].quantile(0.95)
        rows.append(row)
    return pd.DataFrame(rows)


def one_per_family_resampling(
    metadata: pd.DataFrame,
    model_order: list[str],
    full_distance_square: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
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


def leave_one_family_out(
    metadata: pd.DataFrame,
    model_order: list[str],
    full_distance_square: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = sorted(metadata["family"].unique())
    anova_rows: list[dict] = []
    cluster_rows: list[dict] = []

    for family in families:
        subset = metadata.loc[metadata["family"] != family].copy().reset_index(drop=True)
        for metric in METRICS:
            row = fit_exact_anova(subset, metric)
            row["left_out_family"] = family
            anova_rows.append(row)

        cluster_row = cluster_alignment(subset, model_order, full_distance_square)
        cluster_row["left_out_family"] = family
        cluster_rows.append(cluster_row)

    return pd.DataFrame(anova_rows), pd.DataFrame(cluster_rows)


def build_summary(
    metadata: pd.DataFrame,
    full_anova_rows: list[dict],
    full_cluster_row: dict,
    resample_anova: pd.DataFrame,
    resample_cluster: pd.DataFrame,
    loo_anova: pd.DataFrame,
    loo_cluster: pd.DataFrame,
) -> dict:
    metric_summary: dict[str, dict] = {}
    for metric, metric_label in METRICS.items():
        full_row = next(row for row in full_anova_rows if row["metric"] == metric)
        subset = resample_anova.loc[resample_anova["metric"] == metric]
        loo_subset = loo_anova.loc[loo_anova["metric"] == metric]
        metric_summary[metric_label] = {
            "full_snapshot": {
                "train_eta_sq": full_row["train_eta_sq"],
                "arch_eta_sq": full_row["arch_eta_sq"],
                "train_minus_arch": full_row["train_minus_arch"],
            },
            "one_per_family_resampling": {
                "n_draws": int(len(subset)),
                "train_eta_sq_median": float(subset["train_eta_sq"].median()),
                "train_eta_sq_q05": float(subset["train_eta_sq"].quantile(0.05)),
                "train_eta_sq_q95": float(subset["train_eta_sq"].quantile(0.95)),
                "arch_eta_sq_median": float(subset["arch_eta_sq"].median()),
                "arch_eta_sq_q05": float(subset["arch_eta_sq"].quantile(0.05)),
                "arch_eta_sq_q95": float(subset["arch_eta_sq"].quantile(0.95)),
                "fraction_train_gt_arch": float((subset["train_eta_sq"] > subset["arch_eta_sq"]).mean()),
            },
            "leave_one_family_out": {
                "n_families": int(len(loo_subset)),
                "min_train_minus_arch": float(loo_subset["train_minus_arch"].min()),
                "median_train_minus_arch": float(loo_subset["train_minus_arch"].median()),
                "fraction_train_gt_arch": float((loo_subset["train_eta_sq"] > loo_subset["arch_eta_sq"]).mean()),
                "worst_left_out_family": loo_subset.sort_values("train_minus_arch").iloc[0]["left_out_family"],
            },
        }

    return {
        "n_models": int(len(metadata)),
        "n_families": int(metadata["family"].nunique()),
        "full_cluster": full_cluster_row,
        "cluster_one_per_family": {
            "n_draws": int(len(resample_cluster)),
            "ari_training_exact_median": float(resample_cluster["ari_training_exact"].median()),
            "ari_training_exact_q05": float(resample_cluster["ari_training_exact"].quantile(0.05)),
            "ari_training_exact_q95": float(resample_cluster["ari_training_exact"].quantile(0.95)),
            "ari_architecture_median": float(resample_cluster["ari_architecture"].median()),
            "ari_architecture_q05": float(resample_cluster["ari_architecture"].quantile(0.05)),
            "ari_architecture_q95": float(resample_cluster["ari_architecture"].quantile(0.95)),
            "fraction_train_gt_arch": float(
                (resample_cluster["ari_training_exact"] > resample_cluster["ari_architecture"]).mean()
            ),
        },
        "cluster_leave_one_family_out": {
            "n_families": int(len(loo_cluster)),
            "min_ari_train_minus_arch": float(loo_cluster["ari_train_minus_arch"].min()),
            "median_ari_train_minus_arch": float(loo_cluster["ari_train_minus_arch"].median()),
            "fraction_train_gt_arch": float(
                (loo_cluster["ari_training_exact"] > loo_cluster["ari_architecture"]).mean()
            ),
            "worst_left_out_family": loo_cluster.sort_values("ari_train_minus_arch").iloc[0]["left_out_family"],
        },
        "metric_summary": metric_summary,
    }


def main() -> None:
    output_table_dir = RESULTS_DIR / "tables" / "analysis_07"
    ensure_dir(output_table_dir)

    metadata = pd.read_csv(PROCESSED_DIR / f"model_metadata_{SNAPSHOT_LABEL}.csv")
    errors = pd.read_parquet(PROCESSED_DIR / f"discovery_error_matrix_{SNAPSHOT_LABEL}.parquet")

    model_order = metadata["model_key"].tolist()
    error_matrix = errors[model_order].copy()
    filled = error_matrix.apply(lambda col: col.fillna(col.mean()), axis=0)
    model_vectors = zscore_rows(filled.to_numpy(dtype=float).T)
    full_distance_square = squareform(pdist(model_vectors, metric="euclidean"))

    full_anova_rows = [fit_exact_anova(metadata, metric) for metric in METRICS]
    full_cluster_row = cluster_alignment(metadata, model_order, full_distance_square)

    resample_anova, resample_cluster = one_per_family_resampling(
        metadata,
        model_order,
        full_distance_square,
    )
    loo_anova, loo_cluster = leave_one_family_out(
        metadata,
        model_order,
        full_distance_square,
    )

    resample_anova_summary = summarize_quantiles(
        resample_anova,
        ["train_eta_sq", "arch_eta_sq", "train_minus_arch", "model_r2"],
        ["metric", "metric_label"],
    )
    loo_anova_summary = summarize_quantiles(
        loo_anova,
        ["train_eta_sq", "arch_eta_sq", "train_minus_arch", "model_r2"],
        ["metric", "metric_label"],
    )

    full_anova_frame = pd.DataFrame(full_anova_rows)
    full_cluster_frame = pd.DataFrame([full_cluster_row])
    summary = build_summary(
        metadata,
        full_anova_rows,
        full_cluster_row,
        resample_anova,
        resample_cluster,
        loo_anova,
        loo_cluster,
    )

    full_anova_frame.to_csv(output_table_dir / "full_snapshot_anova.csv", index=False)
    full_cluster_frame.to_csv(output_table_dir / "full_snapshot_cluster.csv", index=False)
    resample_anova.to_csv(output_table_dir / "one_per_family_resampling_anova.csv", index=False)
    resample_cluster.to_csv(output_table_dir / "one_per_family_resampling_cluster.csv", index=False)
    resample_anova_summary.to_csv(output_table_dir / "one_per_family_resampling_anova_summary.csv", index=False)
    loo_anova.to_csv(output_table_dir / "leave_one_family_out_anova.csv", index=False)
    loo_cluster.to_csv(output_table_dir / "leave_one_family_out_cluster.csv", index=False)
    loo_anova_summary.to_csv(output_table_dir / "leave_one_family_out_anova_summary.csv", index=False)
    (output_table_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

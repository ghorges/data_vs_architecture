from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score
from statsmodels.stats.anova import anova_lm

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


METRICS = {
    "f1_full_test": "F1",
    "mae_full_test": "MAE",
    "daf_full_test": "DAF",
    "r2_full_test": "R2",
}
SCOPES = {
    "snapshot_45": {
        "metadata_path": PROCESSED_DIR / "model_metadata_snapshot_45.csv",
        "error_matrix_path": PROCESSED_DIR / "discovery_error_matrix_snapshot_45.parquet",
    },
    "live_53": {
        "metadata_path": PROCESSED_DIR / "model_metadata.csv",
        "error_matrix_path": PROCESSED_DIR / "discovery_error_matrix.parquet",
    },
}
DEFAULT_N_PERMUTATIONS = 2000
DEFAULT_SEED = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run permutation-based significance checks for the main data-vs-architecture claims.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help="Number of label-permutation draws to evaluate for each scope.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for the permutation generator.",
    )
    return parser.parse_args()


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def fit_exact_anova(metadata: pd.DataFrame, metric: str) -> dict:
    model = smf.ols(
        f"{metric} ~ C(training_combo) + C(architecture_group) + log_model_params",
        data=metadata,
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


def run_anova_permutations(
    metadata: pd.DataFrame,
    scope_name: str,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_rows = []
    draw_rows: list[dict] = []

    for metric in METRICS:
        observed_rows.append({"scope": scope_name, **fit_exact_anova(metadata, metric)})

    for draw in range(n_permutations):
        permuted = metadata.copy()
        permuted["training_combo"] = rng.permutation(permuted["training_combo"].to_numpy())
        permuted["architecture_group"] = rng.permutation(permuted["architecture_group"].to_numpy())
        for metric in METRICS:
            draw_rows.append({"scope": scope_name, "draw": draw, **fit_exact_anova(permuted, metric)})

    return pd.DataFrame(observed_rows), pd.DataFrame(draw_rows)


def prepare_cluster_alignment(metadata: pd.DataFrame, errors: pd.DataFrame) -> dict:
    model_order = metadata["model_key"].tolist()
    error_matrix = errors[model_order].copy()
    filled = error_matrix.apply(lambda col: col.fillna(col.mean()), axis=0)
    model_vectors = zscore_rows(filled.to_numpy(dtype=float).T)
    linkage_matrix = linkage(model_vectors, method="ward")
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
    observed_train = float(adjusted_rand_score(metadata["training_combo"], train_clusters))
    observed_arch = float(adjusted_rand_score(metadata["architecture_group"], arch_clusters))
    return {
        "train_clusters": train_clusters,
        "arch_clusters": arch_clusters,
        "observed": {
            "n_models": int(len(metadata)),
            "ari_training_exact": observed_train,
            "ari_architecture": observed_arch,
            "ari_train_minus_arch": float(observed_train - observed_arch),
        },
    }


def run_cluster_permutations(
    metadata: pd.DataFrame,
    errors: pd.DataFrame,
    scope_name: str,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_cluster_alignment(metadata, errors)
    observed = pd.DataFrame([{"scope": scope_name, **prepared["observed"]}])
    train_clusters = prepared["train_clusters"]
    arch_clusters = prepared["arch_clusters"]

    draw_rows: list[dict] = []
    training_labels = metadata["training_combo"].to_numpy()
    architecture_labels = metadata["architecture_group"].to_numpy()
    for draw in range(n_permutations):
        perm_train = rng.permutation(training_labels)
        perm_arch = rng.permutation(architecture_labels)
        ari_train = float(adjusted_rand_score(perm_train, train_clusters))
        ari_arch = float(adjusted_rand_score(perm_arch, arch_clusters))
        draw_rows.append(
            {
                "scope": scope_name,
                "draw": draw,
                "n_models": int(len(metadata)),
                "ari_training_exact": ari_train,
                "ari_architecture": ari_arch,
                "ari_train_minus_arch": float(ari_train - ari_arch),
            }
        )

    return observed, pd.DataFrame(draw_rows)


def summarize_permutation_test(
    observed: float,
    null_values: pd.Series,
) -> dict:
    null_values = null_values.astype(float)
    return {
        "observed": float(observed),
        "null_mean": float(null_values.mean()),
        "null_std": float(null_values.std(ddof=1)),
        "null_q95": float(null_values.quantile(0.95)),
        "null_q975": float(null_values.quantile(0.975)),
        "p_one_sided_ge": float(((null_values >= observed).sum() + 1) / (len(null_values) + 1)),
        "z_vs_null": float((observed - null_values.mean()) / null_values.std(ddof=1))
        if float(null_values.std(ddof=1)) > 0
        else float("nan"),
    }


def summarize_anova_permutations(observed: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for scope_name in observed["scope"].unique():
        scope_observed = observed.loc[observed["scope"] == scope_name]
        scope_draws = draws.loc[draws["scope"] == scope_name]
        for metric in METRICS:
            obs_row = scope_observed.loc[scope_observed["metric"] == metric].iloc[0]
            draw_subset = scope_draws.loc[scope_draws["metric"] == metric]
            row = {
                "scope": scope_name,
                "metric": metric,
                "metric_label": METRICS[metric],
                "n_models": int(obs_row["n_models"]),
            }
            for column in ["train_eta_sq", "arch_eta_sq", "train_minus_arch", "model_r2"]:
                stats = summarize_permutation_test(obs_row[column], draw_subset[column])
                row.update({f"{column}_{key}": value for key, value in stats.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_cluster_permutations(observed: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for scope_name in observed["scope"].unique():
        obs_row = observed.loc[observed["scope"] == scope_name].iloc[0]
        draw_subset = draws.loc[draws["scope"] == scope_name]
        row = {
            "scope": scope_name,
            "n_models": int(obs_row["n_models"]),
        }
        for column in ["ari_training_exact", "ari_architecture", "ari_train_minus_arch"]:
            stats = summarize_permutation_test(obs_row[column], draw_subset[column])
            row.update({f"{column}_{key}": value for key, value in stats.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = RESULTS_DIR / "tables" / "analysis_24"
    ensure_dir(output_dir)

    rng = np.random.default_rng(args.seed)

    anova_observed_tables: list[pd.DataFrame] = []
    anova_draw_tables: list[pd.DataFrame] = []
    cluster_observed_tables: list[pd.DataFrame] = []
    cluster_draw_tables: list[pd.DataFrame] = []

    for scope_name, scope_paths in SCOPES.items():
        metadata = pd.read_csv(scope_paths["metadata_path"]).copy()
        metadata["log_model_params"] = np.log10(metadata["model_params"].astype(float))
        anova_observed, anova_draws = run_anova_permutations(
            metadata=metadata,
            scope_name=scope_name,
            n_permutations=args.n_permutations,
            rng=rng,
        )
        anova_observed_tables.append(anova_observed)
        anova_draw_tables.append(anova_draws)

        errors = pd.read_parquet(scope_paths["error_matrix_path"])
        cluster_observed, cluster_draws = run_cluster_permutations(
            metadata=metadata,
            errors=errors,
            scope_name=scope_name,
            n_permutations=args.n_permutations,
            rng=rng,
        )
        cluster_observed_tables.append(cluster_observed)
        cluster_draw_tables.append(cluster_draws)

    anova_observed = pd.concat(anova_observed_tables, ignore_index=True)
    anova_draws = pd.concat(anova_draw_tables, ignore_index=True)
    anova_summary = summarize_anova_permutations(anova_observed, anova_draws)

    cluster_observed = pd.concat(cluster_observed_tables, ignore_index=True)
    cluster_draws = pd.concat(cluster_draw_tables, ignore_index=True)
    cluster_summary = summarize_cluster_permutations(cluster_observed, cluster_draws)

    summary = {
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
        "anova": {
            scope_name: {
                row["metric"]: {
                    "observed_train_minus_arch": float(row["train_minus_arch_observed"]),
                    "train_minus_arch_p_one_sided_ge": float(row["train_minus_arch_p_one_sided_ge"]),
                    "observed_train_eta_sq": float(row["train_eta_sq_observed"]),
                    "train_eta_sq_p_one_sided_ge": float(row["train_eta_sq_p_one_sided_ge"]),
                }
                for _, row in anova_summary.loc[anova_summary["scope"] == scope_name].iterrows()
            }
            for scope_name in SCOPES
        },
        "cluster_alignment": {
            scope_name: {
                "observed_ari_train_minus_arch": float(
                    cluster_summary.loc[cluster_summary["scope"] == scope_name, "ari_train_minus_arch_observed"].iloc[0]
                ),
                "ari_train_minus_arch_p_one_sided_ge": float(
                    cluster_summary.loc[
                        cluster_summary["scope"] == scope_name, "ari_train_minus_arch_p_one_sided_ge"
                    ].iloc[0]
                ),
                "observed_ari_training_exact": float(
                    cluster_summary.loc[cluster_summary["scope"] == scope_name, "ari_training_exact_observed"].iloc[0]
                ),
                "ari_training_exact_p_one_sided_ge": float(
                    cluster_summary.loc[
                        cluster_summary["scope"] == scope_name, "ari_training_exact_p_one_sided_ge"
                    ].iloc[0]
                ),
            }
            for scope_name in SCOPES
        },
    }

    anova_observed.to_csv(output_dir / "anova_observed.csv", index=False)
    anova_draws.to_csv(output_dir / "anova_permutation_draws.csv", index=False)
    anova_summary.to_csv(output_dir / "anova_permutation_summary.csv", index=False)
    cluster_observed.to_csv(output_dir / "cluster_observed.csv", index=False)
    cluster_draws.to_csv(output_dir / "cluster_permutation_draws.csv", index=False)
    cluster_summary.to_csv(output_dir / "cluster_permutation_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

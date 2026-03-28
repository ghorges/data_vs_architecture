from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


OUTCOME_COLUMNS = [
    "collective_failure",
    "collective_false_negative",
]
CV_N_SPLITS = 5
CV_N_REPEATS = 10
BOOTSTRAP_RESAMPLES = 2000
ROBUST_BUCKET_MIN_N = 20


def assign_rank_quartile(series: pd.Series) -> pd.Series:
    return pd.qcut(series.rank(method="first"), 4, labels=["Q1_rarest", "Q2", "Q3", "Q4_most_common"])


def build_bucket_rate_table(frame: pd.DataFrame, bucket_column: str, outcome_columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(bucket_column, as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
            collective_false_positive_rate=("collective_false_positive", "mean"),
            prototype_token_count_in_density_tier=("prototype_token_count_in_density_tier", "median"),
            spacegroup_count_in_density_tier=("spacegroup_count_in_density_tier", "median"),
        )
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )


def bootstrap_bucket_vs_other(
    frame: pd.DataFrame,
    bucket_column: str,
    bucket_label: str,
    outcome: str,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict:
    bucket_values = frame.loc[frame[bucket_column] == bucket_label, outcome].to_numpy(dtype=float)
    other_values = frame.loc[frame[bucket_column] != bucket_label, outcome].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    observed = float(bucket_values.mean() - other_values.mean())
    bootstrap = np.empty(n_resamples, dtype=float)
    for idx in range(n_resamples):
        sample_bucket = rng.choice(bucket_values, size=len(bucket_values), replace=True)
        sample_other = rng.choice(other_values, size=len(other_values), replace=True)
        bootstrap[idx] = sample_bucket.mean() - sample_other.mean()

    return {
        "bucket_column": bucket_column,
        "bucket_label": bucket_label,
        "outcome": outcome,
        "n_bucket": int(len(bucket_values)),
        "n_other": int(len(other_values)),
        "observed_rate_diff": observed,
        "q025": float(np.quantile(bootstrap, 0.025)),
        "q975": float(np.quantile(bootstrap, 0.975)),
        "fraction_diff_gt_zero": float((bootstrap > 0).mean()),
    }


def evaluate_feature_set(
    frame: pd.DataFrame,
    feature_columns: list[str],
    outcome: str,
    feature_set_name: str,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    X = frame[feature_columns].copy()
    y = frame[outcome].astype(int).to_numpy()

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), feature_columns)],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )

    cv = RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS,
        n_repeats=CV_N_REPEATS,
        random_state=seed,
    )

    fold_rows: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline.fit(X_train, y_train)
        prob = pipeline.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        fold_rows.append(
            {
                "feature_set": feature_set_name,
                "outcome": outcome,
                "fold": fold_idx,
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "roc_auc": float(roc_auc_score(y_test, prob)),
                "average_precision": float(average_precision_score(y_test, prob)),
                "log_loss": float(log_loss(y_test, prob, labels=[0, 1])),
            }
        )

    fold_table = pd.DataFrame(fold_rows)
    summary = {
        "feature_set": feature_set_name,
        "outcome": outcome,
        "n_folds": int(len(fold_table)),
        "balanced_accuracy_mean": float(fold_table["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(fold_table["balanced_accuracy"].std(ddof=1)),
        "roc_auc_mean": float(fold_table["roc_auc"].mean()),
        "roc_auc_std": float(fold_table["roc_auc"].std(ddof=1)),
        "average_precision_mean": float(fold_table["average_precision"].mean()),
        "average_precision_std": float(fold_table["average_precision"].std(ddof=1)),
        "log_loss_mean": float(fold_table["log_loss"].mean()),
        "log_loss_std": float(fold_table["log_loss"].std(ddof=1)),
    }
    return fold_table, summary


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_13"
    ensure_dir(output_dir)

    features = pd.read_csv(
        RESULTS_DIR / "tables" / "analysis_04" / "collective_failure_features.csv",
        usecols=[
            "material_id",
            "collective_failure",
            "collective_success",
            "collective_false_negative",
            "collective_false_positive",
            "crystal_system",
        ],
    )
    rarity = pd.read_parquet(
        PROCESSED_DIR / "wbm_mptrj_mp2022_union_material_prototype_spacegroup_rarity_proxy.parquet",
        columns=[
            "material_id",
            "prototype_token_count_in_density_tier",
            "spacegroup_count_in_density_tier",
        ],
    )
    hotspot = pd.read_parquet(
        PROCESSED_DIR / "wbm_mptrj_mp2022_union_material_motif_hotspot_proxy.parquet",
        columns=[
            "material_id",
            "singleton_high_risk_candidate",
            "frequent_prototype_token_bucket",
            "frequent_spacegroup_bucket",
        ],
    )

    frame = (
        features.merge(rarity, on="material_id", how="left", validate="one_to_one")
        .merge(hotspot, on="material_id", how="left", validate="one_to_one")
    )
    frame = frame.loc[frame["collective_failure"] | frame["collective_success"]].copy()
    frame = frame.loc[frame["singleton_high_risk_candidate"]].copy()
    frame["prototype_density_rarity_quartile"] = assign_rank_quartile(frame["prototype_token_count_in_density_tier"])
    frame["spacegroup_density_rarity_quartile"] = assign_rank_quartile(frame["spacegroup_count_in_density_tier"])
    frame["frequent_prototype_token_bucket"] = frame["frequent_prototype_token_bucket"].fillna("other")
    frame["frequent_spacegroup_bucket"] = frame["frequent_spacegroup_bucket"].fillna("other")
    frame["crystal_system"] = frame["crystal_system"].fillna("unknown")

    prototype_bucket_rates = build_bucket_rate_table(
        frame,
        "frequent_prototype_token_bucket",
        OUTCOME_COLUMNS,
    )
    spacegroup_bucket_rates = build_bucket_rate_table(
        frame,
        "frequent_spacegroup_bucket",
        OUTCOME_COLUMNS,
    )
    robust_prototype_bucket_rates = prototype_bucket_rates.loc[
        (prototype_bucket_rates["n_materials"] >= ROBUST_BUCKET_MIN_N)
        & (prototype_bucket_rates["frequent_prototype_token_bucket"] != "other")
    ].copy()
    robust_spacegroup_bucket_rates = spacegroup_bucket_rates.loc[
        (spacegroup_bucket_rates["n_materials"] >= ROBUST_BUCKET_MIN_N)
        & (spacegroup_bucket_rates["frequent_spacegroup_bucket"] != "other")
    ].copy()

    bootstrap_rows: list[dict] = []
    for bucket_label in robust_prototype_bucket_rates["frequent_prototype_token_bucket"]:
        for outcome in OUTCOME_COLUMNS:
            bootstrap_rows.append(
                bootstrap_bucket_vs_other(
                    frame=frame,
                    bucket_column="frequent_prototype_token_bucket",
                    bucket_label=bucket_label,
                    outcome=outcome,
                    seed=101 + len(bootstrap_rows),
                )
            )
    for bucket_label in robust_spacegroup_bucket_rates["frequent_spacegroup_bucket"]:
        for outcome in OUTCOME_COLUMNS:
            bootstrap_rows.append(
                bootstrap_bucket_vs_other(
                    frame=frame,
                    bucket_column="frequent_spacegroup_bucket",
                    bucket_label=bucket_label,
                    outcome=outcome,
                    seed=101 + len(bootstrap_rows),
                )
            )
    bootstrap_summary = pd.DataFrame(bootstrap_rows)

    feature_sets = {
        "crystal_system_only": ["crystal_system"],
        "rarity_bins": [
            "crystal_system",
            "prototype_density_rarity_quartile",
            "spacegroup_density_rarity_quartile",
        ],
        "spacegroup_bucket": ["crystal_system", "frequent_spacegroup_bucket"],
        "prototype_bucket": ["crystal_system", "frequent_prototype_token_bucket"],
        "combined_hotspot_buckets": [
            "crystal_system",
            "frequent_spacegroup_bucket",
            "frequent_prototype_token_bucket",
        ],
    }
    cv_fold_tables: list[pd.DataFrame] = []
    cv_summary_rows: list[dict] = []
    for outcome in OUTCOME_COLUMNS:
        for feature_set_name, feature_columns in feature_sets.items():
            fold_table, summary = evaluate_feature_set(
                frame=frame,
                feature_columns=feature_columns,
                outcome=outcome,
                feature_set_name=feature_set_name,
                seed=17,
            )
            cv_fold_tables.append(fold_table)
            cv_summary_rows.append(summary)
    cv_folds = pd.concat(cv_fold_tables, ignore_index=True)
    cv_summary = pd.DataFrame(cv_summary_rows)

    best_failure = cv_summary.loc[cv_summary["outcome"] == "collective_failure"].sort_values(
        ["balanced_accuracy_mean", "average_precision_mean"],
        ascending=[False, False],
    )
    best_false_negative = cv_summary.loc[cv_summary["outcome"] == "collective_false_negative"].sort_values(
        ["balanced_accuracy_mean", "average_precision_mean"],
        ascending=[False, False],
    )

    summary = {
        "singleton_high_risk_subset": {
            "n_materials": int(len(frame)),
            "best_failure_feature_set": best_failure.iloc[0][
                ["feature_set", "balanced_accuracy_mean", "average_precision_mean", "log_loss_mean"]
            ].to_dict(),
            "best_false_negative_feature_set": best_false_negative.iloc[0][
                ["feature_set", "balanced_accuracy_mean", "average_precision_mean", "log_loss_mean"]
            ].to_dict(),
            "top_robust_prototype_bucket": (
                robust_prototype_bucket_rates.iloc[0][
                    [
                        "frequent_prototype_token_bucket",
                        "n_materials",
                        "collective_failure_rate",
                        "collective_false_negative_rate",
                    ]
                ].to_dict()
                if not robust_prototype_bucket_rates.empty
                else {}
            ),
            "top_robust_spacegroup_bucket": (
                robust_spacegroup_bucket_rates.iloc[0][
                    [
                        "frequent_spacegroup_bucket",
                        "n_materials",
                        "collective_failure_rate",
                        "collective_false_negative_rate",
                    ]
                ].to_dict()
                if not robust_spacegroup_bucket_rates.empty
                else {}
            ),
        }
    }

    prototype_bucket_rates.to_csv(output_dir / "prototype_bucket_rates.csv", index=False)
    spacegroup_bucket_rates.to_csv(output_dir / "spacegroup_bucket_rates.csv", index=False)
    robust_prototype_bucket_rates.to_csv(output_dir / "robust_prototype_bucket_rates.csv", index=False)
    robust_spacegroup_bucket_rates.to_csv(output_dir / "robust_spacegroup_bucket_rates.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "hotspot_bucket_bootstrap_differences.csv", index=False)
    cv_folds.to_csv(output_dir / "feature_set_cv_fold_metrics.csv", index=False)
    cv_summary.to_csv(output_dir / "feature_set_cv_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

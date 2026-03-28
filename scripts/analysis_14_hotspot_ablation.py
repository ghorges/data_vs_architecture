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


OUTCOMES = [
    "collective_failure",
    "collective_false_negative",
]
CV_N_SPLITS = 5
CV_N_REPEATS = 10
ROBUST_BUCKET_MIN_N = 20
PRIMARY_FEATURE_SETS = {
    "rarity_bins": [
        "crystal_system",
        "prototype_density_rarity_quartile",
        "spacegroup_density_rarity_quartile",
    ],
    "combined_hotspot_buckets": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_prototype_token_bucket",
    ],
}


def assign_rank_quartile(series: pd.Series) -> pd.Series:
    return pd.qcut(series.rank(method="first"), 4, labels=["Q1_rarest", "Q2", "Q3", "Q4_most_common"])


def build_bucket_rate_table(frame: pd.DataFrame, bucket_column: str) -> pd.DataFrame:
    return (
        frame.groupby(bucket_column, as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
            collective_false_positive_rate=("collective_false_positive", "mean"),
        )
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )


def load_analysis_frame() -> pd.DataFrame:
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
    return frame


def evaluate_feature_set(
    frame: pd.DataFrame,
    feature_columns: list[str],
    outcome: str,
    feature_set_name: str,
    scenario_name: str,
    seed: int = 17,
) -> tuple[pd.DataFrame, dict]:
    X = frame[feature_columns].copy()
    y = frame[outcome].astype(int).to_numpy()
    class_counts = np.bincount(y, minlength=2)
    n_splits = int(min(CV_N_SPLITS, class_counts[0], class_counts[1]))
    if n_splits < 2:
        raise ValueError(f"not enough class support for {scenario_name=} and {outcome=}")

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
        n_splits=n_splits,
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
                "scenario": scenario_name,
                "feature_set": feature_set_name,
                "outcome": outcome,
                "fold": fold_idx,
                "n_materials": int(len(frame)),
                "positive_rate": float(y.mean()),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "roc_auc": float(roc_auc_score(y_test, prob)),
                "average_precision": float(average_precision_score(y_test, prob)),
                "log_loss": float(log_loss(y_test, prob, labels=[0, 1])),
            }
        )

    fold_table = pd.DataFrame(fold_rows)
    summary = {
        "scenario": scenario_name,
        "feature_set": feature_set_name,
        "outcome": outcome,
        "n_materials": int(len(frame)),
        "positive_rate": float(y.mean()),
        "n_folds": int(len(fold_table)),
        "n_splits": n_splits,
        "n_repeats": CV_N_REPEATS,
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


def run_feature_set_suite(
    frame: pd.DataFrame,
    scenario_name: str,
    feature_sets: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = feature_sets or PRIMARY_FEATURE_SETS
    fold_tables: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for outcome in OUTCOMES:
        for feature_set_name, feature_columns in feature_sets.items():
            fold_table, summary = evaluate_feature_set(
                frame=frame,
                feature_columns=feature_columns,
                outcome=outcome,
                feature_set_name=feature_set_name,
                scenario_name=scenario_name,
            )
            fold_tables.append(fold_table)
            summary_rows.append(summary)
    return pd.concat(fold_tables, ignore_index=True), pd.DataFrame(summary_rows)


def mask_bucket(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    masked = frame.copy()
    masked.loc[masked[column] == label, column] = "other"
    return masked


def drop_bucket_rows(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    return frame.loc[frame[column] != label].copy()


def compute_advantage_table(summary_table: pd.DataFrame, scenario_group: str) -> pd.DataFrame:
    value_columns = [
        "n_materials",
        "positive_rate",
        "balanced_accuracy_mean",
        "average_precision_mean",
        "roc_auc_mean",
        "log_loss_mean",
    ]
    pivot = (
        summary_table.set_index(["scenario", "outcome", "feature_set"])[value_columns]
        .unstack("feature_set")
        .sort_index()
    )
    advantage = pd.DataFrame(index=pivot.index).reset_index()
    advantage["scenario_group"] = scenario_group
    advantage["n_materials"] = pivot[("n_materials", "combined_hotspot_buckets")].to_numpy()
    advantage["positive_rate"] = pivot[("positive_rate", "combined_hotspot_buckets")].to_numpy()
    advantage["balanced_accuracy_delta"] = (
        pivot[("balanced_accuracy_mean", "combined_hotspot_buckets")]
        - pivot[("balanced_accuracy_mean", "rarity_bins")]
    ).to_numpy()
    advantage["average_precision_delta"] = (
        pivot[("average_precision_mean", "combined_hotspot_buckets")]
        - pivot[("average_precision_mean", "rarity_bins")]
    ).to_numpy()
    advantage["roc_auc_delta"] = (
        pivot[("roc_auc_mean", "combined_hotspot_buckets")] - pivot[("roc_auc_mean", "rarity_bins")]
    ).to_numpy()
    advantage["log_loss_improvement"] = (
        pivot[("log_loss_mean", "rarity_bins")] - pivot[("log_loss_mean", "combined_hotspot_buckets")]
    ).to_numpy()
    return advantage


def summarize_leave_one_out_advantage(advantage: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for ablation_kind, chunk in advantage.groupby("ablation_kind"):
        for outcome, outcome_chunk in chunk.groupby("outcome"):
            sorted_chunk = outcome_chunk.sort_values("balanced_accuracy_delta")
            worst_row = sorted_chunk.iloc[0]
            rows.append(
                {
                    "ablation_kind": ablation_kind,
                    "outcome": outcome,
                    "n_scenarios": int(len(outcome_chunk)),
                    "fraction_delta_gt_zero": float((outcome_chunk["balanced_accuracy_delta"] > 0).mean()),
                    "min_balanced_accuracy_delta": float(outcome_chunk["balanced_accuracy_delta"].min()),
                    "median_balanced_accuracy_delta": float(outcome_chunk["balanced_accuracy_delta"].median()),
                    "min_average_precision_delta": float(outcome_chunk["average_precision_delta"].min()),
                    "median_average_precision_delta": float(outcome_chunk["average_precision_delta"].median()),
                    "min_log_loss_improvement": float(outcome_chunk["log_loss_improvement"].min()),
                    "median_log_loss_improvement": float(outcome_chunk["log_loss_improvement"].median()),
                    "worst_case_bucket": str(worst_row["dropped_bucket"]),
                    "worst_case_dropped_bucket_n_materials": int(worst_row["dropped_bucket_n_materials"]),
                    "worst_case_remaining_n_materials": int(worst_row["n_materials"]),
                    "worst_case_balanced_accuracy_delta": float(worst_row["balanced_accuracy_delta"]),
                    "worst_case_average_precision_delta": float(worst_row["average_precision_delta"]),
                }
            )
    return rows


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_14"
    ensure_dir(output_dir)

    frame = load_analysis_frame()
    prototype_bucket_rates = build_bucket_rate_table(frame, "frequent_prototype_token_bucket")
    spacegroup_bucket_rates = build_bucket_rate_table(frame, "frequent_spacegroup_bucket")
    robust_prototypes = prototype_bucket_rates.loc[
        (prototype_bucket_rates["n_materials"] >= ROBUST_BUCKET_MIN_N)
        & (prototype_bucket_rates["frequent_prototype_token_bucket"] != "other")
    ].copy()
    robust_spacegroups = spacegroup_bucket_rates.loc[
        (spacegroup_bucket_rates["n_materials"] >= ROBUST_BUCKET_MIN_N)
        & (spacegroup_bucket_rates["frequent_spacegroup_bucket"] != "other")
    ].copy()

    top_prototype = str(robust_prototypes.iloc[0]["frequent_prototype_token_bucket"])
    top_spacegroup = str(robust_spacegroups.iloc[0]["frequent_spacegroup_bucket"])

    targeted_scenarios = {
        "baseline": frame,
        "mask_top_prototype_bucket": mask_bucket(frame, "frequent_prototype_token_bucket", top_prototype),
        "mask_top_spacegroup_bucket": mask_bucket(frame, "frequent_spacegroup_bucket", top_spacegroup),
        "mask_both_top_buckets": mask_bucket(
            mask_bucket(frame, "frequent_prototype_token_bucket", top_prototype),
            "frequent_spacegroup_bucket",
            top_spacegroup,
        ),
        "drop_top_prototype_bucket_materials": drop_bucket_rows(frame, "frequent_prototype_token_bucket", top_prototype),
        "drop_top_spacegroup_bucket_materials": drop_bucket_rows(frame, "frequent_spacegroup_bucket", top_spacegroup),
        "drop_either_top_bucket_materials": frame.loc[
            (frame["frequent_prototype_token_bucket"] != top_prototype)
            & (frame["frequent_spacegroup_bucket"] != top_spacegroup)
        ].copy(),
    }

    targeted_fold_tables: list[pd.DataFrame] = []
    targeted_summary_tables: list[pd.DataFrame] = []
    for scenario_name, scenario_frame in targeted_scenarios.items():
        folds, summary = run_feature_set_suite(scenario_frame, scenario_name)
        targeted_fold_tables.append(folds)
        targeted_summary_tables.append(summary)
    targeted_folds = pd.concat(targeted_fold_tables, ignore_index=True)
    targeted_summary = pd.concat(targeted_summary_tables, ignore_index=True)
    targeted_advantage = compute_advantage_table(targeted_summary, scenario_group="targeted")

    loo_fold_tables: list[pd.DataFrame] = []
    loo_summary_tables: list[pd.DataFrame] = []
    loo_meta_rows: list[dict] = []
    for bucket in robust_prototypes["frequent_prototype_token_bucket"]:
        scenario_name = f"drop_prototype__{bucket}"
        scenario_frame = drop_bucket_rows(frame, "frequent_prototype_token_bucket", bucket)
        folds, summary = run_feature_set_suite(scenario_frame, scenario_name)
        loo_fold_tables.append(folds)
        loo_summary_tables.append(summary)
        loo_meta_rows.append(
            {
                "scenario": scenario_name,
                "ablation_kind": "prototype",
                "dropped_bucket": bucket,
                "dropped_bucket_n_materials": int((frame["frequent_prototype_token_bucket"] == bucket).sum()),
                "dropped_fraction": float((frame["frequent_prototype_token_bucket"] == bucket).mean()),
            }
        )
    for bucket in robust_spacegroups["frequent_spacegroup_bucket"]:
        scenario_name = f"drop_spacegroup__{bucket}"
        scenario_frame = drop_bucket_rows(frame, "frequent_spacegroup_bucket", bucket)
        folds, summary = run_feature_set_suite(scenario_frame, scenario_name)
        loo_fold_tables.append(folds)
        loo_summary_tables.append(summary)
        loo_meta_rows.append(
            {
                "scenario": scenario_name,
                "ablation_kind": "spacegroup",
                "dropped_bucket": bucket,
                "dropped_bucket_n_materials": int((frame["frequent_spacegroup_bucket"] == bucket).sum()),
                "dropped_fraction": float((frame["frequent_spacegroup_bucket"] == bucket).mean()),
            }
        )

    loo_folds = pd.concat(loo_fold_tables, ignore_index=True)
    loo_summary = pd.concat(loo_summary_tables, ignore_index=True)
    loo_meta = pd.DataFrame(loo_meta_rows)
    loo_advantage = compute_advantage_table(loo_summary, scenario_group="leave_one_out").merge(
        loo_meta,
        on="scenario",
        how="left",
        validate="many_to_one",
    )
    loo_advantage_summary = pd.DataFrame(summarize_leave_one_out_advantage(loo_advantage))

    targeted_lookup = targeted_advantage.set_index(["scenario", "outcome"])
    summary = {
        "singleton_high_risk_subset": {
            "n_materials": int(len(frame)),
            "top_prototype_bucket": top_prototype,
            "top_spacegroup_bucket": top_spacegroup,
        },
        "targeted_ablation": {
            "collective_failure": {
                "baseline_balanced_accuracy_delta": float(
                    targeted_lookup.loc[("baseline", "collective_failure"), "balanced_accuracy_delta"]
                ),
                "mask_both_top_buckets_balanced_accuracy_delta": float(
                    targeted_lookup.loc[("mask_both_top_buckets", "collective_failure"), "balanced_accuracy_delta"]
                ),
                "drop_either_top_bucket_materials_balanced_accuracy_delta": float(
                    targeted_lookup.loc[
                        ("drop_either_top_bucket_materials", "collective_failure"),
                        "balanced_accuracy_delta",
                    ]
                ),
            },
            "collective_false_negative": {
                "baseline_balanced_accuracy_delta": float(
                    targeted_lookup.loc[("baseline", "collective_false_negative"), "balanced_accuracy_delta"]
                ),
                "mask_both_top_buckets_balanced_accuracy_delta": float(
                    targeted_lookup.loc[("mask_both_top_buckets", "collective_false_negative"), "balanced_accuracy_delta"]
                ),
                "drop_either_top_bucket_materials_balanced_accuracy_delta": float(
                    targeted_lookup.loc[
                        ("drop_either_top_bucket_materials", "collective_false_negative"),
                        "balanced_accuracy_delta",
                    ]
                ),
            },
        },
        "leave_one_hotspot_out": loo_advantage_summary.to_dict(orient="records"),
    }

    prototype_bucket_rates.to_csv(output_dir / "prototype_bucket_rates.csv", index=False)
    spacegroup_bucket_rates.to_csv(output_dir / "spacegroup_bucket_rates.csv", index=False)
    robust_prototypes.to_csv(output_dir / "robust_prototype_bucket_rates.csv", index=False)
    robust_spacegroups.to_csv(output_dir / "robust_spacegroup_bucket_rates.csv", index=False)
    targeted_folds.to_csv(output_dir / "targeted_ablation_cv_fold_metrics.csv", index=False)
    targeted_summary.to_csv(output_dir / "targeted_ablation_cv_summary.csv", index=False)
    targeted_advantage.to_csv(output_dir / "targeted_ablation_advantage.csv", index=False)
    loo_folds.to_csv(output_dir / "leave_one_hotspot_out_cv_fold_metrics.csv", index=False)
    loo_summary.to_csv(output_dir / "leave_one_hotspot_out_cv_summary.csv", index=False)
    loo_advantage.to_csv(output_dir / "leave_one_hotspot_out_advantage.csv", index=False)
    loo_advantage_summary.to_csv(output_dir / "leave_one_hotspot_out_advantage_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

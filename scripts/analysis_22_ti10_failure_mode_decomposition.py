from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis_14_hotspot_ablation import evaluate_feature_set
from analysis_15_prototype_component_decomposition import load_analysis_frame
from analysis_16_component_ablation import assign_component_branch
from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


OUTCOMES = [
    "collective_failure",
    "collective_false_negative",
    "collective_false_positive",
]
FEATURE_SETS = {
    "spacegroup_only": ["frequent_spacegroup_bucket"],
    "formula_plus_spacegroup": ["frequent_formula_family_bucket", "frequent_spacegroup_bucket"],
    "spacegroup_plus_wyckoff": ["frequent_spacegroup_bucket", "frequent_wyckoff_signature_bucket"],
    "formula_plus_spacegroup_plus_wyckoff": [
        "frequent_formula_family_bucket",
        "frequent_spacegroup_bucket",
        "frequent_wyckoff_signature_bucket",
    ],
    "frequent_ti10_spacegroup_wyckoff_bucket": ["frequent_ti10_spacegroup_wyckoff_bucket"],
    "full_token": ["frequent_prototype_token_bucket"],
}
BOOTSTRAP_RESAMPLES = 5000


def load_ti10_frame() -> pd.DataFrame:
    frame = assign_component_branch(load_analysis_frame())
    subset = frame.loc[frame["component_branch"] == "tI10_branch"].copy()

    proxy = pd.read_parquet(
        PROCESSED_DIR / "singleton_high_risk_ti10_branch_mode_proxy.parquet",
        columns=[
            "material_id",
            "ti10_spacegroup_wyckoff_bucket",
            "ti10_formula_spacegroup_bucket",
            "ti10_formula_spacegroup_wyckoff_bucket",
            "frequent_ti10_spacegroup_wyckoff_bucket",
            "frequent_ti10_formula_spacegroup_wyckoff_bucket",
        ],
    )
    subset = subset.merge(proxy, on="material_id", how="left", validate="one_to_one")
    return subset


def build_rate_table(frame: pd.DataFrame, bucket_column: str) -> pd.DataFrame:
    return (
        frame.groupby(bucket_column, as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
            collective_false_positive_rate=("collective_false_positive", "mean"),
        )
        .sort_values(
            ["collective_false_negative_rate", "collective_false_positive_rate", "n_materials"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def run_feature_set_suite(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_tables: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for outcome in OUTCOMES:
        for feature_set_name, feature_columns in FEATURE_SETS.items():
            fold_table, summary = evaluate_feature_set(
                frame=frame,
                feature_columns=feature_columns,
                outcome=outcome,
                feature_set_name=feature_set_name,
                scenario_name="tI10_branch",
            )
            fold_tables.append(fold_table)
            summary_rows.append(summary)
    return pd.concat(fold_tables, ignore_index=True), pd.DataFrame(summary_rows)


def compute_delta_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "n_materials",
        "positive_rate",
        "balanced_accuracy_mean",
        "average_precision_mean",
        "roc_auc_mean",
        "log_loss_mean",
    ]
    pivot = summary.set_index(["outcome", "feature_set"])[metric_columns].unstack("feature_set")
    rows: list[dict] = []
    for outcome in sorted(summary["outcome"].unique()):
        row: dict[str, float | str] = {"outcome": outcome}
        for feature_set in FEATURE_SETS:
            row[f"{feature_set}_balanced_accuracy"] = float(pivot.loc[outcome, ("balanced_accuracy_mean", feature_set)])
            row[f"{feature_set}_average_precision"] = float(pivot.loc[outcome, ("average_precision_mean", feature_set)])
            row[f"{feature_set}_log_loss"] = float(pivot.loc[outcome, ("log_loss_mean", feature_set)])

        row["frequent_bucket_vs_spacegroup_plus_wyckoff_balanced_accuracy_delta"] = (
            row["frequent_ti10_spacegroup_wyckoff_bucket_balanced_accuracy"]
            - row["spacegroup_plus_wyckoff_balanced_accuracy"]
        )
        row["frequent_bucket_vs_full_token_balanced_accuracy_delta"] = (
            row["frequent_ti10_spacegroup_wyckoff_bucket_balanced_accuracy"] - row["full_token_balanced_accuracy"]
        )
        row["frequent_bucket_vs_spacegroup_plus_wyckoff_average_precision_delta"] = (
            row["frequent_ti10_spacegroup_wyckoff_bucket_average_precision"]
            - row["spacegroup_plus_wyckoff_average_precision"]
        )
        row["frequent_bucket_vs_full_token_average_precision_delta"] = (
            row["frequent_ti10_spacegroup_wyckoff_bucket_average_precision"] - row["full_token_average_precision"]
        )
        row["frequent_bucket_vs_spacegroup_plus_wyckoff_log_loss_improvement"] = (
            row["spacegroup_plus_wyckoff_log_loss"] - row["frequent_ti10_spacegroup_wyckoff_bucket_log_loss"]
        )
        row["frequent_bucket_vs_full_token_log_loss_improvement"] = (
            row["full_token_log_loss"] - row["frequent_ti10_spacegroup_wyckoff_bucket_log_loss"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_bucket_difference(
    frame: pd.DataFrame,
    bucket_column: str,
    bucket_a: str,
    bucket_b: str,
    outcome: str,
    seed: int,
) -> dict:
    a_values = frame.loc[frame[bucket_column] == bucket_a, outcome].to_numpy(dtype=float)
    b_values = frame.loc[frame[bucket_column] == bucket_b, outcome].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    observed = float(a_values.mean() - b_values.mean())
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for idx in range(BOOTSTRAP_RESAMPLES):
        draws[idx] = (
            rng.choice(a_values, size=len(a_values), replace=True).mean()
            - rng.choice(b_values, size=len(b_values), replace=True).mean()
        )
    return {
        "bucket_column": bucket_column,
        "bucket_a": bucket_a,
        "bucket_b": bucket_b,
        "outcome": outcome,
        "n_bucket_a": int(len(a_values)),
        "n_bucket_b": int(len(b_values)),
        "observed_rate_diff": observed,
        "q025": float(np.quantile(draws, 0.025)),
        "q975": float(np.quantile(draws, 0.975)),
        "fraction_diff_gt_zero": float((draws > 0).mean()),
    }


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_22"
    ensure_dir(output_dir)

    frame = load_ti10_frame()
    cv_folds, cv_summary = run_feature_set_suite(frame)
    delta_summary = compute_delta_summary(cv_summary)

    spacegroup_rates = build_rate_table(frame, "frequent_spacegroup_bucket")
    formula_rates = build_rate_table(frame, "frequent_formula_family_bucket")
    wyckoff_rates = build_rate_table(frame, "frequent_wyckoff_signature_bucket")
    pair_rates = build_rate_table(frame, "ti10_spacegroup_wyckoff_bucket")
    frequent_pair_rates = build_rate_table(frame, "frequent_ti10_spacegroup_wyckoff_bucket")
    token_rates = build_rate_table(frame, "frequent_prototype_token_bucket")

    contrast_pairs = [
        ("sg_87__other", "sg_139__d_a_e"),
        ("sg_107__a_ab_a", "sg_107__a_a_ab"),
        ("sg_139__a_e_d", "sg_139__d_a_e"),
        ("sg_139__a_d_e", "sg_139__d_a_e"),
    ]
    bootstrap_rows: list[dict] = []
    for pair_idx, (bucket_a, bucket_b) in enumerate(contrast_pairs):
        for outcome_idx, outcome in enumerate(OUTCOMES):
            bootstrap_rows.append(
                bootstrap_bucket_difference(
                    frame=frame,
                    bucket_column="ti10_spacegroup_wyckoff_bucket",
                    bucket_a=bucket_a,
                    bucket_b=bucket_b,
                    outcome=outcome,
                    seed=1000 + pair_idx * 100 + outcome_idx,
                )
            )
    bootstrap_table = pd.DataFrame(bootstrap_rows)

    cv_lookup = cv_summary.set_index(["outcome", "feature_set"])
    pair_lookup = pair_rates.set_index("ti10_spacegroup_wyckoff_bucket")
    summary = {
        "tI10_branch": {
            "n_materials": int(len(frame)),
            "feature_sets": {
                outcome: {
                    feature_set: {
                        "balanced_accuracy_mean": float(
                            cv_lookup.loc[(outcome, feature_set), "balanced_accuracy_mean"]
                        ),
                        "average_precision_mean": float(
                            cv_lookup.loc[(outcome, feature_set), "average_precision_mean"]
                        ),
                        "log_loss_mean": float(cv_lookup.loc[(outcome, feature_set), "log_loss_mean"]),
                    }
                    for feature_set in FEATURE_SETS
                }
                for outcome in OUTCOMES
            },
            "key_structural_modes": {
                bucket: {
                    "n_materials": int(pair_lookup.loc[bucket, "n_materials"]),
                    "collective_failure_rate": float(pair_lookup.loc[bucket, "collective_failure_rate"]),
                    "collective_false_negative_rate": float(pair_lookup.loc[bucket, "collective_false_negative_rate"]),
                    "collective_false_positive_rate": float(pair_lookup.loc[bucket, "collective_false_positive_rate"]),
                }
                for bucket in [
                    "sg_87__other",
                    "sg_107__a_ab_a",
                    "sg_107__a_a_ab",
                    "sg_139__a_e_d",
                    "sg_139__d_a_e",
                ]
            },
            "key_deltas": {
                outcome: {
                    "frequent_bucket_vs_spacegroup_plus_wyckoff_balanced_accuracy_delta": float(
                        delta_summary.set_index("outcome").loc[
                            outcome, "frequent_bucket_vs_spacegroup_plus_wyckoff_balanced_accuracy_delta"
                        ]
                    ),
                    "frequent_bucket_vs_full_token_balanced_accuracy_delta": float(
                        delta_summary.set_index("outcome").loc[
                            outcome, "frequent_bucket_vs_full_token_balanced_accuracy_delta"
                        ]
                    ),
                    "frequent_bucket_vs_spacegroup_plus_wyckoff_average_precision_delta": float(
                        delta_summary.set_index("outcome").loc[
                            outcome, "frequent_bucket_vs_spacegroup_plus_wyckoff_average_precision_delta"
                        ]
                    ),
                    "frequent_bucket_vs_full_token_average_precision_delta": float(
                        delta_summary.set_index("outcome").loc[
                            outcome, "frequent_bucket_vs_full_token_average_precision_delta"
                        ]
                    ),
                }
                for outcome in OUTCOMES
            },
        }
    }

    cv_folds.to_csv(output_dir / "feature_set_cv_fold_metrics.csv", index=False)
    cv_summary.to_csv(output_dir / "feature_set_cv_summary.csv", index=False)
    delta_summary.to_csv(output_dir / "feature_set_delta_summary.csv", index=False)
    spacegroup_rates.to_csv(output_dir / "spacegroup_rates.csv", index=False)
    formula_rates.to_csv(output_dir / "formula_family_rates.csv", index=False)
    wyckoff_rates.to_csv(output_dir / "wyckoff_signature_rates.csv", index=False)
    pair_rates.to_csv(output_dir / "spacegroup_wyckoff_rates.csv", index=False)
    frequent_pair_rates.to_csv(output_dir / "frequent_spacegroup_wyckoff_rates.csv", index=False)
    token_rates.to_csv(output_dir / "prototype_token_rates.csv", index=False)
    bootstrap_table.to_csv(output_dir / "spacegroup_wyckoff_bootstrap_differences.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

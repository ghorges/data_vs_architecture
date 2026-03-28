from __future__ import annotations

import json

import pandas as pd

from analysis_14_hotspot_ablation import OUTCOMES, run_feature_set_suite
from analysis_15_prototype_component_decomposition import load_analysis_frame
from analysis_16_component_ablation import assign_component_branch
from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


INTERACTION_COLUMNS = [
    "frequent_formula_x_pearson_bucket",
    "frequent_formula_x_spacegroup_bucket",
    "frequent_pearson_x_spacegroup_bucket",
    "frequent_pearson_x_wyckoff_bucket",
    "frequent_formula_x_wyckoff_bucket",
    "frequent_spacegroup_x_wyckoff_bucket",
]
FEATURE_SETS = {
    "components_full": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
    ],
    "components_plus_archetype": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        "frequent_archetype_bucket",
    ],
    "components_plus_formula_x_pearson": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        "frequent_formula_x_pearson_bucket",
    ],
    "components_plus_pearson_x_spacegroup": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        "frequent_pearson_x_spacegroup_bucket",
    ],
    "components_plus_formula_x_wyckoff": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        "frequent_formula_x_wyckoff_bucket",
    ],
    "components_plus_spacegroup_x_wyckoff": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        "frequent_spacegroup_x_wyckoff_bucket",
    ],
    "components_plus_interaction_bundle": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        "frequent_formula_x_pearson_bucket",
        "frequent_formula_x_spacegroup_bucket",
        "frequent_pearson_x_spacegroup_bucket",
        "frequent_pearson_x_wyckoff_bucket",
        "frequent_formula_x_wyckoff_bucket",
        "frequent_spacegroup_x_wyckoff_bucket",
    ],
    "components_plus_archetype_plus_interactions": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        "frequent_archetype_bucket",
        "frequent_formula_x_pearson_bucket",
        "frequent_formula_x_spacegroup_bucket",
        "frequent_pearson_x_spacegroup_bucket",
        "frequent_pearson_x_wyckoff_bucket",
        "frequent_formula_x_wyckoff_bucket",
        "frequent_spacegroup_x_wyckoff_bucket",
    ],
    "components_plus_full_token": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        "frequent_prototype_token_bucket",
    ],
}
ROBUST_INTERACTION_MIN_N = 10


def load_background_frame() -> pd.DataFrame:
    frame = assign_component_branch(load_analysis_frame())
    frame = frame.loc[frame["component_branch"] == "background"].copy()
    interactions = pd.read_parquet(
        PROCESSED_DIR / "singleton_high_risk_background_component_interaction_proxy.parquet",
    )
    merged = frame.merge(interactions, on="material_id", how="left", validate="one_to_one", suffixes=("", "_proxy"))
    for column in INTERACTION_COLUMNS:
        merged[column] = merged[column].fillna("other")
    return merged


def compute_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "balanced_accuracy_mean",
        "average_precision_mean",
        "roc_auc_mean",
        "log_loss_mean",
        "n_materials",
        "positive_rate",
    ]
    pivot = (
        summary.set_index(["outcome", "feature_set"])[value_columns]
        .unstack("feature_set")
        .sort_index()
    )
    out = pd.DataFrame(index=pivot.index).reset_index()
    out["n_materials"] = pivot[("n_materials", "components_full")].to_numpy()
    out["positive_rate"] = pivot[("positive_rate", "components_full")].to_numpy()
    baseline_name = "components_full"
    full_token_name = "components_plus_full_token"
    interaction_name = "components_plus_interaction_bundle"
    archetype_interaction_name = "components_plus_archetype_plus_interactions"

    for feature_set_name in FEATURE_SETS:
        if feature_set_name == baseline_name:
            continue
        out[f"{feature_set_name}_balanced_accuracy_gain"] = (
            pivot[("balanced_accuracy_mean", feature_set_name)]
            - pivot[("balanced_accuracy_mean", baseline_name)]
        ).to_numpy()
        out[f"{feature_set_name}_average_precision_gain"] = (
            pivot[("average_precision_mean", feature_set_name)]
            - pivot[("average_precision_mean", baseline_name)]
        ).to_numpy()
        out[f"{feature_set_name}_log_loss_improvement"] = (
            pivot[("log_loss_mean", baseline_name)] - pivot[("log_loss_mean", feature_set_name)]
        ).to_numpy()

    out["interaction_bundle_vs_full_token_balanced_accuracy_delta"] = (
        pivot[("balanced_accuracy_mean", interaction_name)]
        - pivot[("balanced_accuracy_mean", full_token_name)]
    ).to_numpy()
    out["archetype_interactions_vs_full_token_balanced_accuracy_delta"] = (
        pivot[("balanced_accuracy_mean", archetype_interaction_name)]
        - pivot[("balanced_accuracy_mean", full_token_name)]
    ).to_numpy()
    out["interaction_bundle_vs_full_token_average_precision_delta"] = (
        pivot[("average_precision_mean", interaction_name)]
        - pivot[("average_precision_mean", full_token_name)]
    ).to_numpy()
    out["archetype_interactions_vs_full_token_average_precision_delta"] = (
        pivot[("average_precision_mean", archetype_interaction_name)]
        - pivot[("average_precision_mean", full_token_name)]
    ).to_numpy()
    return out


def build_interaction_rate_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        frame.groupby(column, as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
        )
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_18"
    ensure_dir(output_dir)

    frame = load_background_frame()
    cv_folds, cv_summary = run_feature_set_suite(
        frame=frame,
        scenario_name="background_interactions",
        feature_sets=FEATURE_SETS,
    )
    delta_table = compute_deltas(cv_summary)

    interaction_tables: dict[str, pd.DataFrame] = {}
    robust_interaction_tables: dict[str, pd.DataFrame] = {}
    for column in INTERACTION_COLUMNS:
        table = build_interaction_rate_table(frame, column)
        interaction_tables[column] = table
        robust_interaction_tables[column] = table.loc[
            (table["n_materials"] >= ROBUST_INTERACTION_MIN_N) & (table[column] != "other")
        ].copy()

    summary_lookup = cv_summary.set_index(["feature_set", "outcome"])
    delta_lookup = delta_table.set_index("outcome")
    summary = {
        "background_branch": {
            "n_materials": int(len(frame)),
            "feature_sets": {
                outcome: {
                    feature_set: summary_lookup.loc[(feature_set, outcome), [
                        "balanced_accuracy_mean",
                        "average_precision_mean",
                        "log_loss_mean",
                    ]].to_dict()
                    for feature_set in FEATURE_SETS
                }
                for outcome in OUTCOMES
            },
            "key_deltas": {
                outcome: {
                    "full_token_balanced_accuracy_gain": float(
                        delta_lookup.loc[outcome, "components_plus_full_token_balanced_accuracy_gain"]
                    ),
                    "interaction_bundle_balanced_accuracy_gain": float(
                        delta_lookup.loc[outcome, "components_plus_interaction_bundle_balanced_accuracy_gain"]
                    ),
                    "archetype_interactions_balanced_accuracy_gain": float(
                        delta_lookup.loc[outcome, "components_plus_archetype_plus_interactions_balanced_accuracy_gain"]
                    ),
                    "interaction_bundle_vs_full_token_balanced_accuracy_delta": float(
                        delta_lookup.loc[outcome, "interaction_bundle_vs_full_token_balanced_accuracy_delta"]
                    ),
                    "archetype_interactions_vs_full_token_balanced_accuracy_delta": float(
                        delta_lookup.loc[outcome, "archetype_interactions_vs_full_token_balanced_accuracy_delta"]
                    ),
                }
                for outcome in OUTCOMES
            },
            "top_robust_interaction_buckets": {
                column: (
                    robust_interaction_tables[column].iloc[0][
                        [column, "n_materials", "collective_failure_rate", "collective_false_negative_rate"]
                    ].to_dict()
                    if not robust_interaction_tables[column].empty
                    else {}
                )
                for column in INTERACTION_COLUMNS
            },
        }
    }

    cv_folds.to_csv(output_dir / "feature_set_cv_fold_metrics.csv", index=False)
    cv_summary.to_csv(output_dir / "feature_set_cv_summary.csv", index=False)
    delta_table.to_csv(output_dir / "feature_set_delta_summary.csv", index=False)
    for column, table in interaction_tables.items():
        table.to_csv(output_dir / f"{column}_rates.csv", index=False)
        robust_interaction_tables[column].to_csv(output_dir / f"robust_{column}_rates.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

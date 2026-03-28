from __future__ import annotations

import json

import pandas as pd

from analysis_14_hotspot_ablation import OUTCOMES, run_feature_set_suite
from analysis_15_prototype_component_decomposition import load_analysis_frame
from analysis_16_component_ablation import assign_component_branch
from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


PAIRWISE_COLUMNS = [
    "frequent_formula_x_pearson_bucket",
    "frequent_formula_x_spacegroup_bucket",
    "frequent_pearson_x_spacegroup_bucket",
    "frequent_pearson_x_wyckoff_bucket",
    "frequent_formula_x_wyckoff_bucket",
    "frequent_spacegroup_x_wyckoff_bucket",
]
TRIPLE_COLUMNS = [
    "frequent_formula_x_pearson_x_spacegroup_bucket",
    "frequent_formula_x_pearson_x_wyckoff_bucket",
    "frequent_formula_x_spacegroup_x_wyckoff_bucket",
    "frequent_pearson_x_spacegroup_x_wyckoff_bucket",
]
FEATURE_SETS = {
    "components_plus_pairwise_plus_triples": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        *PAIRWISE_COLUMNS,
        *TRIPLE_COLUMNS,
    ],
    "components_plus_pairwise_plus_triples_plus_residual_bucket": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        *PAIRWISE_COLUMNS,
        *TRIPLE_COLUMNS,
        "frequent_background_residual_token_bucket",
    ],
    "components_plus_pairwise_plus_triples_plus_full_token": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        *PAIRWISE_COLUMNS,
        *TRIPLE_COLUMNS,
        "frequent_prototype_token_bucket",
    ],
}
ROBUST_RESIDUAL_MIN_N = 10


def load_background_frame() -> pd.DataFrame:
    frame = assign_component_branch(load_analysis_frame())
    frame = frame.loc[frame["component_branch"] == "background"].copy()
    pairwise = pd.read_parquet(
        PROCESSED_DIR / "singleton_high_risk_background_component_interaction_proxy.parquet",
    )
    higher_order = pd.read_parquet(
        PROCESSED_DIR / "singleton_high_risk_background_higher_order_interaction_proxy.parquet",
    )
    residual = pd.read_parquet(
        PROCESSED_DIR / "singleton_high_risk_background_residual_token_proxy.parquet",
    )
    merged = (
        frame.merge(pairwise, on="material_id", how="left", validate="one_to_one", suffixes=("", "_pair"))
        .merge(higher_order, on="material_id", how="left", validate="one_to_one", suffixes=("", "_triple"))
        .merge(residual[["material_id", "frequent_background_residual_token_bucket"]], on="material_id", how="left", validate="one_to_one")
    )
    fill_columns = PAIRWISE_COLUMNS + TRIPLE_COLUMNS + ["frequent_background_residual_token_bucket"]
    for column in fill_columns:
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
    out["n_materials"] = pivot[("n_materials", "components_plus_pairwise_plus_triples")].to_numpy()
    out["positive_rate"] = pivot[("positive_rate", "components_plus_pairwise_plus_triples")].to_numpy()

    base_name = "components_plus_pairwise_plus_triples"
    residual_name = "components_plus_pairwise_plus_triples_plus_residual_bucket"
    full_name = "components_plus_pairwise_plus_triples_plus_full_token"

    out["residual_bucket_balanced_accuracy_gain"] = (
        pivot[("balanced_accuracy_mean", residual_name)] - pivot[("balanced_accuracy_mean", base_name)]
    ).to_numpy()
    out["full_token_balanced_accuracy_gain"] = (
        pivot[("balanced_accuracy_mean", full_name)] - pivot[("balanced_accuracy_mean", base_name)]
    ).to_numpy()
    out["residual_vs_full_token_balanced_accuracy_delta"] = (
        pivot[("balanced_accuracy_mean", residual_name)] - pivot[("balanced_accuracy_mean", full_name)]
    ).to_numpy()
    out["residual_bucket_average_precision_gain"] = (
        pivot[("average_precision_mean", residual_name)] - pivot[("average_precision_mean", base_name)]
    ).to_numpy()
    out["full_token_average_precision_gain"] = (
        pivot[("average_precision_mean", full_name)] - pivot[("average_precision_mean", base_name)]
    ).to_numpy()
    out["residual_vs_full_token_average_precision_delta"] = (
        pivot[("average_precision_mean", residual_name)] - pivot[("average_precision_mean", full_name)]
    ).to_numpy()
    out["residual_bucket_log_loss_improvement"] = (
        pivot[("log_loss_mean", base_name)] - pivot[("log_loss_mean", residual_name)]
    ).to_numpy()
    out["full_token_log_loss_improvement"] = (
        pivot[("log_loss_mean", base_name)] - pivot[("log_loss_mean", full_name)]
    ).to_numpy()
    out["residual_vs_full_token_log_loss_delta"] = (
        pivot[("log_loss_mean", full_name)] - pivot[("log_loss_mean", residual_name)]
    ).to_numpy()
    return out


def build_rate_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("frequent_background_residual_token_bucket", as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
        )
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_20"
    ensure_dir(output_dir)

    frame = load_background_frame()
    cv_folds, cv_summary = run_feature_set_suite(
        frame=frame,
        scenario_name="background_residual_token_layer",
        feature_sets=FEATURE_SETS,
    )
    delta_table = compute_deltas(cv_summary)

    residual_rates = build_rate_table(frame)
    robust_residual_rates = residual_rates.loc[
        (residual_rates["n_materials"] >= ROBUST_RESIDUAL_MIN_N)
        & (residual_rates["frequent_background_residual_token_bucket"] != "other")
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
                    "residual_bucket_balanced_accuracy_gain": float(
                        delta_lookup.loc[outcome, "residual_bucket_balanced_accuracy_gain"]
                    ),
                    "full_token_balanced_accuracy_gain": float(
                        delta_lookup.loc[outcome, "full_token_balanced_accuracy_gain"]
                    ),
                    "residual_vs_full_token_balanced_accuracy_delta": float(
                        delta_lookup.loc[outcome, "residual_vs_full_token_balanced_accuracy_delta"]
                    ),
                    "residual_bucket_average_precision_gain": float(
                        delta_lookup.loc[outcome, "residual_bucket_average_precision_gain"]
                    ),
                    "full_token_average_precision_gain": float(
                        delta_lookup.loc[outcome, "full_token_average_precision_gain"]
                    ),
                    "residual_vs_full_token_average_precision_delta": float(
                        delta_lookup.loc[outcome, "residual_vs_full_token_average_precision_delta"]
                    ),
                }
                for outcome in OUTCOMES
            },
            "top_robust_residual_buckets": robust_residual_rates.head(15).to_dict(orient="records"),
        }
    }

    cv_folds.to_csv(output_dir / "feature_set_cv_fold_metrics.csv", index=False)
    cv_summary.to_csv(output_dir / "feature_set_cv_summary.csv", index=False)
    delta_table.to_csv(output_dir / "feature_set_delta_summary.csv", index=False)
    residual_rates.to_csv(output_dir / "background_residual_token_bucket_rates.csv", index=False)
    robust_residual_rates.to_csv(output_dir / "robust_background_residual_token_bucket_rates.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

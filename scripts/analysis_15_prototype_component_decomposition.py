from __future__ import annotations

import json

import pandas as pd

from analysis_14_hotspot_ablation import OUTCOMES, run_feature_set_suite
from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


FEATURE_SETS = {
    "spacegroup_control": ["crystal_system", "frequent_spacegroup_bucket"],
    "spacegroup_plus_formula_family": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
    ],
    "spacegroup_plus_pearson": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_pearson_symbol_bucket",
    ],
    "spacegroup_plus_archetype": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_archetype_bucket",
    ],
    "spacegroup_plus_wyckoff_signature": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_wyckoff_signature_bucket",
    ],
    "spacegroup_plus_components": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
    ],
    "combined_hotspot_buckets": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_prototype_token_bucket",
    ],
}
ROBUST_COMPONENT_MIN_N = 20


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
    hotspot = pd.read_parquet(
        PROCESSED_DIR / "wbm_mptrj_mp2022_union_material_motif_hotspot_proxy.parquet",
        columns=[
            "material_id",
            "singleton_high_risk_candidate",
            "frequent_prototype_token_bucket",
            "frequent_spacegroup_bucket",
        ],
    )
    components = pd.read_parquet(
        PROCESSED_DIR / "wbm_mptrj_mp2022_union_material_prototype_component_proxy.parquet",
        columns=[
            "material_id",
            "frequent_formula_family_bucket",
            "frequent_pearson_symbol_bucket",
            "frequent_archetype_bucket",
            "frequent_wyckoff_signature_bucket",
            "prototype_formula_family",
            "prototype_pearson_symbol",
            "prototype_archetype_token",
            "prototype_wyckoff_signature",
        ],
    )

    frame = (
        features.merge(hotspot, on="material_id", how="left", validate="one_to_one")
        .merge(components, on="material_id", how="left", validate="one_to_one")
    )
    frame = frame.loc[frame["collective_failure"] | frame["collective_success"]].copy()
    frame = frame.loc[frame["singleton_high_risk_candidate"]].copy()
    fill_columns = [
        "frequent_prototype_token_bucket",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_archetype_bucket",
        "frequent_wyckoff_signature_bucket",
        "prototype_formula_family",
        "prototype_pearson_symbol",
        "prototype_archetype_token",
        "prototype_wyckoff_signature",
    ]
    for column in fill_columns:
        frame[column] = frame[column].fillna("other")
    frame["crystal_system"] = frame["crystal_system"].fillna("unknown")
    return frame


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
    delta = pd.DataFrame(index=pivot.index).reset_index()
    delta["n_materials"] = pivot[("n_materials", "combined_hotspot_buckets")].to_numpy()
    delta["positive_rate"] = pivot[("positive_rate", "combined_hotspot_buckets")].to_numpy()
    delta["combined_vs_spacegroup_balanced_accuracy_delta"] = (
        pivot[("balanced_accuracy_mean", "combined_hotspot_buckets")]
        - pivot[("balanced_accuracy_mean", "spacegroup_control")]
    ).to_numpy()
    delta["components_vs_spacegroup_balanced_accuracy_delta"] = (
        pivot[("balanced_accuracy_mean", "spacegroup_plus_components")]
        - pivot[("balanced_accuracy_mean", "spacegroup_control")]
    ).to_numpy()
    delta["combined_vs_components_balanced_accuracy_delta"] = (
        pivot[("balanced_accuracy_mean", "combined_hotspot_buckets")]
        - pivot[("balanced_accuracy_mean", "spacegroup_plus_components")]
    ).to_numpy()
    delta["archetype_vs_spacegroup_balanced_accuracy_delta"] = (
        pivot[("balanced_accuracy_mean", "spacegroup_plus_archetype")]
        - pivot[("balanced_accuracy_mean", "spacegroup_control")]
    ).to_numpy()
    delta["wyckoff_vs_spacegroup_balanced_accuracy_delta"] = (
        pivot[("balanced_accuracy_mean", "spacegroup_plus_wyckoff_signature")]
        - pivot[("balanced_accuracy_mean", "spacegroup_control")]
    ).to_numpy()
    delta["combined_vs_spacegroup_average_precision_delta"] = (
        pivot[("average_precision_mean", "combined_hotspot_buckets")]
        - pivot[("average_precision_mean", "spacegroup_control")]
    ).to_numpy()
    delta["components_vs_spacegroup_average_precision_delta"] = (
        pivot[("average_precision_mean", "spacegroup_plus_components")]
        - pivot[("average_precision_mean", "spacegroup_control")]
    ).to_numpy()
    delta["combined_vs_components_average_precision_delta"] = (
        pivot[("average_precision_mean", "combined_hotspot_buckets")]
        - pivot[("average_precision_mean", "spacegroup_plus_components")]
    ).to_numpy()
    delta["combined_vs_spacegroup_log_loss_improvement"] = (
        pivot[("log_loss_mean", "spacegroup_control")]
        - pivot[("log_loss_mean", "combined_hotspot_buckets")]
    ).to_numpy()
    delta["components_vs_spacegroup_log_loss_improvement"] = (
        pivot[("log_loss_mean", "spacegroup_control")]
        - pivot[("log_loss_mean", "spacegroup_plus_components")]
    ).to_numpy()
    delta["combined_vs_components_log_loss_improvement"] = (
        pivot[("log_loss_mean", "spacegroup_plus_components")]
        - pivot[("log_loss_mean", "combined_hotspot_buckets")]
    ).to_numpy()
    return delta


def build_bucket_support_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    formula = (
        frame.groupby("frequent_formula_family_bucket", as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
        )
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )
    pearson = (
        frame.groupby("frequent_pearson_symbol_bucket", as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
        )
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )
    archetype = (
        frame.groupby("frequent_archetype_bucket", as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
        )
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )
    wyckoff = (
        frame.groupby("frequent_wyckoff_signature_bucket", as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
        )
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return formula, pearson, archetype, wyckoff


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_15"
    ensure_dir(output_dir)

    frame = load_analysis_frame()
    cv_folds, cv_summary = run_feature_set_suite(
        frame=frame,
        scenario_name="prototype_component_decomposition",
        feature_sets=FEATURE_SETS,
    )
    delta_table = compute_deltas(cv_summary)
    formula_rates, pearson_rates, archetype_rates, wyckoff_rates = build_bucket_support_tables(frame)
    robust_formula_rates = formula_rates.loc[
        (formula_rates["n_materials"] >= ROBUST_COMPONENT_MIN_N)
        & (formula_rates["frequent_formula_family_bucket"] != "other")
    ].copy()
    robust_pearson_rates = pearson_rates.loc[
        (pearson_rates["n_materials"] >= ROBUST_COMPONENT_MIN_N)
        & (pearson_rates["frequent_pearson_symbol_bucket"] != "other")
    ].copy()
    robust_archetype_rates = archetype_rates.loc[
        (archetype_rates["n_materials"] >= ROBUST_COMPONENT_MIN_N)
        & (archetype_rates["frequent_archetype_bucket"] != "other")
    ].copy()
    robust_wyckoff_rates = wyckoff_rates.loc[
        (wyckoff_rates["n_materials"] >= ROBUST_COMPONENT_MIN_N)
        & (wyckoff_rates["frequent_wyckoff_signature_bucket"] != "other")
    ].copy()

    summary_lookup = cv_summary.set_index(["feature_set", "outcome"])
    delta_lookup = delta_table.set_index("outcome")
    summary = {
        "singleton_high_risk_subset": {
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
                    "combined_vs_spacegroup_balanced_accuracy_delta": float(
                        delta_lookup.loc[outcome, "combined_vs_spacegroup_balanced_accuracy_delta"]
                    ),
                    "components_vs_spacegroup_balanced_accuracy_delta": float(
                        delta_lookup.loc[outcome, "components_vs_spacegroup_balanced_accuracy_delta"]
                    ),
                    "combined_vs_components_balanced_accuracy_delta": float(
                        delta_lookup.loc[outcome, "combined_vs_components_balanced_accuracy_delta"]
                    ),
                    "archetype_vs_spacegroup_balanced_accuracy_delta": float(
                        delta_lookup.loc[outcome, "archetype_vs_spacegroup_balanced_accuracy_delta"]
                    ),
                    "wyckoff_vs_spacegroup_balanced_accuracy_delta": float(
                        delta_lookup.loc[outcome, "wyckoff_vs_spacegroup_balanced_accuracy_delta"]
                    ),
                }
                for outcome in OUTCOMES
            },
            "top_robust_component_buckets": {
                "formula_family": (
                    robust_formula_rates.iloc[0][
                        [
                            "frequent_formula_family_bucket",
                            "n_materials",
                            "collective_failure_rate",
                            "collective_false_negative_rate",
                        ]
                    ].to_dict()
                    if not robust_formula_rates.empty
                    else {}
                ),
                "pearson_symbol": (
                    robust_pearson_rates.iloc[0][
                        [
                            "frequent_pearson_symbol_bucket",
                            "n_materials",
                            "collective_failure_rate",
                            "collective_false_negative_rate",
                        ]
                    ].to_dict()
                    if not robust_pearson_rates.empty
                    else {}
                ),
                "archetype": (
                    robust_archetype_rates.iloc[0][
                        [
                            "frequent_archetype_bucket",
                            "n_materials",
                            "collective_failure_rate",
                            "collective_false_negative_rate",
                        ]
                    ].to_dict()
                    if not robust_archetype_rates.empty
                    else {}
                ),
                "wyckoff_signature": (
                    robust_wyckoff_rates.iloc[0][
                        [
                            "frequent_wyckoff_signature_bucket",
                            "n_materials",
                            "collective_failure_rate",
                            "collective_false_negative_rate",
                        ]
                    ].to_dict()
                    if not robust_wyckoff_rates.empty
                    else {}
                ),
            },
        }
    }

    cv_folds.to_csv(output_dir / "feature_set_cv_fold_metrics.csv", index=False)
    cv_summary.to_csv(output_dir / "feature_set_cv_summary.csv", index=False)
    delta_table.to_csv(output_dir / "feature_set_delta_summary.csv", index=False)
    formula_rates.to_csv(output_dir / "formula_family_bucket_rates.csv", index=False)
    pearson_rates.to_csv(output_dir / "pearson_symbol_bucket_rates.csv", index=False)
    archetype_rates.to_csv(output_dir / "archetype_bucket_rates.csv", index=False)
    wyckoff_rates.to_csv(output_dir / "wyckoff_signature_bucket_rates.csv", index=False)
    robust_formula_rates.to_csv(output_dir / "robust_formula_family_bucket_rates.csv", index=False)
    robust_pearson_rates.to_csv(output_dir / "robust_pearson_symbol_bucket_rates.csv", index=False)
    robust_archetype_rates.to_csv(output_dir / "robust_archetype_bucket_rates.csv", index=False)
    robust_wyckoff_rates.to_csv(output_dir / "robust_wyckoff_signature_bucket_rates.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

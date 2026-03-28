from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis_14_hotspot_ablation import OUTCOMES, run_feature_set_suite
from analysis_15_prototype_component_decomposition import load_analysis_frame
from dva_project.settings import RESULTS_DIR
from dva_project.utils import ensure_dir


BOOTSTRAP_RESAMPLES = 2000
FEATURE_SETS = {
    "spacegroup_control": ["crystal_system", "frequent_spacegroup_bucket"],
    "components_full": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
    ],
    "components_minus_formula_family": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
    ],
    "components_minus_pearson": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_wyckoff_signature_bucket",
    ],
    "components_minus_wyckoff": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
    ],
    "components_plus_archetype": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        "frequent_archetype_bucket",
    ],
    "components_plus_full_token": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        "frequent_prototype_token_bucket",
    ],
    "components_plus_archetype_plus_full_token": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
        "frequent_archetype_bucket",
        "frequent_prototype_token_bucket",
    ],
}


def compute_ablation_deltas(summary: pd.DataFrame) -> pd.DataFrame:
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
    out["components_vs_spacegroup_balanced_accuracy_delta"] = (
        pivot[("balanced_accuracy_mean", "components_full")]
        - pivot[("balanced_accuracy_mean", "spacegroup_control")]
    ).to_numpy()
    out["drop_formula_family_balanced_accuracy"] = (
        pivot[("balanced_accuracy_mean", "components_full")]
        - pivot[("balanced_accuracy_mean", "components_minus_formula_family")]
    ).to_numpy()
    out["drop_pearson_balanced_accuracy"] = (
        pivot[("balanced_accuracy_mean", "components_full")]
        - pivot[("balanced_accuracy_mean", "components_minus_pearson")]
    ).to_numpy()
    out["drop_wyckoff_balanced_accuracy"] = (
        pivot[("balanced_accuracy_mean", "components_full")]
        - pivot[("balanced_accuracy_mean", "components_minus_wyckoff")]
    ).to_numpy()
    out["add_archetype_balanced_accuracy"] = (
        pivot[("balanced_accuracy_mean", "components_plus_archetype")]
        - pivot[("balanced_accuracy_mean", "components_full")]
    ).to_numpy()
    out["add_full_token_balanced_accuracy"] = (
        pivot[("balanced_accuracy_mean", "components_plus_full_token")]
        - pivot[("balanced_accuracy_mean", "components_full")]
    ).to_numpy()
    out["add_archetype_plus_full_token_balanced_accuracy"] = (
        pivot[("balanced_accuracy_mean", "components_plus_archetype_plus_full_token")]
        - pivot[("balanced_accuracy_mean", "components_full")]
    ).to_numpy()
    out["drop_formula_family_average_precision"] = (
        pivot[("average_precision_mean", "components_full")]
        - pivot[("average_precision_mean", "components_minus_formula_family")]
    ).to_numpy()
    out["drop_pearson_average_precision"] = (
        pivot[("average_precision_mean", "components_full")]
        - pivot[("average_precision_mean", "components_minus_pearson")]
    ).to_numpy()
    out["drop_wyckoff_average_precision"] = (
        pivot[("average_precision_mean", "components_full")]
        - pivot[("average_precision_mean", "components_minus_wyckoff")]
    ).to_numpy()
    out["add_archetype_average_precision"] = (
        pivot[("average_precision_mean", "components_plus_archetype")]
        - pivot[("average_precision_mean", "components_full")]
    ).to_numpy()
    out["add_full_token_average_precision"] = (
        pivot[("average_precision_mean", "components_plus_full_token")]
        - pivot[("average_precision_mean", "components_full")]
    ).to_numpy()
    out["add_archetype_plus_full_token_average_precision"] = (
        pivot[("average_precision_mean", "components_plus_archetype_plus_full_token")]
        - pivot[("average_precision_mean", "components_full")]
    ).to_numpy()
    out["drop_formula_family_log_loss_cost"] = (
        pivot[("log_loss_mean", "components_minus_formula_family")]
        - pivot[("log_loss_mean", "components_full")]
    ).to_numpy()
    out["drop_pearson_log_loss_cost"] = (
        pivot[("log_loss_mean", "components_minus_pearson")]
        - pivot[("log_loss_mean", "components_full")]
    ).to_numpy()
    out["drop_wyckoff_log_loss_cost"] = (
        pivot[("log_loss_mean", "components_minus_wyckoff")]
        - pivot[("log_loss_mean", "components_full")]
    ).to_numpy()
    out["add_archetype_log_loss_improvement"] = (
        pivot[("log_loss_mean", "components_full")]
        - pivot[("log_loss_mean", "components_plus_archetype")]
    ).to_numpy()
    out["add_full_token_log_loss_improvement"] = (
        pivot[("log_loss_mean", "components_full")]
        - pivot[("log_loss_mean", "components_plus_full_token")]
    ).to_numpy()
    out["add_archetype_plus_full_token_log_loss_improvement"] = (
        pivot[("log_loss_mean", "components_full")]
        - pivot[("log_loss_mean", "components_plus_archetype_plus_full_token")]
    ).to_numpy()
    return out


def assign_component_branch(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    is_ti10 = enriched["frequent_pearson_symbol_bucket"] == "tI10"
    is_archetype = enriched["frequent_archetype_bucket"] == "ABC_oP12_62"
    is_wyckoff = enriched["frequent_wyckoff_signature_bucket"] == "c_c_c"

    enriched["is_tI10_branch"] = is_ti10
    enriched["is_ABC_oP12_62_branch"] = is_archetype
    enriched["is_c_c_c_branch"] = is_wyckoff
    enriched["top_component_combo"] = (
        is_ti10.astype(int).astype(str)
        + is_archetype.astype(int).astype(str)
        + is_wyckoff.astype(int).astype(str)
    )
    enriched["component_branch"] = "background"
    enriched.loc[is_wyckoff & ~is_archetype, "component_branch"] = "c_c_c_residual_branch"
    enriched.loc[is_archetype, "component_branch"] = "ABC_oP12_62_branch"
    enriched.loc[is_ti10, "component_branch"] = "tI10_branch"
    return enriched


def build_branch_rate_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combo = (
        frame.groupby("top_component_combo", as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
        )
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )
    branch = (
        frame.groupby("component_branch", as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
        )
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return combo, branch


def bootstrap_rate_difference(
    frame: pd.DataFrame,
    group_a: str,
    group_b: str,
    outcome: str,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict:
    a_values = frame.loc[frame["component_branch"] == group_a, outcome].to_numpy(dtype=float)
    b_values = frame.loc[frame["component_branch"] == group_b, outcome].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    observed = float(a_values.mean() - b_values.mean())
    draws = np.empty(n_resamples, dtype=float)
    for idx in range(n_resamples):
        draws[idx] = (
            rng.choice(a_values, size=len(a_values), replace=True).mean()
            - rng.choice(b_values, size=len(b_values), replace=True).mean()
        )

    return {
        "group_a": group_a,
        "group_b": group_b,
        "outcome": outcome,
        "n_group_a": int(len(a_values)),
        "n_group_b": int(len(b_values)),
        "observed_rate_diff": observed,
        "q025": float(np.quantile(draws, 0.025)),
        "q975": float(np.quantile(draws, 0.975)),
        "fraction_diff_gt_zero": float((draws > 0).mean()),
    }


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_16"
    ensure_dir(output_dir)

    frame = assign_component_branch(load_analysis_frame())
    cv_folds, cv_summary = run_feature_set_suite(
        frame=frame,
        scenario_name="component_ablation",
        feature_sets=FEATURE_SETS,
    )
    delta_table = compute_ablation_deltas(cv_summary)

    combo_rates, branch_rates = build_branch_rate_tables(frame)
    branch_bootstrap_rows: list[dict] = []
    branch_pairs = [
        ("tI10_branch", "background"),
        ("ABC_oP12_62_branch", "background"),
        ("c_c_c_residual_branch", "background"),
        ("ABC_oP12_62_branch", "c_c_c_residual_branch"),
        ("tI10_branch", "ABC_oP12_62_branch"),
    ]
    for group_a, group_b in branch_pairs:
        for outcome in OUTCOMES:
            branch_bootstrap_rows.append(
                bootstrap_rate_difference(
                    frame=frame,
                    group_a=group_a,
                    group_b=group_b,
                    outcome=outcome,
                    seed=101 + len(branch_bootstrap_rows),
                )
            )
    branch_bootstrap = pd.DataFrame(branch_bootstrap_rows)

    delta_lookup = delta_table.set_index("outcome")
    branch_lookup = branch_rates.set_index("component_branch")
    summary = {
        "singleton_high_risk_subset": {
            "n_materials": int(len(frame)),
            "component_ablation": {
                outcome: {
                    "components_vs_spacegroup_balanced_accuracy_delta": float(
                        delta_lookup.loc[outcome, "components_vs_spacegroup_balanced_accuracy_delta"]
                    ),
                    "drop_formula_family_balanced_accuracy": float(
                        delta_lookup.loc[outcome, "drop_formula_family_balanced_accuracy"]
                    ),
                    "drop_pearson_balanced_accuracy": float(
                        delta_lookup.loc[outcome, "drop_pearson_balanced_accuracy"]
                    ),
                    "drop_wyckoff_balanced_accuracy": float(
                        delta_lookup.loc[outcome, "drop_wyckoff_balanced_accuracy"]
                    ),
                    "add_archetype_balanced_accuracy": float(
                        delta_lookup.loc[outcome, "add_archetype_balanced_accuracy"]
                    ),
                    "add_full_token_balanced_accuracy": float(
                        delta_lookup.loc[outcome, "add_full_token_balanced_accuracy"]
                    ),
                    "add_archetype_plus_full_token_balanced_accuracy": float(
                        delta_lookup.loc[outcome, "add_archetype_plus_full_token_balanced_accuracy"]
                    ),
                }
                for outcome in OUTCOMES
            },
            "branch_rates": {
                branch: {
                    "n_materials": int(branch_lookup.loc[branch, "n_materials"]),
                    "collective_failure_rate": float(branch_lookup.loc[branch, "collective_failure_rate"]),
                    "collective_false_negative_rate": float(branch_lookup.loc[branch, "collective_false_negative_rate"]),
                }
                for branch in ["tI10_branch", "ABC_oP12_62_branch", "c_c_c_residual_branch", "background"]
            },
        }
    }

    cv_folds.to_csv(output_dir / "feature_set_cv_fold_metrics.csv", index=False)
    cv_summary.to_csv(output_dir / "feature_set_cv_summary.csv", index=False)
    delta_table.to_csv(output_dir / "feature_set_ablation_summary.csv", index=False)
    combo_rates.to_csv(output_dir / "top_component_combo_rates.csv", index=False)
    branch_rates.to_csv(output_dir / "component_branch_rates.csv", index=False)
    branch_bootstrap.to_csv(output_dir / "component_branch_bootstrap_differences.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

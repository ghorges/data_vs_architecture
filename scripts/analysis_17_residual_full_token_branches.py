from __future__ import annotations

import json

import pandas as pd

from analysis_13_motif_hotspot_modeling import bootstrap_bucket_vs_other
from analysis_14_hotspot_ablation import OUTCOMES, evaluate_feature_set
from analysis_15_prototype_component_decomposition import load_analysis_frame
from analysis_16_component_ablation import assign_component_branch
from dva_project.settings import RESULTS_DIR
from dva_project.utils import ensure_dir


GLOBAL_FEATURE_SETS = {
    "components_full": [
        "crystal_system",
        "frequent_spacegroup_bucket",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
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
BRANCH_MIN_TOKEN_N = 4


def run_feature_suite(
    frame: pd.DataFrame,
    scenario_name: str,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def compute_token_gain(summary: pd.DataFrame, scenario_group: str) -> pd.DataFrame:
    value_columns = [
        "balanced_accuracy_mean",
        "average_precision_mean",
        "roc_auc_mean",
        "log_loss_mean",
        "n_materials",
        "positive_rate",
    ]
    pivot = (
        summary.set_index(["scenario", "outcome", "feature_set"])[value_columns]
        .unstack("feature_set")
        .sort_index()
    )
    gain = pd.DataFrame(index=pivot.index).reset_index()
    gain["scenario_group"] = scenario_group
    base_feature_set = "components_full"
    full_token_feature_set = "components_plus_full_token"
    if ("n_materials", base_feature_set) not in pivot.columns:
        base_feature_set = "branch_components"
        full_token_feature_set = "branch_components_plus_full_token"

    gain["n_materials"] = pivot[("n_materials", base_feature_set)].to_numpy()
    gain["positive_rate"] = pivot[("positive_rate", base_feature_set)].to_numpy()
    gain["full_token_balanced_accuracy_gain"] = (
        pivot[("balanced_accuracy_mean", full_token_feature_set)]
        - pivot[("balanced_accuracy_mean", base_feature_set)]
    ).to_numpy()
    gain["full_token_average_precision_gain"] = (
        pivot[("average_precision_mean", full_token_feature_set)]
        - pivot[("average_precision_mean", base_feature_set)]
    ).to_numpy()
    gain["full_token_roc_auc_gain"] = (
        pivot[("roc_auc_mean", full_token_feature_set)]
        - pivot[("roc_auc_mean", base_feature_set)]
    ).to_numpy()
    gain["full_token_log_loss_improvement"] = (
        pivot[("log_loss_mean", base_feature_set)] - pivot[("log_loss_mean", full_token_feature_set)]
    ).to_numpy()
    return gain


def mask_tokens_in_branch(frame: pd.DataFrame, branch_name: str) -> pd.DataFrame:
    masked = frame.copy()
    mask = masked["component_branch"] == branch_name
    masked.loc[mask, "frequent_prototype_token_bucket"] = f"masked__{branch_name}"
    return masked


def build_global_branch_mask_scenarios(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    background_mask = frame["component_branch"] == "background"
    nonbackground_mask = frame["component_branch"].isin(
        ["tI10_branch", "ABC_oP12_62_branch", "c_c_c_residual_branch"]
    )

    scenarios = {
        "baseline": frame.copy(),
        "mask_tI10_branch_tokens": mask_tokens_in_branch(frame, "tI10_branch"),
        "mask_ABC_oP12_62_branch_tokens": mask_tokens_in_branch(frame, "ABC_oP12_62_branch"),
        "mask_c_c_c_residual_branch_tokens": mask_tokens_in_branch(frame, "c_c_c_residual_branch"),
        "mask_background_tokens": frame.copy(),
        "mask_nonbackground_tokens": frame.copy(),
    }
    scenarios["mask_background_tokens"].loc[background_mask, "frequent_prototype_token_bucket"] = "masked__background"
    scenarios["mask_nonbackground_tokens"].loc[
        nonbackground_mask, "frequent_prototype_token_bucket"
    ] = "masked__nonbackground"
    return scenarios


def build_branch_token_rate_table(frame: pd.DataFrame, branch_name: str) -> pd.DataFrame:
    subset = frame.loc[frame["component_branch"] == branch_name].copy()
    return (
        subset.groupby("frequent_prototype_token_bucket", as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
        )
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )


def bootstrap_branch_token_vs_rest(
    frame: pd.DataFrame,
    branch_name: str,
    token_label: str,
    outcome: str,
    seed: int,
) -> dict:
    subset = frame.loc[frame["component_branch"] == branch_name].copy()
    return bootstrap_bucket_vs_other(
        frame=subset,
        bucket_column="frequent_prototype_token_bucket",
        bucket_label=token_label,
        outcome=outcome,
        seed=seed,
    ) | {"branch_name": branch_name}


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_17"
    ensure_dir(output_dir)

    frame = assign_component_branch(load_analysis_frame())

    global_scenarios = build_global_branch_mask_scenarios(frame)
    global_fold_tables: list[pd.DataFrame] = []
    global_summary_tables: list[pd.DataFrame] = []
    for scenario_name, scenario_frame in global_scenarios.items():
        folds, summary = run_feature_suite(
            frame=scenario_frame,
            scenario_name=scenario_name,
            feature_sets=GLOBAL_FEATURE_SETS,
        )
        global_fold_tables.append(folds)
        global_summary_tables.append(summary)
    global_folds = pd.concat(global_fold_tables, ignore_index=True)
    global_summary = pd.concat(global_summary_tables, ignore_index=True)
    global_gain = compute_token_gain(global_summary, scenario_group="global_mask")

    branch_feature_sets = {
        "branch_components": GLOBAL_FEATURE_SETS["components_full"],
        "branch_components_plus_full_token": GLOBAL_FEATURE_SETS["components_plus_full_token"],
    }
    branch_subsets = {
        branch_name: frame.loc[frame["component_branch"] == branch_name].copy()
        for branch_name in ["tI10_branch", "background"]
    }
    branch_fold_tables: list[pd.DataFrame] = []
    branch_summary_tables: list[pd.DataFrame] = []
    for branch_name, branch_frame in branch_subsets.items():
        folds, summary = run_feature_suite(
            frame=branch_frame,
            scenario_name=f"branch__{branch_name}",
            feature_sets=branch_feature_sets,
        )
        branch_fold_tables.append(folds)
        branch_summary_tables.append(summary)
    branch_folds = pd.concat(branch_fold_tables, ignore_index=True)
    branch_summary = pd.concat(branch_summary_tables, ignore_index=True)
    branch_gain = compute_token_gain(branch_summary, scenario_group="within_branch")

    tI10_token_rates = build_branch_token_rate_table(frame, "tI10_branch")
    background_token_rates = build_branch_token_rate_table(frame, "background")
    robust_tI10_tokens = tI10_token_rates.loc[
        tI10_token_rates["n_materials"] >= BRANCH_MIN_TOKEN_N
    ].copy()

    bootstrap_rows: list[dict] = []
    for token_label in robust_tI10_tokens["frequent_prototype_token_bucket"]:
        for outcome in OUTCOMES:
            bootstrap_rows.append(
                bootstrap_branch_token_vs_rest(
                    frame=frame,
                    branch_name="tI10_branch",
                    token_label=token_label,
                    outcome=outcome,
                    seed=101 + len(bootstrap_rows),
                )
            )
    tI10_bootstrap = pd.DataFrame(bootstrap_rows)

    global_lookup = global_gain.set_index(["scenario", "outcome"])
    branch_lookup = branch_gain.set_index(["scenario", "outcome"])
    summary = {
        "singleton_high_risk_subset": {
            "n_materials": int(len(frame)),
            "global_branch_masking": {
                outcome: {
                    "baseline_full_token_balanced_accuracy_gain": float(
                        global_lookup.loc[("baseline", outcome), "full_token_balanced_accuracy_gain"]
                    ),
                    "mask_tI10_branch_tokens_gain": float(
                        global_lookup.loc[("mask_tI10_branch_tokens", outcome), "full_token_balanced_accuracy_gain"]
                    ),
                    "mask_ABC_oP12_62_branch_tokens_gain": float(
                        global_lookup.loc[("mask_ABC_oP12_62_branch_tokens", outcome), "full_token_balanced_accuracy_gain"]
                    ),
                    "mask_c_c_c_residual_branch_tokens_gain": float(
                        global_lookup.loc[("mask_c_c_c_residual_branch_tokens", outcome), "full_token_balanced_accuracy_gain"]
                    ),
                    "mask_background_tokens_gain": float(
                        global_lookup.loc[("mask_background_tokens", outcome), "full_token_balanced_accuracy_gain"]
                    ),
                    "mask_nonbackground_tokens_gain": float(
                        global_lookup.loc[("mask_nonbackground_tokens", outcome), "full_token_balanced_accuracy_gain"]
                    ),
                }
                for outcome in OUTCOMES
            },
            "within_branch_gain": {
                outcome: {
                    "tI10_branch_full_token_gain": float(
                        branch_lookup.loc[("branch__tI10_branch", outcome), "full_token_balanced_accuracy_gain"]
                    ),
                    "background_full_token_gain": float(
                        branch_lookup.loc[("branch__background", outcome), "full_token_balanced_accuracy_gain"]
                    ),
                }
                for outcome in OUTCOMES
            },
            "top_tI10_tokens": robust_tI10_tokens.to_dict(orient="records"),
        }
    }

    global_folds.to_csv(output_dir / "global_mask_cv_fold_metrics.csv", index=False)
    global_summary.to_csv(output_dir / "global_mask_cv_summary.csv", index=False)
    global_gain.to_csv(output_dir / "global_mask_token_gain_summary.csv", index=False)
    branch_folds.to_csv(output_dir / "branch_cv_fold_metrics.csv", index=False)
    branch_summary.to_csv(output_dir / "branch_cv_summary.csv", index=False)
    branch_gain.to_csv(output_dir / "branch_token_gain_summary.csv", index=False)
    tI10_token_rates.to_csv(output_dir / "tI10_branch_token_rates.csv", index=False)
    background_token_rates.to_csv(output_dir / "background_branch_token_rates.csv", index=False)
    robust_tI10_tokens.to_csv(output_dir / "robust_tI10_branch_token_rates.csv", index=False)
    tI10_bootstrap.to_csv(output_dir / "tI10_branch_token_bootstrap_differences.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from numpy.linalg import LinAlgError

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


OUTCOME_COLUMNS = [
    "collective_failure",
    "collective_false_negative",
    "collective_false_positive",
]
BOOTSTRAP_RESAMPLES = 2000


def assign_structure_match_tier(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                ~frame["exact_formula_in_reference"],
                frame["exact_formula_in_reference"] & ~frame["exact_formula_nsites_in_reference"],
                frame["exact_formula_nsites_in_reference"] & ~frame["exact_formula_nsites_volume_in_reference_range"],
                frame["exact_formula_nsites_volume_in_reference_range"],
            ],
            [
                "no_exact_formula",
                "exact_formula_only",
                "exact_formula_nsites_out_of_range",
                "exact_formula_nsites_in_range",
            ],
            default="other",
        ),
        index=frame.index,
    )


def build_rate_table(frame: pd.DataFrame, subset_label: str) -> pd.DataFrame:
    grouped = (
        frame.groupby("structure_match_tier", as_index=False)
        .agg(
            n_materials=("collective_failure", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
            collective_false_positive_rate=("collective_false_positive", "mean"),
            unique_prototype_rate=("unique_prototype", "mean"),
            exact_formula_in_mptrj_rate=("exact_formula_in_mptrj", "mean"),
            exact_formula_nsites_hit_rate=("exact_formula_nsites_in_reference", "mean"),
            exact_formula_nsites_in_range_rate=("exact_formula_nsites_volume_in_reference_range", "mean"),
            median_min_abs_nsites_diff_exact_formula=("min_abs_nsites_diff_exact_formula", "median"),
            median_min_volume_range_distance_exact_formula=("min_volume_range_distance_exact_formula", "median"),
            median_min_volume_range_distance_exact_formula_nsites=(
                "min_volume_range_distance_exact_formula_nsites",
                "median",
            ),
        )
        .sort_values("collective_failure_rate", ascending=False)
        .reset_index(drop=True)
    )
    grouped.insert(0, "subset", subset_label)
    return grouped


def build_distance_summary(frame: pd.DataFrame, subset_label: str) -> pd.DataFrame:
    labeled = frame.loc[frame["collective_failure"] | frame["collective_success"]].copy()
    labeled["label"] = np.where(labeled["collective_failure"], "failure", "success")
    summary = (
        labeled.groupby("label", as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            exact_formula_in_reference_rate=("exact_formula_in_reference", "mean"),
            exact_formula_nsites_in_reference_rate=("exact_formula_nsites_in_reference", "mean"),
            exact_formula_nsites_in_range_rate=("exact_formula_nsites_volume_in_reference_range", "mean"),
            median_exact_formula_signature_count=("exact_formula_signature_count", "median"),
            median_exact_formula_total_structure_count=("exact_formula_total_structure_count", "median"),
            median_exact_formula_nsites_structure_count=("exact_formula_nsites_structure_count", "median"),
            median_min_abs_nsites_diff_exact_formula=("min_abs_nsites_diff_exact_formula", "median"),
            median_min_volume_range_distance_exact_formula=("min_volume_range_distance_exact_formula", "median"),
            median_min_volume_range_distance_exact_formula_nsites=(
                "min_volume_range_distance_exact_formula_nsites",
                "median",
            ),
        )
        .sort_values("label")
        .reset_index(drop=True)
    )
    summary.insert(0, "subset", subset_label)
    return summary


def fit_logit(formula: str, frame: pd.DataFrame, subset_label: str, outcome: str, model_name: str) -> pd.DataFrame:
    try:
        model = smf.logit(formula, data=frame).fit(disp=False)
        table = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
        table["odds_ratio"] = np.exp(table["Coef."])
        table["odds_ratio_ci_low"] = np.exp(table["[0.025"])
        table["odds_ratio_ci_high"] = np.exp(table["0.975]"])
        table["status"] = "ok"
    except (LinAlgError, ValueError) as exc:
        table = pd.DataFrame(
            [
                {
                    "term": "__model__",
                    "Coef.": np.nan,
                    "Std.Err.": np.nan,
                    "z": np.nan,
                    "P>|z|": np.nan,
                    "[0.025": np.nan,
                    "0.975]": np.nan,
                    "odds_ratio": np.nan,
                    "odds_ratio_ci_low": np.nan,
                    "odds_ratio_ci_high": np.nan,
                    "status": f"failed: {type(exc).__name__}: {exc}",
                }
            ]
        )
    table["subset"] = subset_label
    table["outcome"] = outcome
    table["model_name"] = model_name
    table["formula"] = formula
    return table


def relative_risk_to_best_supported(rate_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for subset_label, subset in rate_table.groupby("subset", sort=False):
        baseline_candidates = subset.loc[subset["structure_match_tier"] == "exact_formula_nsites_in_range"]
        if baseline_candidates.empty:
            continue
        baseline = baseline_candidates.iloc[0]
        for row in subset.itertuples(index=False):
            rows.append(
                {
                    "subset": subset_label,
                    "structure_match_tier": row.structure_match_tier,
                    "n_materials": int(row.n_materials),
                    "collective_failure_rate": float(row.collective_failure_rate),
                    "failure_rate_vs_best_supported": float(row.collective_failure_rate / baseline.collective_failure_rate)
                    if baseline.collective_failure_rate > 0
                    else float("nan"),
                    "collective_false_negative_rate": float(row.collective_false_negative_rate),
                    "false_negative_rate_vs_best_supported": float(
                        row.collective_false_negative_rate / baseline.collective_false_negative_rate
                    )
                    if baseline.collective_false_negative_rate > 0
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def to_int_bools(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    converted = frame.copy()
    for column in columns:
        converted[column] = converted[column].astype(int)
    return converted


def bootstrap_rate_difference(
    frame: pd.DataFrame,
    subset_label: str,
    tier_a: str,
    tier_b: str,
    outcome: str,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict:
    tier_a_values = frame.loc[frame["structure_match_tier"] == tier_a, outcome].to_numpy(dtype=float)
    tier_b_values = frame.loc[frame["structure_match_tier"] == tier_b, outcome].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    observed = float(tier_a_values.mean() - tier_b_values.mean())
    bootstrap = np.empty(n_resamples, dtype=float)
    for idx in range(n_resamples):
        sample_a = rng.choice(tier_a_values, size=len(tier_a_values), replace=True)
        sample_b = rng.choice(tier_b_values, size=len(tier_b_values), replace=True)
        bootstrap[idx] = sample_a.mean() - sample_b.mean()

    return {
        "subset": subset_label,
        "tier_a": tier_a,
        "tier_b": tier_b,
        "outcome": outcome,
        "n_tier_a": int(len(tier_a_values)),
        "n_tier_b": int(len(tier_b_values)),
        "observed_rate_diff": observed,
        "q025": float(np.quantile(bootstrap, 0.025)),
        "q975": float(np.quantile(bootstrap, 0.975)),
        "fraction_diff_gt_zero": float((bootstrap > 0).mean()),
    }


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_10"
    ensure_dir(output_dir)

    features = pd.read_csv(RESULTS_DIR / "tables" / "analysis_04" / "collective_failure_features.csv")
    structure_proxy = pd.read_parquet(
        PROCESSED_DIR / "wbm_mptrj_mp2022_union_material_structure_overlap_proxy.parquet"
    )
    frame = features.merge(
        structure_proxy[
            [
                "material_id",
                "exact_formula_in_reference",
                "exact_formula_signature_count",
                "exact_formula_total_structure_count",
                "exact_formula_total_material_count",
                "exact_formula_nsites_in_reference",
                "exact_formula_nsites_structure_count",
                "exact_formula_nsites_material_count",
                "min_abs_nsites_diff_exact_formula",
                "nearest_nsites_exact_formula",
                "min_abs_volume_per_atom_diff_exact_formula_mean",
                "min_volume_range_distance_exact_formula",
                "exact_formula_volume_in_reference_range",
                "min_abs_volume_per_atom_diff_exact_formula_nsites_mean",
                "min_volume_range_distance_exact_formula_nsites",
                "exact_formula_nsites_volume_in_reference_range",
            ]
        ],
        on="material_id",
        how="left",
        validate="one_to_one",
    )

    frame["structure_match_tier"] = assign_structure_match_tier(frame)

    analysis_frame = frame.loc[frame["collective_failure"] | frame["collective_success"]].copy()
    exact_formula_subset = analysis_frame.loc[analysis_frame["exact_formula_in_reference"] == 1].copy()
    exact_formula_nsites_subset = exact_formula_subset.loc[
        exact_formula_subset["exact_formula_nsites_in_reference"] == 1
    ].copy()
    high_risk_subset = analysis_frame.loc[
        (analysis_frame["unique_prototype"] == 0) & (analysis_frame["exact_formula_in_mptrj"] == 1)
    ].copy()
    high_risk_nsites_subset = high_risk_subset.loc[high_risk_subset["exact_formula_nsites_in_reference"] == 1].copy()

    bool_columns = [
        "collective_failure",
        "collective_success",
        "collective_false_negative",
        "collective_false_positive",
        "unique_prototype",
        "exact_formula_in_mptrj",
        "exact_formula_in_reference",
        "exact_formula_nsites_in_reference",
        "exact_formula_volume_in_reference_range",
        "exact_formula_nsites_volume_in_reference_range",
    ]
    exact_formula_subset_for_logit = to_int_bools(exact_formula_subset, bool_columns)
    exact_formula_nsites_subset_for_logit = to_int_bools(exact_formula_nsites_subset, bool_columns)
    high_risk_subset_for_logit = to_int_bools(high_risk_subset, bool_columns)
    high_risk_nsites_subset_for_logit = to_int_bools(high_risk_nsites_subset, bool_columns)

    rate_tables = [
        build_rate_table(analysis_frame, "overall_failure_success"),
        build_rate_table(exact_formula_subset, "exact_formula_reference_subset"),
        build_rate_table(high_risk_subset, "high_risk_mptrj_overlap_subset"),
    ]
    rate_table = pd.concat(rate_tables, ignore_index=True)
    risk_summary = relative_risk_to_best_supported(rate_table)

    distance_summary = pd.concat(
        [
            build_distance_summary(analysis_frame, "overall_failure_success"),
            build_distance_summary(exact_formula_subset, "exact_formula_reference_subset"),
            build_distance_summary(high_risk_subset, "high_risk_mptrj_overlap_subset"),
        ],
        ignore_index=True,
    )

    model_specs = [
        (
            "exact_formula_reference_step1",
            "exact_formula_reference_subset",
            exact_formula_subset_for_logit,
            "{outcome} ~ unique_prototype + exact_formula_in_mptrj + exact_formula_nsites_in_reference + np.log1p(exact_formula_signature_count)",
        ),
        (
            "exact_formula_nsites_step2",
            "exact_formula_nsites_subset",
            exact_formula_nsites_subset_for_logit,
            "{outcome} ~ unique_prototype + exact_formula_in_mptrj + exact_formula_nsites_volume_in_reference_range + np.log1p(exact_formula_nsites_structure_count)",
        ),
        (
            "high_risk_overlap_step1",
            "high_risk_mptrj_overlap_subset",
            high_risk_subset_for_logit,
            "{outcome} ~ exact_formula_nsites_in_reference + np.log1p(exact_formula_signature_count)",
        ),
        (
            "high_risk_overlap_step2",
            "high_risk_mptrj_overlap_nsites_subset",
            high_risk_nsites_subset_for_logit,
            "{outcome} ~ exact_formula_nsites_volume_in_reference_range + np.log1p(exact_formula_nsites_structure_count)",
        ),
    ]
    logit_tables: list[pd.DataFrame] = []
    for model_name, subset_label, subset_frame, formula_template in model_specs:
        for outcome in OUTCOME_COLUMNS:
            logit_tables.append(
                fit_logit(
                    formula=formula_template.format(outcome=outcome),
                    frame=subset_frame,
                    subset_label=subset_label,
                    outcome=outcome,
                    model_name=model_name,
                )
            )
    logit_summary = pd.concat(logit_tables, ignore_index=True)

    bootstrap_tasks = [
        ("exact_formula_reference_subset", exact_formula_subset, "exact_formula_nsites_in_range", "exact_formula_nsites_out_of_range"),
        ("exact_formula_reference_subset", exact_formula_subset, "exact_formula_nsites_in_range", "exact_formula_only"),
        ("high_risk_mptrj_overlap_subset", high_risk_subset, "exact_formula_nsites_in_range", "exact_formula_nsites_out_of_range"),
        ("high_risk_mptrj_overlap_subset", high_risk_subset, "exact_formula_nsites_in_range", "exact_formula_only"),
    ]
    bootstrap_rows: list[dict] = []
    for subset_label, subset_frame, tier_a, tier_b in bootstrap_tasks:
        for outcome in ["collective_failure", "collective_false_negative"]:
            bootstrap_rows.append(
                bootstrap_rate_difference(
                    frame=subset_frame,
                    subset_label=subset_label,
                    tier_a=tier_a,
                    tier_b=tier_b,
                    outcome=outcome,
                    seed=17 + len(bootstrap_rows),
                )
            )
    bootstrap_summary = pd.DataFrame(bootstrap_rows)

    best_supported_high_risk = rate_table.loc[
        (rate_table["subset"] == "high_risk_mptrj_overlap_subset")
        & (rate_table["structure_match_tier"] == "exact_formula_nsites_in_range")
    ].iloc[0]
    highest_risk_high_risk = rate_table.loc[rate_table["subset"] == "high_risk_mptrj_overlap_subset"].sort_values(
        "collective_failure_rate",
        ascending=False,
    ).iloc[0]

    summary = {
        "overall": {
            "n_materials": int(len(analysis_frame)),
            "exact_formula_reference_fraction": float(analysis_frame["exact_formula_in_reference"].mean()),
            "exact_formula_nsites_fraction": float(analysis_frame["exact_formula_nsites_in_reference"].mean()),
            "exact_formula_nsites_in_range_fraction": float(
                analysis_frame["exact_formula_nsites_volume_in_reference_range"].mean()
            ),
        },
        "high_risk_subset": {
            "n_materials": int(len(high_risk_subset)),
            "highest_failure_tier": highest_risk_high_risk[
                [
                    "structure_match_tier",
                    "n_materials",
                    "collective_failure_rate",
                    "collective_false_negative_rate",
                ]
            ].to_dict(),
            "best_supported_tier": best_supported_high_risk[
                [
                    "structure_match_tier",
                    "n_materials",
                    "collective_failure_rate",
                    "collective_false_negative_rate",
                ]
            ].to_dict(),
            "bootstrap_in_range_vs_out_of_range_failure": bootstrap_summary.loc[
                (bootstrap_summary["subset"] == "high_risk_mptrj_overlap_subset")
                & (bootstrap_summary["tier_b"] == "exact_formula_nsites_out_of_range")
                & (bootstrap_summary["outcome"] == "collective_failure")
            ]
            .iloc[0][["observed_rate_diff", "q025", "q975", "fraction_diff_gt_zero"]]
            .to_dict(),
            "bootstrap_in_range_vs_out_of_range_false_negative": bootstrap_summary.loc[
                (bootstrap_summary["subset"] == "high_risk_mptrj_overlap_subset")
                & (bootstrap_summary["tier_b"] == "exact_formula_nsites_out_of_range")
                & (bootstrap_summary["outcome"] == "collective_false_negative")
            ]
            .iloc[0][["observed_rate_diff", "q025", "q975", "fraction_diff_gt_zero"]]
            .to_dict(),
        },
    }

    rate_table.to_csv(output_dir / "structure_match_tier_rates.csv", index=False)
    risk_summary.to_csv(output_dir / "structure_match_tier_risk_summary.csv", index=False)
    distance_summary.to_csv(output_dir / "structure_overlap_distance_summary.csv", index=False)
    logit_summary.to_csv(output_dir / "structure_overlap_logit_summary.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "structure_match_tier_bootstrap_differences.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

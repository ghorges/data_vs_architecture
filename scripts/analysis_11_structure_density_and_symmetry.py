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
REFERENCE_TIER = "minority_multi_signature"


def build_density_rate_table(frame: pd.DataFrame, subset_label: str) -> pd.DataFrame:
    grouped = (
        frame.groupby("exact_formula_nsites_density_tier", as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
            collective_false_positive_rate=("collective_false_positive", "mean"),
            unique_prototype_rate=("unique_prototype", "mean"),
            median_exact_formula_signature_count=("exact_formula_signature_count", "median"),
            median_exact_formula_nsites_structure_count=("exact_formula_nsites_structure_count", "median"),
            median_exact_formula_nsites_structure_share=("exact_formula_nsites_structure_share", "median"),
            median_anonymous_formula_signature_count=("anonymous_formula_signature_count", "median"),
            median_anonymous_nsites_structure_count=("anonymous_nsites_structure_count", "median"),
            median_anonymous_nsites_structure_share=("anonymous_nsites_structure_share", "median"),
        )
        .sort_values("collective_failure_rate", ascending=False)
        .reset_index(drop=True)
    )
    grouped.insert(0, "subset", subset_label)
    return grouped


def build_singleton_anonymous_summary(frame: pd.DataFrame, subset_label: str) -> pd.DataFrame:
    singleton = frame.loc[frame["exact_formula_nsites_density_tier"] == "singleton_formula_signature"].copy()
    singleton["label"] = np.where(singleton["collective_failure"] == 1, "failure", "success")
    summary = (
        singleton.groupby("label", as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            median_anonymous_formula_signature_count=("anonymous_formula_signature_count", "median"),
            median_anonymous_formula_total_structure_count=("anonymous_formula_total_structure_count", "median"),
            median_anonymous_nsites_structure_count=("anonymous_nsites_structure_count", "median"),
            median_anonymous_nsites_structure_share=("anonymous_nsites_structure_share", "median"),
        )
        .sort_values("label")
        .reset_index(drop=True)
    )
    summary.insert(0, "subset", subset_label)
    return summary


def build_singleton_crystal_system_rates(frame: pd.DataFrame, subset_label: str) -> pd.DataFrame:
    singleton = frame.loc[frame["exact_formula_nsites_density_tier"] == "singleton_formula_signature"].copy()
    grouped = (
        singleton.groupby("crystal_system", as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
            collective_false_positive_rate=("collective_false_positive", "mean"),
        )
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )
    grouped.insert(0, "subset", subset_label)
    return grouped


def bootstrap_rate_difference(
    frame: pd.DataFrame,
    subset_label: str,
    tier_a: str,
    tier_b: str,
    outcome: str,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict:
    tier_a_values = frame.loc[frame["exact_formula_nsites_density_tier"] == tier_a, outcome].to_numpy(dtype=float)
    tier_b_values = frame.loc[frame["exact_formula_nsites_density_tier"] == tier_b, outcome].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    if len(tier_a_values) == 0 or len(tier_b_values) == 0:
        return {
            "subset": subset_label,
            "tier_a": tier_a,
            "tier_b": tier_b,
            "outcome": outcome,
            "n_tier_a": int(len(tier_a_values)),
            "n_tier_b": int(len(tier_b_values)),
            "observed_rate_diff": np.nan,
            "q025": np.nan,
            "q975": np.nan,
            "fraction_diff_gt_zero": np.nan,
        }

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


def to_int_bools(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    converted = frame.copy()
    for column in columns:
        converted[column] = converted[column].astype(int)
    return converted


def fit_logit(formula: str, frame: pd.DataFrame, subset_label: str, outcome: str, model_name: str) -> pd.DataFrame:
    try:
        model = smf.logit(formula, data=frame).fit(disp=False)
        table = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
        table["odds_ratio"] = np.exp(table["Coef."])
        table["odds_ratio_ci_low"] = np.exp(table["[0.025"])
        table["odds_ratio_ci_high"] = np.exp(table["0.975]"])
        table["status"] = "ok"
    except (LinAlgError, ValueError, Exception) as exc:
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


def lookup_rate(rate_table: pd.DataFrame, subset_label: str, tier: str, column: str) -> float:
    match = rate_table.loc[
        (rate_table["subset"] == subset_label) & (rate_table["exact_formula_nsites_density_tier"] == tier),
        column,
    ]
    if match.empty:
        return float("nan")
    return float(match.iloc[0])


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_11"
    ensure_dir(output_dir)

    features = pd.read_csv(
        RESULTS_DIR / "tables" / "analysis_04" / "collective_failure_features.csv",
        usecols=[
            "material_id",
            "collective_failure",
            "collective_success",
            "collective_false_negative",
            "collective_false_positive",
            "unique_prototype",
            "exact_formula_in_mptrj",
            "crystal_system",
        ],
    )
    density_proxy = pd.read_parquet(
        PROCESSED_DIR / "wbm_mptrj_mp2022_union_material_structure_density_proxy.parquet"
    )
    frame = features.merge(
        density_proxy[
            [
                "material_id",
                "exact_formula_nsites_in_reference",
                "exact_formula_signature_count",
                "exact_formula_nsites_structure_count",
                "exact_formula_nsites_structure_share",
                "exact_formula_is_singleton_signature",
                "exact_formula_nsites_density_tier",
                "anonymous_formula_signature_count",
                "anonymous_formula_total_structure_count",
                "anonymous_nsites_structure_count",
                "anonymous_nsites_structure_share",
            ]
        ],
        on="material_id",
        how="left",
        validate="one_to_one",
    )

    analysis_frame = frame.loc[frame["collective_failure"] | frame["collective_success"]].copy()
    exact_formula_nsites_subset = analysis_frame.loc[
        analysis_frame["exact_formula_nsites_in_reference"] == 1
    ].copy()
    exact_formula_nsites_subset = exact_formula_nsites_subset.loc[
        exact_formula_nsites_subset["exact_formula_nsites_density_tier"] != "no_match"
    ].copy()
    high_risk_nsites_subset = exact_formula_nsites_subset.loc[
        (exact_formula_nsites_subset["exact_formula_in_mptrj"] == 1)
        & (exact_formula_nsites_subset["unique_prototype"] == 0)
    ].copy()

    bool_columns = [
        "collective_failure",
        "collective_success",
        "collective_false_negative",
        "collective_false_positive",
        "unique_prototype",
        "exact_formula_in_mptrj",
        "exact_formula_nsites_in_reference",
        "exact_formula_is_singleton_signature",
    ]
    exact_formula_nsites_subset_for_logit = to_int_bools(exact_formula_nsites_subset, bool_columns)
    high_risk_nsites_subset_for_logit = to_int_bools(high_risk_nsites_subset, bool_columns)
    singleton_high_risk_subset_for_logit = to_int_bools(
        high_risk_nsites_subset.loc[
            high_risk_nsites_subset["exact_formula_nsites_density_tier"] == "singleton_formula_signature"
        ].copy(),
        bool_columns,
    )

    density_rates = pd.concat(
        [
            build_density_rate_table(exact_formula_nsites_subset, "exact_formula_nsites_subset"),
            build_density_rate_table(high_risk_nsites_subset, "high_risk_nsites_subset"),
        ],
        ignore_index=True,
    )
    singleton_anonymous_summary = pd.concat(
        [
            build_singleton_anonymous_summary(exact_formula_nsites_subset, "exact_formula_nsites_subset"),
            build_singleton_anonymous_summary(high_risk_nsites_subset, "high_risk_nsites_subset"),
        ],
        ignore_index=True,
    )
    singleton_crystal_rates = pd.concat(
        [
            build_singleton_crystal_system_rates(exact_formula_nsites_subset, "exact_formula_nsites_subset"),
            build_singleton_crystal_system_rates(high_risk_nsites_subset, "high_risk_nsites_subset"),
        ],
        ignore_index=True,
    )

    bootstrap_tasks = [
        ("exact_formula_nsites_subset", exact_formula_nsites_subset, "singleton_formula_signature", REFERENCE_TIER),
        ("exact_formula_nsites_subset", exact_formula_nsites_subset, "dominant_multi_signature", REFERENCE_TIER),
        ("high_risk_nsites_subset", high_risk_nsites_subset, "singleton_formula_signature", REFERENCE_TIER),
        ("high_risk_nsites_subset", high_risk_nsites_subset, "dominant_multi_signature", REFERENCE_TIER),
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
                    seed=37 + len(bootstrap_rows),
                )
            )
    bootstrap_summary = pd.DataFrame(bootstrap_rows)

    model_specs = [
        (
            "exact_formula_nsites_density",
            "exact_formula_nsites_subset",
            exact_formula_nsites_subset_for_logit,
            (
                "{outcome} ~ unique_prototype + "
                "C(exact_formula_nsites_density_tier) + "
                "np.log1p(anonymous_nsites_structure_count)"
            ),
        ),
        (
            "high_risk_nsites_density",
            "high_risk_nsites_subset",
            high_risk_nsites_subset_for_logit,
            (
                "{outcome} ~ "
                "C(exact_formula_nsites_density_tier) + "
                "np.log1p(anonymous_nsites_structure_count)"
            ),
        ),
        (
            "singleton_high_risk_crystal_system",
            "singleton_high_risk_subset",
            singleton_high_risk_subset_for_logit,
            "{outcome} ~ C(crystal_system)",
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

    robust_singleton_high_risk = singleton_crystal_rates.loc[
        (singleton_crystal_rates["subset"] == "high_risk_nsites_subset")
        & (singleton_crystal_rates["n_materials"] >= 50)
    ].sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])

    summary = {
        "exact_formula_nsites_subset": {
            "n_materials": int(len(exact_formula_nsites_subset)),
            "singleton_failure_rate": lookup_rate(
                density_rates,
                "exact_formula_nsites_subset",
                "singleton_formula_signature",
                "collective_failure_rate",
            ),
            "dominant_failure_rate": lookup_rate(
                density_rates,
                "exact_formula_nsites_subset",
                "dominant_multi_signature",
                "collective_failure_rate",
            ),
            "minority_failure_rate": lookup_rate(
                density_rates,
                "exact_formula_nsites_subset",
                "minority_multi_signature",
                "collective_failure_rate",
            ),
            "singleton_false_negative_rate": lookup_rate(
                density_rates,
                "exact_formula_nsites_subset",
                "singleton_formula_signature",
                "collective_false_negative_rate",
            ),
        },
        "high_risk_nsites_subset": {
            "n_materials": int(len(high_risk_nsites_subset)),
            "singleton_failure_rate": lookup_rate(
                density_rates,
                "high_risk_nsites_subset",
                "singleton_formula_signature",
                "collective_failure_rate",
            ),
            "dominant_failure_rate": lookup_rate(
                density_rates,
                "high_risk_nsites_subset",
                "dominant_multi_signature",
                "collective_failure_rate",
            ),
            "minority_failure_rate": lookup_rate(
                density_rates,
                "high_risk_nsites_subset",
                "minority_multi_signature",
                "collective_failure_rate",
            ),
            "singleton_false_negative_rate": lookup_rate(
                density_rates,
                "high_risk_nsites_subset",
                "singleton_formula_signature",
                "collective_false_negative_rate",
            ),
            "bootstrap_singleton_vs_minority_failure": bootstrap_summary.loc[
                (bootstrap_summary["subset"] == "high_risk_nsites_subset")
                & (bootstrap_summary["tier_a"] == "singleton_formula_signature")
                & (bootstrap_summary["outcome"] == "collective_failure")
            ]
            .iloc[0][["observed_rate_diff", "q025", "q975", "fraction_diff_gt_zero"]]
            .to_dict(),
            "bootstrap_singleton_vs_minority_false_negative": bootstrap_summary.loc[
                (bootstrap_summary["subset"] == "high_risk_nsites_subset")
                & (bootstrap_summary["tier_a"] == "singleton_formula_signature")
                & (bootstrap_summary["outcome"] == "collective_false_negative")
            ]
            .iloc[0][["observed_rate_diff", "q025", "q975", "fraction_diff_gt_zero"]]
            .to_dict(),
            "top_crystal_system_singleton_high_risk_n_ge_50": (
                robust_singleton_high_risk.iloc[0][
                    ["crystal_system", "n_materials", "collective_failure_rate", "collective_false_negative_rate"]
                ].to_dict()
                if not robust_singleton_high_risk.empty
                else {}
            ),
        },
    }

    density_rates.to_csv(output_dir / "density_tier_rates.csv", index=False)
    singleton_anonymous_summary.to_csv(output_dir / "singleton_anonymous_support_summary.csv", index=False)
    singleton_crystal_rates.to_csv(output_dir / "singleton_crystal_system_rates.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "density_tier_bootstrap_differences.csv", index=False)
    logit_summary.to_csv(output_dir / "density_tier_logit_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

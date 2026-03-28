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
RARITY_LABELS = ["Q1_rarest", "Q2", "Q3", "Q4_most_common"]
BOOTSTRAP_RESAMPLES = 2000


def assign_rank_quartile(series: pd.Series) -> pd.Series:
    ranks = series.rank(method="first")
    return pd.qcut(ranks, 4, labels=RARITY_LABELS)


def build_rarity_rate_table(frame: pd.DataFrame, subset_label: str, rarity_column: str, count_column: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(rarity_column, as_index=False, observed=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
            collective_false_positive_rate=("collective_false_positive", "mean"),
            median_count=(count_column, "median"),
            median_spacegroup_count=("spacegroup_count_global", "median"),
        )
        .sort_values(rarity_column)
        .reset_index(drop=True)
    )
    grouped.insert(0, "subset", subset_label)
    grouped.insert(1, "rarity_axis", rarity_column)
    return grouped


def bootstrap_rate_difference(
    frame: pd.DataFrame,
    subset_label: str,
    rarity_column: str,
    rare_label: str,
    common_label: str,
    outcome: str,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict:
    rare_values = frame.loc[frame[rarity_column] == rare_label, outcome].to_numpy(dtype=float)
    common_values = frame.loc[frame[rarity_column] == common_label, outcome].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    observed = float(rare_values.mean() - common_values.mean())
    bootstrap = np.empty(n_resamples, dtype=float)
    for idx in range(n_resamples):
        sample_rare = rng.choice(rare_values, size=len(rare_values), replace=True)
        sample_common = rng.choice(common_values, size=len(common_values), replace=True)
        bootstrap[idx] = sample_rare.mean() - sample_common.mean()

    return {
        "subset": subset_label,
        "rarity_axis": rarity_column,
        "rare_label": rare_label,
        "common_label": common_label,
        "outcome": outcome,
        "n_rare": int(len(rare_values)),
        "n_common": int(len(common_values)),
        "observed_rate_diff": observed,
        "q025": float(np.quantile(bootstrap, 0.025)),
        "q975": float(np.quantile(bootstrap, 0.975)),
        "fraction_diff_gt_zero": float((bootstrap > 0).mean()),
    }


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


def to_int_bools(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    converted = frame.copy()
    for column in columns:
        converted[column] = converted[column].astype(int)
    return converted


def lookup_rate(rate_table: pd.DataFrame, subset: str, rarity_axis: str, quartile: str, column: str) -> float:
    match = rate_table.loc[
        (rate_table["subset"] == subset)
        & (rate_table["rarity_axis"] == rarity_axis)
        & (rate_table[rarity_axis] == quartile),
        column,
    ]
    if match.empty:
        return float("nan")
    return float(match.iloc[0])


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_12"
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
            "wyckoff_spglib",
            "spacegroup_number",
        ],
    )
    density = pd.read_parquet(
        PROCESSED_DIR / "wbm_mptrj_mp2022_union_material_structure_density_proxy.parquet",
        columns=["material_id", "exact_formula_nsites_density_tier"],
    )
    rarity = pd.read_parquet(
        PROCESSED_DIR / "wbm_mptrj_mp2022_union_material_prototype_spacegroup_rarity_proxy.parquet"
    )
    frame = (
        features.merge(density, on="material_id", how="left", validate="one_to_one")
        .merge(
            rarity[
                [
                    "material_id",
                    "prototype_token",
                    "prototype_token_count_global",
                    "prototype_token_count_in_density_tier",
                    "prototype_token_share_in_density_tier",
                    "spacegroup_count_global",
                    "spacegroup_count_in_density_tier",
                    "spacegroup_share_in_density_tier",
                ]
            ],
            on="material_id",
            how="left",
            validate="one_to_one",
        )
    )

    analysis_frame = frame.loc[frame["collective_failure"] | frame["collective_success"]].copy()
    singleton_exact_subset = analysis_frame.loc[
        analysis_frame["exact_formula_nsites_density_tier"] == "singleton_formula_signature"
    ].copy()
    singleton_high_risk_subset = singleton_exact_subset.loc[
        (singleton_exact_subset["exact_formula_in_mptrj"]) & (~singleton_exact_subset["unique_prototype"])
    ].copy()

    for subset in [singleton_exact_subset, singleton_high_risk_subset]:
        subset["prototype_density_rarity_quartile"] = assign_rank_quartile(subset["prototype_token_count_in_density_tier"])
        subset["spacegroup_density_rarity_quartile"] = assign_rank_quartile(subset["spacegroup_count_in_density_tier"])

    rarity_rates = pd.concat(
        [
            build_rarity_rate_table(
                singleton_exact_subset,
                "singleton_exact_subset",
                "prototype_density_rarity_quartile",
                "prototype_token_count_in_density_tier",
            ),
            build_rarity_rate_table(
                singleton_high_risk_subset,
                "singleton_high_risk_subset",
                "prototype_density_rarity_quartile",
                "prototype_token_count_in_density_tier",
            ),
            build_rarity_rate_table(
                singleton_exact_subset,
                "singleton_exact_subset",
                "spacegroup_density_rarity_quartile",
                "spacegroup_count_in_density_tier",
            ),
            build_rarity_rate_table(
                singleton_high_risk_subset,
                "singleton_high_risk_subset",
                "spacegroup_density_rarity_quartile",
                "spacegroup_count_in_density_tier",
            ),
        ],
        ignore_index=True,
    )

    bootstrap_rows: list[dict] = []
    bootstrap_specs = [
        ("singleton_exact_subset", singleton_exact_subset, "prototype_density_rarity_quartile"),
        ("singleton_high_risk_subset", singleton_high_risk_subset, "prototype_density_rarity_quartile"),
        ("singleton_exact_subset", singleton_exact_subset, "spacegroup_density_rarity_quartile"),
        ("singleton_high_risk_subset", singleton_high_risk_subset, "spacegroup_density_rarity_quartile"),
    ]
    for subset_label, subset_frame, rarity_column in bootstrap_specs:
        for outcome in ["collective_failure", "collective_false_negative"]:
            bootstrap_rows.append(
                bootstrap_rate_difference(
                    frame=subset_frame,
                    subset_label=subset_label,
                    rarity_column=rarity_column,
                    rare_label="Q1_rarest",
                    common_label="Q4_most_common",
                    outcome=outcome,
                    seed=73 + len(bootstrap_rows),
                )
            )
    bootstrap_summary = pd.DataFrame(bootstrap_rows)

    top_prototypes = (
        singleton_high_risk_subset.groupby(["prototype_token", "crystal_system", "spacegroup_number"], as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
            prototype_token_count_global=("prototype_token_count_global", "first"),
        )
        .query("n_materials >= 20")
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )
    top_spacegroups = (
        singleton_high_risk_subset.groupby(["spacegroup_number", "crystal_system"], as_index=False)
        .agg(
            n_materials=("material_id", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
            spacegroup_count_global=("spacegroup_count_global", "first"),
        )
        .query("n_materials >= 20")
        .sort_values(["collective_failure_rate", "n_materials"], ascending=[False, False])
        .reset_index(drop=True)
    )

    bool_columns = [
        "collective_failure",
        "collective_success",
        "collective_false_negative",
        "collective_false_positive",
        "unique_prototype",
        "exact_formula_in_mptrj",
    ]
    singleton_exact_for_logit = to_int_bools(singleton_exact_subset, bool_columns)
    singleton_high_risk_for_logit = to_int_bools(singleton_high_risk_subset, bool_columns)

    model_specs = [
        (
            "singleton_exact_rarity",
            "singleton_exact_subset",
            singleton_exact_for_logit,
            "{outcome} ~ np.log1p(prototype_token_count_in_density_tier) + np.log1p(spacegroup_count_in_density_tier) + C(crystal_system)",
        ),
        (
            "singleton_high_risk_rarity",
            "singleton_high_risk_subset",
            singleton_high_risk_for_logit,
            "{outcome} ~ np.log1p(prototype_token_count_in_density_tier) + np.log1p(spacegroup_count_in_density_tier) + C(crystal_system)",
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

    summary = {
        "singleton_exact_subset": {
            "n_materials": int(len(singleton_exact_subset)),
            "prototype_rarest_failure_rate": lookup_rate(
                rarity_rates,
                "singleton_exact_subset",
                "prototype_density_rarity_quartile",
                "Q1_rarest",
                "collective_failure_rate",
            ),
            "prototype_most_common_failure_rate": lookup_rate(
                rarity_rates,
                "singleton_exact_subset",
                "prototype_density_rarity_quartile",
                "Q4_most_common",
                "collective_failure_rate",
            ),
            "spacegroup_rarest_failure_rate": lookup_rate(
                rarity_rates,
                "singleton_exact_subset",
                "spacegroup_density_rarity_quartile",
                "Q1_rarest",
                "collective_failure_rate",
            ),
        },
        "singleton_high_risk_subset": {
            "n_materials": int(len(singleton_high_risk_subset)),
            "prototype_rarest_failure_rate": lookup_rate(
                rarity_rates,
                "singleton_high_risk_subset",
                "prototype_density_rarity_quartile",
                "Q1_rarest",
                "collective_failure_rate",
            ),
            "prototype_most_common_failure_rate": lookup_rate(
                rarity_rates,
                "singleton_high_risk_subset",
                "prototype_density_rarity_quartile",
                "Q4_most_common",
                "collective_failure_rate",
            ),
            "prototype_rarest_false_negative_rate": lookup_rate(
                rarity_rates,
                "singleton_high_risk_subset",
                "prototype_density_rarity_quartile",
                "Q1_rarest",
                "collective_false_negative_rate",
            ),
            "spacegroup_rarest_failure_rate": lookup_rate(
                rarity_rates,
                "singleton_high_risk_subset",
                "spacegroup_density_rarity_quartile",
                "Q1_rarest",
                "collective_failure_rate",
            ),
            "bootstrap_prototype_rarest_vs_common_failure": bootstrap_summary.loc[
                (bootstrap_summary["subset"] == "singleton_high_risk_subset")
                & (bootstrap_summary["rarity_axis"] == "prototype_density_rarity_quartile")
                & (bootstrap_summary["outcome"] == "collective_failure")
            ]
            .iloc[0][["observed_rate_diff", "q025", "q975", "fraction_diff_gt_zero"]]
            .to_dict(),
            "bootstrap_prototype_rarest_vs_common_false_negative": bootstrap_summary.loc[
                (bootstrap_summary["subset"] == "singleton_high_risk_subset")
                & (bootstrap_summary["rarity_axis"] == "prototype_density_rarity_quartile")
                & (bootstrap_summary["outcome"] == "collective_false_negative")
            ]
            .iloc[0][["observed_rate_diff", "q025", "q975", "fraction_diff_gt_zero"]]
            .to_dict(),
            "bootstrap_spacegroup_rarest_vs_common_failure": bootstrap_summary.loc[
                (bootstrap_summary["subset"] == "singleton_high_risk_subset")
                & (bootstrap_summary["rarity_axis"] == "spacegroup_density_rarity_quartile")
                & (bootstrap_summary["outcome"] == "collective_failure")
            ]
            .iloc[0][["observed_rate_diff", "q025", "q975", "fraction_diff_gt_zero"]]
            .to_dict(),
            "top_robust_prototype_hotspot": (
                top_prototypes.iloc[0][
                    [
                        "prototype_token",
                        "crystal_system",
                        "spacegroup_number",
                        "n_materials",
                        "collective_failure_rate",
                        "collective_false_negative_rate",
                    ]
                ].to_dict()
                if not top_prototypes.empty
                else {}
            ),
        },
    }

    rarity_rates.to_csv(output_dir / "rarity_quartile_rates.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "rarity_bootstrap_differences.csv", index=False)
    top_prototypes.to_csv(output_dir / "singleton_high_risk_top_prototypes.csv", index=False)
    top_spacegroups.to_csv(output_dir / "singleton_high_risk_spacegroup_hotspots.csv", index=False)
    logit_summary.to_csv(output_dir / "rarity_logit_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from dva_project.settings import RESULTS_DIR
from dva_project.utils import ensure_dir


OUTCOME_COLUMNS = [
    "collective_failure",
    "collective_false_negative",
    "collective_false_positive",
]


def fit_logit(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    model = smf.logit(
        f"{outcome} ~ unique_prototype + exact_formula_in_mptrj + unique_prototype:exact_formula_in_mptrj",
        data=frame,
    ).fit(disp=False)
    table = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    table["outcome"] = outcome
    table["odds_ratio"] = np.exp(table["Coef."])
    table["odds_ratio_ci_low"] = np.exp(table["[0.025"])
    table["odds_ratio_ci_high"] = np.exp(table["0.975]"])
    return table


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_09"
    ensure_dir(output_dir)

    frame = pd.read_csv(
        RESULTS_DIR / "tables" / "analysis_04" / "collective_failure_features.csv",
        usecols=[
            "collective_failure",
            "collective_success",
            "collective_false_negative",
            "collective_false_positive",
            "unique_prototype",
            "exact_formula_in_mptrj",
            "same_element_set_in_mptrj",
        ],
    )
    frame = frame.loc[frame["collective_failure"] | frame["collective_success"]].copy()
    for outcome in OUTCOME_COLUMNS:
        frame[outcome] = frame[outcome].astype(int)
    frame["unique_prototype"] = frame["unique_prototype"].astype(int)
    frame["exact_formula_in_mptrj"] = frame["exact_formula_in_mptrj"].astype(int)
    frame["same_element_set_in_mptrj"] = frame["same_element_set_in_mptrj"].astype(int)

    stratified = (
        frame.groupby(["unique_prototype", "exact_formula_in_mptrj"], as_index=False)
        .agg(
            n_materials=("collective_failure", "size"),
            collective_failure_rate=("collective_failure", "mean"),
            collective_false_negative_rate=("collective_false_negative", "mean"),
            collective_false_positive_rate=("collective_false_positive", "mean"),
            same_element_set_hit_rate=("same_element_set_in_mptrj", "mean"),
        )
        .sort_values(["unique_prototype", "exact_formula_in_mptrj"])
        .reset_index(drop=True)
    )

    baseline = stratified.loc[
        (stratified["unique_prototype"] == 1) & (stratified["exact_formula_in_mptrj"] == 0)
    ].iloc[0]
    risk_rows: list[dict] = []
    for row in stratified.itertuples(index=False):
        risk_rows.append(
            {
                "unique_prototype": int(row.unique_prototype),
                "exact_formula_in_mptrj": int(row.exact_formula_in_mptrj),
                "n_materials": int(row.n_materials),
                "collective_failure_rate": float(row.collective_failure_rate),
                "failure_rate_vs_baseline": float(row.collective_failure_rate / baseline.collective_failure_rate),
                "collective_false_negative_rate": float(row.collective_false_negative_rate),
                "false_negative_rate_vs_baseline": float(
                    row.collective_false_negative_rate / baseline.collective_false_negative_rate
                )
                if baseline.collective_false_negative_rate > 0
                else float("nan"),
            }
        )
    risk_summary = pd.DataFrame(risk_rows)

    logit_tables = [fit_logit(frame, outcome) for outcome in OUTCOME_COLUMNS]
    logit_summary = pd.concat(logit_tables, ignore_index=True)

    summary = {
        "baseline_group": {
            "unique_prototype": int(baseline["unique_prototype"]),
            "exact_formula_in_mptrj": int(baseline["exact_formula_in_mptrj"]),
            "collective_failure_rate": float(baseline["collective_failure_rate"]),
            "collective_false_negative_rate": float(baseline["collective_false_negative_rate"]),
        },
        "highest_failure_group": risk_summary.sort_values("collective_failure_rate", ascending=False).iloc[0].to_dict(),
        "highest_false_negative_group": risk_summary.sort_values(
            "collective_false_negative_rate",
            ascending=False,
        ).iloc[0].to_dict(),
    }

    stratified.to_csv(output_dir / "prototype_formula_overlap_stratified_rates.csv", index=False)
    risk_summary.to_csv(output_dir / "prototype_formula_overlap_risk_summary.csv", index=False)
    logit_summary.to_csv(output_dir / "prototype_formula_overlap_logit_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

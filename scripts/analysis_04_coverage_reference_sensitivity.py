from __future__ import annotations

import json

import numpy as np
import pandas as pd

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


REFERENCE_PREFIXES = [
    "wbm_mptrj",
    "wbm_mp2022",
    "wbm_mptrj_mp2022_union",
]


def summarize_by_label(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    rows: list[dict] = []
    for label, group in frame.groupby(label_column):
        same_set = group.loc[group["same_element_set_in_mptrj"]]
        rows.append(
            {
                "label": label,
                "n_materials": int(len(group)),
                "exact_formula_hit_rate": float(group["exact_formula_in_mptrj"].mean()),
                "same_element_set_hit_rate": float(group["same_element_set_in_mptrj"].mean()),
                "mean_same_element_set_formula_count": float(group["same_element_set_formula_count"].mean()),
                "median_same_element_set_formula_count": float(group["same_element_set_formula_count"].median()),
                "mean_min_l1_distance_same_element_set": float(
                    same_set["min_l1_distance_same_element_set"].mean()
                )
                if len(same_set)
                else np.nan,
                "median_min_l1_distance_same_element_set": float(
                    same_set["min_l1_distance_same_element_set"].median()
                )
                if len(same_set)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_04_coverage_sensitivity"
    ensure_dir(output_dir)

    labels = pd.read_csv(
        RESULTS_DIR / "tables" / "analysis_04" / "collective_failure_features.csv",
        usecols=["material_id", "collective_failure", "collective_success", "failure_type"],
    )
    subset = labels.loc[labels["collective_failure"] | labels["collective_success"]].copy()
    subset["binary_label"] = np.where(subset["collective_failure"], "failure", "success")
    subset["fine_label"] = subset["failure_type"].where(subset["collective_failure"], "success")

    binary_rows: list[pd.DataFrame] = []
    fine_rows: list[pd.DataFrame] = []
    delta_rows: list[dict] = []

    for prefix in REFERENCE_PREFIXES:
        coverage = pd.read_parquet(
            PROCESSED_DIR / f"{prefix}_material_coverage_proxy.parquet",
            columns=[
                "material_id",
                "exact_formula_in_mptrj",
                "same_element_set_in_mptrj",
                "same_element_set_formula_count",
                "min_l1_distance_same_element_set",
            ],
        ).drop_duplicates(subset=["material_id"])
        merged = subset.merge(coverage, on="material_id", how="left", validate="one_to_one")
        merged["exact_formula_in_mptrj"] = merged["exact_formula_in_mptrj"].fillna(False).astype(bool)
        merged["same_element_set_in_mptrj"] = merged["same_element_set_in_mptrj"].fillna(False).astype(bool)
        merged["same_element_set_formula_count"] = merged["same_element_set_formula_count"].fillna(0).astype(int)

        binary_summary = summarize_by_label(merged, "binary_label")
        binary_summary.insert(0, "reference_prefix", prefix)
        fine_summary = summarize_by_label(merged, "fine_label")
        fine_summary.insert(0, "reference_prefix", prefix)

        binary_rows.append(binary_summary)
        fine_rows.append(fine_summary)

        failure_row = binary_summary.loc[binary_summary["label"] == "failure"].iloc[0]
        success_row = binary_summary.loc[binary_summary["label"] == "success"].iloc[0]
        delta_rows.append(
            {
                "reference_prefix": prefix,
                "exact_formula_hit_rate_delta_failure_minus_success": float(
                    failure_row["exact_formula_hit_rate"] - success_row["exact_formula_hit_rate"]
                ),
                "same_element_set_hit_rate_delta_failure_minus_success": float(
                    failure_row["same_element_set_hit_rate"] - success_row["same_element_set_hit_rate"]
                ),
                "median_distance_delta_failure_minus_success": float(
                    failure_row["median_min_l1_distance_same_element_set"]
                    - success_row["median_min_l1_distance_same_element_set"]
                ),
                "mean_distance_delta_failure_minus_success": float(
                    failure_row["mean_min_l1_distance_same_element_set"]
                    - success_row["mean_min_l1_distance_same_element_set"]
                ),
            }
        )

    binary_summary_all = pd.concat(binary_rows, ignore_index=True)
    fine_summary_all = pd.concat(fine_rows, ignore_index=True)
    delta_summary = pd.DataFrame(delta_rows).sort_values("reference_prefix").reset_index(drop=True)

    summary = {
        "reference_prefixes": REFERENCE_PREFIXES,
        "failure_vs_success_exact_hit_deltas": {
            row.reference_prefix: row.exact_formula_hit_rate_delta_failure_minus_success
            for row in delta_summary.itertuples(index=False)
        },
        "failure_vs_success_same_element_hit_deltas": {
            row.reference_prefix: row.same_element_set_hit_rate_delta_failure_minus_success
            for row in delta_summary.itertuples(index=False)
        },
        "failure_vs_success_median_distance_deltas": {
            row.reference_prefix: row.median_distance_delta_failure_minus_success
            for row in delta_summary.itertuples(index=False)
        },
    }

    binary_summary_all.to_csv(output_dir / "coverage_reference_binary_summary.csv", index=False)
    fine_summary_all.to_csv(output_dir / "coverage_reference_fine_summary.csv", index=False)
    delta_summary.to_csv(output_dir / "coverage_reference_delta_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

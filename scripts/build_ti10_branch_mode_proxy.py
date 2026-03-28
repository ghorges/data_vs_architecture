from __future__ import annotations

import argparse
import json

import pandas as pd

from analysis_15_prototype_component_decomposition import load_analysis_frame
from analysis_16_component_ablation import assign_component_branch
from dva_project.settings import PROCESSED_DIR
from dva_project.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reusable structural-mode buckets for the singleton high-risk tI10 branch.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=4,
        help="Minimum tI10-branch count required for a structural bucket to receive its own label.",
    )
    parser.add_argument(
        "--output-prefix",
        default="singleton_high_risk_ti10_branch",
        help="Prefix for the generated processed output files.",
    )
    return parser.parse_args()


def build_frequent_bucket(series: pd.Series, min_count: int) -> tuple[pd.Series, pd.DataFrame]:
    counts = (
        series.value_counts(dropna=False)
        .rename_axis("bucket_label")
        .reset_index(name="candidate_count")
        .sort_values(["candidate_count", "bucket_label"], ascending=[False, True])
        .reset_index(drop=True)
    )
    selected = counts.loc[counts["candidate_count"] >= min_count, "bucket_label"].tolist()
    bucket = series.where(series.isin(selected), "other")
    return bucket, counts


def main() -> None:
    args = parse_args()
    ensure_dir(PROCESSED_DIR)

    frame = assign_component_branch(load_analysis_frame())
    subset = frame.loc[frame["component_branch"] == "tI10_branch"].copy()

    output = subset[
        [
            "material_id",
            "component_branch",
            "crystal_system",
            "frequent_spacegroup_bucket",
            "frequent_formula_family_bucket",
            "frequent_archetype_bucket",
            "frequent_wyckoff_signature_bucket",
            "frequent_prototype_token_bucket",
            "collective_failure",
            "collective_false_negative",
            "collective_false_positive",
        ]
    ].copy()

    output["ti10_spacegroup_wyckoff_bucket"] = (
        output["frequent_spacegroup_bucket"].astype(str)
        + "__"
        + output["frequent_wyckoff_signature_bucket"].astype(str)
    )
    output["ti10_formula_spacegroup_bucket"] = (
        output["frequent_formula_family_bucket"].astype(str)
        + "__"
        + output["frequent_spacegroup_bucket"].astype(str)
    )
    output["ti10_formula_spacegroup_wyckoff_bucket"] = (
        output["frequent_formula_family_bucket"].astype(str)
        + "__"
        + output["frequent_spacegroup_bucket"].astype(str)
        + "__"
        + output["frequent_wyckoff_signature_bucket"].astype(str)
    )

    pair_bucket, pair_counts = build_frequent_bucket(output["ti10_spacegroup_wyckoff_bucket"], min_count=args.min_count)
    triple_bucket, triple_counts = build_frequent_bucket(
        output["ti10_formula_spacegroup_wyckoff_bucket"], min_count=args.min_count
    )
    output["frequent_ti10_spacegroup_wyckoff_bucket"] = pair_bucket
    output["frequent_ti10_formula_spacegroup_wyckoff_bucket"] = triple_bucket

    summary = {
        "n_ti10_branch_materials": int(len(subset)),
        "min_count": int(args.min_count),
        "spacegroup_wyckoff_buckets": {
            "selected_count": int((pair_counts["candidate_count"] >= args.min_count).sum()),
            "coverage": float(
                pair_counts.loc[pair_counts["candidate_count"] >= args.min_count, "candidate_count"].sum() / len(subset)
            ),
            "top_selected": pair_counts.loc[pair_counts["candidate_count"] >= args.min_count].head(20).to_dict(
                orient="records"
            ),
        },
        "formula_spacegroup_wyckoff_buckets": {
            "selected_count": int((triple_counts["candidate_count"] >= args.min_count).sum()),
            "coverage": float(
                triple_counts.loc[triple_counts["candidate_count"] >= args.min_count, "candidate_count"].sum()
                / len(subset)
            ),
            "top_selected": triple_counts.loc[triple_counts["candidate_count"] >= args.min_count]
            .head(20)
            .to_dict(orient="records"),
        },
    }

    parquet_path = PROCESSED_DIR / f"{args.output_prefix}_mode_proxy.parquet"
    csv_path = PROCESSED_DIR / f"{args.output_prefix}_mode_proxy.csv"
    summary_path = PROCESSED_DIR / f"{args.output_prefix}_mode_proxy_summary.json"

    output.to_parquet(parquet_path, index=False)
    output.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

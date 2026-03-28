from __future__ import annotations

import argparse
import json

import pandas as pd

from analysis_15_prototype_component_decomposition import load_analysis_frame
from analysis_16_component_ablation import assign_component_branch
from dva_project.settings import PROCESSED_DIR
from dva_project.utils import ensure_dir


TRIPLE_INTERACTION_DEFINITIONS = {
    "formula_x_pearson_x_spacegroup": (
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_spacegroup_bucket",
    ),
    "formula_x_pearson_x_wyckoff": (
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_wyckoff_signature_bucket",
    ),
    "formula_x_spacegroup_x_wyckoff": (
        "frequent_formula_family_bucket",
        "frequent_spacegroup_bucket",
        "frequent_wyckoff_signature_bucket",
    ),
    "pearson_x_spacegroup_x_wyckoff": (
        "frequent_pearson_symbol_bucket",
        "frequent_spacegroup_bucket",
        "frequent_wyckoff_signature_bucket",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reusable higher-order interaction buckets for the singleton high-risk background branch.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Minimum background-branch count required for a triple-interaction bucket to receive its own label.",
    )
    parser.add_argument(
        "--output-prefix",
        default="singleton_high_risk_background",
        help="Prefix for the generated processed output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(PROCESSED_DIR)

    frame = assign_component_branch(load_analysis_frame())
    background = frame.loc[frame["component_branch"] == "background"].copy()

    output = background[
        [
            "material_id",
            "component_branch",
            "crystal_system",
            "frequent_spacegroup_bucket",
            "frequent_formula_family_bucket",
            "frequent_pearson_symbol_bucket",
            "frequent_archetype_bucket",
            "frequent_wyckoff_signature_bucket",
            "frequent_prototype_token_bucket",
        ]
    ].copy()

    summary: dict[str, object] = {
        "n_background_materials": int(len(background)),
        "min_count": int(args.min_count),
        "triple_interaction_buckets": {},
    }

    for interaction_name, columns in TRIPLE_INTERACTION_DEFINITIONS.items():
        raw_column = f"{interaction_name}_raw"
        bucket_column = f"frequent_{interaction_name}_bucket"
        output[raw_column] = output[list(columns)].astype(str).agg("__".join, axis=1)
        counts = (
            output.groupby(raw_column, as_index=False)
            .agg(candidate_count=("material_id", "size"))
            .sort_values(["candidate_count", raw_column], ascending=[False, True])
            .reset_index(drop=True)
        )
        selected = counts.loc[counts["candidate_count"] >= args.min_count, raw_column].tolist()
        output[bucket_column] = output[raw_column].where(output[raw_column].isin(selected), "other")
        output = output.drop(columns=[raw_column])
        summary["triple_interaction_buckets"][interaction_name] = {
            "selected_count": int(len(selected)),
            "coverage": float(counts.loc[counts["candidate_count"] >= args.min_count, "candidate_count"].sum() / len(background)),
            "top_selected": counts.loc[counts["candidate_count"] >= args.min_count].head(20).to_dict(orient="records"),
        }

    output_path_parquet = PROCESSED_DIR / f"{args.output_prefix}_higher_order_interaction_proxy.parquet"
    output_path_csv = PROCESSED_DIR / f"{args.output_prefix}_higher_order_interaction_proxy.csv"
    summary_path = PROCESSED_DIR / f"{args.output_prefix}_higher_order_interaction_proxy_summary.json"

    output.to_parquet(output_path_parquet, index=False)
    output.to_csv(output_path_csv, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

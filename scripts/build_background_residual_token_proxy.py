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
        description="Build a compact residual-token bucket proxy for the singleton high-risk background branch.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Minimum background-branch count required for a token to keep its own residual bucket.",
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

    counts = (
        output.groupby("frequent_prototype_token_bucket", as_index=False)
        .agg(candidate_count=("material_id", "size"))
        .sort_values(["candidate_count", "frequent_prototype_token_bucket"], ascending=[False, True])
        .reset_index(drop=True)
    )
    selected = counts.loc[counts["candidate_count"] >= args.min_count, "frequent_prototype_token_bucket"].tolist()
    output["frequent_background_residual_token_bucket"] = output["frequent_prototype_token_bucket"].where(
        output["frequent_prototype_token_bucket"].isin(selected),
        "other",
    )

    summary = {
        "n_background_materials": int(len(background)),
        "min_count": int(args.min_count),
        "selected_count": int(len(selected)),
        "coverage": float(counts.loc[counts["candidate_count"] >= args.min_count, "candidate_count"].sum() / len(background)),
        "selected_tokens": counts.loc[counts["candidate_count"] >= args.min_count].to_dict(orient="records"),
    }

    output_path_parquet = PROCESSED_DIR / f"{args.output_prefix}_residual_token_proxy.parquet"
    output_path_csv = PROCESSED_DIR / f"{args.output_prefix}_residual_token_proxy.csv"
    summary_path = PROCESSED_DIR / f"{args.output_prefix}_residual_token_proxy_summary.json"

    output.to_parquet(output_path_parquet, index=False)
    output.to_csv(output_path_csv, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

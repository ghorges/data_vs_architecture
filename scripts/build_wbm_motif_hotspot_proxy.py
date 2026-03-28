from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reusable motif-hotspot buckets for the singleton familiar-region subset.",
    )
    parser.add_argument(
        "--feature-table",
        default=str(RESULTS_DIR / "tables" / "analysis_04" / "collective_failure_features.csv"),
        help="Feature table that contains wyckoff_spglib and familiar-region flags.",
    )
    parser.add_argument(
        "--density-proxy",
        default=str(PROCESSED_DIR / "wbm_mptrj_mp2022_union_material_structure_density_proxy.parquet"),
        help="Density proxy that contains exact-formula density tiers.",
    )
    parser.add_argument(
        "--min-prototype-count",
        type=int,
        default=20,
        help="Minimum singleton-high-risk candidate count for a prototype token to receive its own bucket.",
    )
    parser.add_argument(
        "--min-spacegroup-count",
        type=int,
        default=20,
        help="Minimum singleton-high-risk candidate count for a spacegroup to receive its own bucket.",
    )
    parser.add_argument(
        "--output-prefix",
        default="wbm_mptrj_mp2022_union",
        help="Prefix for the generated processed output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(PROCESSED_DIR)

    features = pd.read_csv(
        Path(args.feature_table),
        usecols=[
            "material_id",
            "wyckoff_spglib",
            "spacegroup_number",
            "crystal_system",
            "unique_prototype",
            "exact_formula_in_mptrj",
        ],
    )
    density = pd.read_parquet(
        Path(args.density_proxy),
        columns=["material_id", "exact_formula_nsites_density_tier"],
    )
    frame = features.merge(density, on="material_id", how="left", validate="one_to_one")
    frame["prototype_token"] = frame["wyckoff_spglib"].fillna("unknown").str.split(":").str[0]
    frame["spacegroup_number"] = frame["spacegroup_number"].fillna(-1).astype(int)
    frame["crystal_system"] = frame["crystal_system"].fillna("unknown")
    frame["singleton_high_risk_candidate"] = (
        frame["exact_formula_in_mptrj"].fillna(False).astype(bool)
        & (~frame["unique_prototype"].fillna(True).astype(bool))
        & (frame["exact_formula_nsites_density_tier"] == "singleton_formula_signature")
    )

    candidate = frame.loc[frame["singleton_high_risk_candidate"]].copy()
    prototype_counts = (
        candidate.groupby(["prototype_token", "spacegroup_number", "crystal_system"], as_index=False)
        .agg(candidate_count=("material_id", "size"))
        .sort_values(["candidate_count", "prototype_token"], ascending=[False, True])
        .reset_index(drop=True)
    )
    selected_prototypes = prototype_counts.loc[
        prototype_counts["candidate_count"] >= args.min_prototype_count,
        "prototype_token",
    ].tolist()

    spacegroup_counts = (
        candidate.groupby(["spacegroup_number", "crystal_system"], as_index=False)
        .agg(candidate_count=("material_id", "size"))
        .sort_values(["candidate_count", "spacegroup_number"], ascending=[False, True])
        .reset_index(drop=True)
    )
    selected_spacegroups = spacegroup_counts.loc[
        spacegroup_counts["candidate_count"] >= args.min_spacegroup_count,
        "spacegroup_number",
    ].astype(int).tolist()

    frame["frequent_prototype_token_bucket"] = frame["prototype_token"].where(
        frame["prototype_token"].isin(selected_prototypes),
        "other",
    )
    frame["frequent_spacegroup_bucket"] = frame["spacegroup_number"].map(
        lambda value: f"sg_{int(value)}" if int(value) in selected_spacegroups else "other"
    )
    frame["is_frequent_prototype_token_bucket"] = frame["frequent_prototype_token_bucket"] != "other"
    frame["is_frequent_spacegroup_bucket"] = frame["frequent_spacegroup_bucket"] != "other"

    summary = {
        "feature_table": str(Path(args.feature_table)),
        "density_proxy": str(Path(args.density_proxy)),
        "output_prefix": args.output_prefix,
        "n_wbm_materials": int(len(frame)),
        "n_singleton_high_risk_candidates": int(len(candidate)),
        "min_prototype_count": args.min_prototype_count,
        "min_spacegroup_count": args.min_spacegroup_count,
        "selected_prototype_tokens": prototype_counts.loc[
            prototype_counts["candidate_count"] >= args.min_prototype_count
        ].to_dict(orient="records"),
        "selected_spacegroups": spacegroup_counts.loc[
            spacegroup_counts["candidate_count"] >= args.min_spacegroup_count
        ].to_dict(orient="records"),
    }

    output_columns = [
        "material_id",
        "prototype_token",
        "spacegroup_number",
        "crystal_system",
        "singleton_high_risk_candidate",
        "frequent_prototype_token_bucket",
        "frequent_spacegroup_bucket",
        "is_frequent_prototype_token_bucket",
        "is_frequent_spacegroup_bucket",
    ]
    frame[output_columns].to_parquet(
        PROCESSED_DIR / f"{args.output_prefix}_material_motif_hotspot_proxy.parquet",
        index=False,
    )
    frame[output_columns].to_csv(
        PROCESSED_DIR / f"{args.output_prefix}_material_motif_hotspot_proxy.csv",
        index=False,
    )
    (PROCESSED_DIR / f"{args.output_prefix}_motif_hotspot_proxy_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

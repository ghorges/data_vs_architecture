from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dva_project.settings import PROCESSED_DIR
from dva_project.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reusable prototype-component buckets for singleton familiar-region motif analysis.",
    )
    parser.add_argument(
        "--motif-hotspot-proxy",
        default=str(PROCESSED_DIR / "wbm_mptrj_mp2022_union_material_motif_hotspot_proxy.parquet"),
        help="Processed motif-hotspot proxy with prototype tokens and singleton-high-risk flags.",
    )
    parser.add_argument(
        "--min-formula-family-count",
        type=int,
        default=50,
        help="Minimum singleton-high-risk candidate count for a formula family bucket.",
    )
    parser.add_argument(
        "--min-pearson-count",
        type=int,
        default=50,
        help="Minimum singleton-high-risk candidate count for a Pearson-symbol bucket.",
    )
    parser.add_argument(
        "--min-archetype-count",
        type=int,
        default=30,
        help="Minimum singleton-high-risk candidate count for an archetype bucket.",
    )
    parser.add_argument(
        "--min-wyckoff-count",
        type=int,
        default=30,
        help="Minimum singleton-high-risk candidate count for a Wyckoff-signature bucket.",
    )
    parser.add_argument(
        "--output-prefix",
        default="wbm_mptrj_mp2022_union",
        help="Prefix for the generated processed output files.",
    )
    return parser.parse_args()


def select_buckets(candidate: pd.DataFrame, column: str, min_count: int) -> tuple[list[str], pd.DataFrame]:
    counts = (
        candidate.groupby(column, as_index=False)
        .agg(candidate_count=("material_id", "size"))
        .sort_values(["candidate_count", column], ascending=[False, True])
        .reset_index(drop=True)
    )
    selected = counts.loc[counts["candidate_count"] >= min_count, column].tolist()
    return selected, counts


def main() -> None:
    args = parse_args()
    ensure_dir(PROCESSED_DIR)

    frame = pd.read_parquet(
        Path(args.motif_hotspot_proxy),
        columns=[
            "material_id",
            "prototype_token",
            "spacegroup_number",
            "crystal_system",
            "singleton_high_risk_candidate",
        ],
    ).copy()
    frame["prototype_token"] = frame["prototype_token"].fillna("unknown")
    frame["spacegroup_number"] = frame["spacegroup_number"].fillna(-1).astype(int)
    frame["crystal_system"] = frame["crystal_system"].fillna("unknown")

    token_parts = frame["prototype_token"].str.split("_", n=3, expand=True)
    frame["prototype_formula_family"] = token_parts[0].fillna("unknown")
    frame["prototype_pearson_symbol"] = token_parts[1].fillna("unknown")
    frame["prototype_spacegroup_component"] = token_parts[2].fillna("unknown")
    frame["prototype_wyckoff_signature"] = token_parts[3].fillna("unknown")
    frame["prototype_archetype_token"] = (
        frame["prototype_formula_family"]
        + "_"
        + frame["prototype_pearson_symbol"]
        + "_"
        + frame["prototype_spacegroup_component"]
    )

    candidate = frame.loc[frame["singleton_high_risk_candidate"]].copy()

    selected_formula_families, formula_counts = select_buckets(
        candidate,
        "prototype_formula_family",
        args.min_formula_family_count,
    )
    selected_pearson_symbols, pearson_counts = select_buckets(
        candidate,
        "prototype_pearson_symbol",
        args.min_pearson_count,
    )
    selected_archetypes, archetype_counts = select_buckets(
        candidate,
        "prototype_archetype_token",
        args.min_archetype_count,
    )
    selected_wyckoff_signatures, wyckoff_counts = select_buckets(
        candidate,
        "prototype_wyckoff_signature",
        args.min_wyckoff_count,
    )

    frame["frequent_formula_family_bucket"] = frame["prototype_formula_family"].where(
        frame["prototype_formula_family"].isin(selected_formula_families),
        "other",
    )
    frame["frequent_pearson_symbol_bucket"] = frame["prototype_pearson_symbol"].where(
        frame["prototype_pearson_symbol"].isin(selected_pearson_symbols),
        "other",
    )
    frame["frequent_archetype_bucket"] = frame["prototype_archetype_token"].where(
        frame["prototype_archetype_token"].isin(selected_archetypes),
        "other",
    )
    frame["frequent_wyckoff_signature_bucket"] = frame["prototype_wyckoff_signature"].where(
        frame["prototype_wyckoff_signature"].isin(selected_wyckoff_signatures),
        "other",
    )
    frame["is_frequent_formula_family_bucket"] = frame["frequent_formula_family_bucket"] != "other"
    frame["is_frequent_pearson_symbol_bucket"] = frame["frequent_pearson_symbol_bucket"] != "other"
    frame["is_frequent_archetype_bucket"] = frame["frequent_archetype_bucket"] != "other"
    frame["is_frequent_wyckoff_signature_bucket"] = frame["frequent_wyckoff_signature_bucket"] != "other"

    summary = {
        "motif_hotspot_proxy": str(Path(args.motif_hotspot_proxy)),
        "output_prefix": args.output_prefix,
        "n_wbm_materials": int(len(frame)),
        "n_singleton_high_risk_candidates": int(len(candidate)),
        "thresholds": {
            "formula_family": args.min_formula_family_count,
            "pearson_symbol": args.min_pearson_count,
            "archetype": args.min_archetype_count,
            "wyckoff_signature": args.min_wyckoff_count,
        },
        "selected_formula_families": formula_counts.loc[
            formula_counts["candidate_count"] >= args.min_formula_family_count
        ].to_dict(orient="records"),
        "selected_pearson_symbols": pearson_counts.loc[
            pearson_counts["candidate_count"] >= args.min_pearson_count
        ].to_dict(orient="records"),
        "selected_archetypes": archetype_counts.loc[
            archetype_counts["candidate_count"] >= args.min_archetype_count
        ].to_dict(orient="records"),
        "selected_wyckoff_signatures": wyckoff_counts.loc[
            wyckoff_counts["candidate_count"] >= args.min_wyckoff_count
        ].to_dict(orient="records"),
    }

    output_columns = [
        "material_id",
        "prototype_token",
        "spacegroup_number",
        "crystal_system",
        "singleton_high_risk_candidate",
        "prototype_formula_family",
        "prototype_pearson_symbol",
        "prototype_spacegroup_component",
        "prototype_wyckoff_signature",
        "prototype_archetype_token",
        "frequent_formula_family_bucket",
        "frequent_pearson_symbol_bucket",
        "frequent_archetype_bucket",
        "frequent_wyckoff_signature_bucket",
        "is_frequent_formula_family_bucket",
        "is_frequent_pearson_symbol_bucket",
        "is_frequent_archetype_bucket",
        "is_frequent_wyckoff_signature_bucket",
    ]
    frame[output_columns].to_parquet(
        PROCESSED_DIR / f"{args.output_prefix}_material_prototype_component_proxy.parquet",
        index=False,
    )
    frame[output_columns].to_csv(
        PROCESSED_DIR / f"{args.output_prefix}_material_prototype_component_proxy.csv",
        index=False,
    )
    (PROCESSED_DIR / f"{args.output_prefix}_prototype_component_proxy_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

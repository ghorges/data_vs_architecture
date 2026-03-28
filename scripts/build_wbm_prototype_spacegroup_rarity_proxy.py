from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


def add_rank(frame: pd.DataFrame, count_column: str, group_column: str | None = None) -> pd.DataFrame:
    enriched = frame.copy()
    if group_column is None:
        enriched[f"{count_column}_rank_desc"] = enriched[count_column].rank(method="dense", ascending=False).astype(int)
    else:
        enriched[f"{count_column}_rank_desc"] = (
            enriched.groupby(group_column)[count_column].rank(method="dense", ascending=False).astype(int)
        )
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build benchmark-side prototype and spacegroup rarity proxies from WBM structural tags.",
    )
    parser.add_argument(
        "--feature-table",
        default=str(RESULTS_DIR / "tables" / "analysis_04" / "collective_failure_features.csv"),
        help="Feature table that contains wyckoff_spglib / spacegroup_number / crystal_system.",
    )
    parser.add_argument(
        "--density-proxy",
        default=str(PROCESSED_DIR / "wbm_mptrj_mp2022_union_material_structure_density_proxy.parquet"),
        help="Processed density proxy used to attach density tiers.",
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

    feature_table = pd.read_csv(
        Path(args.feature_table),
        usecols=["material_id", "wyckoff_spglib", "spacegroup_number", "crystal_system"],
    )
    density_proxy = pd.read_parquet(
        Path(args.density_proxy),
        columns=["material_id", "exact_formula_nsites_density_tier", "exact_formula_nsites_in_reference"],
    )
    frame = feature_table.merge(density_proxy, on="material_id", how="left", validate="one_to_one")
    frame["prototype_token"] = frame["wyckoff_spglib"].fillna("unknown").str.split(":").str[0]
    frame["spacegroup_number"] = frame["spacegroup_number"].fillna(-1).astype(int)
    frame["crystal_system"] = frame["crystal_system"].fillna("unknown")
    frame["exact_formula_nsites_density_tier"] = frame["exact_formula_nsites_density_tier"].fillna("no_match")
    frame["exact_formula_nsites_in_reference"] = frame["exact_formula_nsites_in_reference"].fillna(False).astype(bool)

    total_n = len(frame)
    crystal_totals = frame.groupby("crystal_system", as_index=False).agg(
        crystal_system_count=("material_id", "size")
    )
    density_totals = frame.groupby("exact_formula_nsites_density_tier", as_index=False).agg(
        density_tier_count=("material_id", "size")
    )

    prototype_global = frame.groupby("prototype_token", as_index=False).agg(
        prototype_token_count_global=("material_id", "size")
    )
    prototype_global = add_rank(prototype_global, "prototype_token_count_global")

    spacegroup_global = frame.groupby("spacegroup_number", as_index=False).agg(
        spacegroup_count_global=("material_id", "size")
    )
    spacegroup_global = add_rank(spacegroup_global, "spacegroup_count_global")

    prototype_by_crystal = frame.groupby(["crystal_system", "prototype_token"], as_index=False).agg(
        prototype_token_count_in_crystal_system=("material_id", "size")
    )
    prototype_by_crystal = prototype_by_crystal.merge(crystal_totals, on="crystal_system", how="left")
    prototype_by_crystal["prototype_token_share_in_crystal_system"] = (
        prototype_by_crystal["prototype_token_count_in_crystal_system"] / prototype_by_crystal["crystal_system_count"]
    )
    prototype_by_crystal = add_rank(
        prototype_by_crystal,
        "prototype_token_count_in_crystal_system",
        group_column="crystal_system",
    )

    prototype_by_density = frame.groupby(["exact_formula_nsites_density_tier", "prototype_token"], as_index=False).agg(
        prototype_token_count_in_density_tier=("material_id", "size")
    )
    prototype_by_density = prototype_by_density.merge(density_totals, on="exact_formula_nsites_density_tier", how="left")
    prototype_by_density["prototype_token_share_in_density_tier"] = (
        prototype_by_density["prototype_token_count_in_density_tier"] / prototype_by_density["density_tier_count"]
    )
    prototype_by_density = add_rank(
        prototype_by_density,
        "prototype_token_count_in_density_tier",
        group_column="exact_formula_nsites_density_tier",
    )

    spacegroup_by_density = frame.groupby(["exact_formula_nsites_density_tier", "spacegroup_number"], as_index=False).agg(
        spacegroup_count_in_density_tier=("material_id", "size")
    )
    spacegroup_by_density = spacegroup_by_density.merge(density_totals, on="exact_formula_nsites_density_tier", how="left")
    spacegroup_by_density["spacegroup_share_in_density_tier"] = (
        spacegroup_by_density["spacegroup_count_in_density_tier"] / spacegroup_by_density["density_tier_count"]
    )
    spacegroup_by_density = add_rank(
        spacegroup_by_density,
        "spacegroup_count_in_density_tier",
        group_column="exact_formula_nsites_density_tier",
    )

    merged = (
        frame.merge(prototype_global, on="prototype_token", how="left")
        .merge(spacegroup_global, on="spacegroup_number", how="left")
        .merge(prototype_by_crystal, on=["crystal_system", "prototype_token"], how="left")
        .merge(prototype_by_density, on=["exact_formula_nsites_density_tier", "prototype_token"], how="left")
        .merge(
            spacegroup_by_density,
            on=["exact_formula_nsites_density_tier", "spacegroup_number"],
            how="left",
            suffixes=("", "_spacegroup_density"),
        )
        .sort_values("material_id")
        .reset_index(drop=True)
    )
    if "density_tier_count_spacegroup_density" in merged.columns:
        merged = merged.drop(columns=["density_tier_count_spacegroup_density"])

    int_columns = [
        "prototype_token_count_global",
        "prototype_token_count_global_rank_desc",
        "spacegroup_count_global",
        "spacegroup_count_global_rank_desc",
        "prototype_token_count_in_crystal_system",
        "prototype_token_count_in_crystal_system_rank_desc",
        "prototype_token_count_in_density_tier",
        "prototype_token_count_in_density_tier_rank_desc",
        "spacegroup_count_in_density_tier",
        "spacegroup_count_in_density_tier_rank_desc",
        "crystal_system_count",
        "density_tier_count",
    ]
    for column in int_columns:
        merged[column] = merged[column].fillna(0).astype(int)

    float_columns = [
        "prototype_token_share_in_crystal_system",
        "prototype_token_share_in_density_tier",
        "spacegroup_share_in_density_tier",
    ]
    for column in float_columns:
        merged[column] = merged[column].astype(float)

    summary = {
        "feature_table": str(Path(args.feature_table)),
        "density_proxy": str(Path(args.density_proxy)),
        "output_prefix": args.output_prefix,
        "n_wbm_materials": int(total_n),
        "n_unique_prototype_tokens": int(merged["prototype_token"].nunique()),
        "n_unique_spacegroups": int(merged["spacegroup_number"].nunique()),
        "top_prototype_tokens": prototype_global.sort_values("prototype_token_count_global", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "top_spacegroups": spacegroup_global.sort_values("spacegroup_count_global", ascending=False)
        .head(10)
        .to_dict(orient="records"),
    }

    merged.to_parquet(PROCESSED_DIR / f"{args.output_prefix}_material_prototype_spacegroup_rarity_proxy.parquet", index=False)
    merged.to_csv(PROCESSED_DIR / f"{args.output_prefix}_material_prototype_spacegroup_rarity_proxy.csv", index=False)
    (PROCESSED_DIR / f"{args.output_prefix}_prototype_spacegroup_rarity_proxy_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

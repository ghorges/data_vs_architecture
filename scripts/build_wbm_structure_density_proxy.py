from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from dva_project.settings import PROCESSED_DIR
from dva_project.utils import ensure_dir


@lru_cache(maxsize=400000)
def normalize_formula(formula: str) -> tuple[str, str, str, int]:
    composition = Composition(formula).fractional_composition
    fraction_map = {str(el): float(amount) for el, amount in composition.as_dict().items()}
    normalized = Composition(fraction_map)
    return (
        normalized.reduced_formula,
        json.dumps(fraction_map, sort_keys=True),
        normalized.anonymized_formula,
        len(fraction_map),
    )


def normalized_entropy(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probs = values / total
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(entropy / math.log(len(values)))


def assign_density_tier(signature_count: int, structure_share: float | None) -> str:
    if signature_count <= 0 or pd.isna(structure_share):
        return "no_match"
    if signature_count == 1:
        return "singleton_formula_signature"
    if structure_share >= 0.5:
        return "dominant_multi_signature"
    return "minority_multi_signature"


def build_formula_density_stats(reference: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for fraction_key, subset in reference.groupby("fractional_composition_json", sort=False):
        subset = subset.sort_values(["n_structures", "n_materials", "n_sites"], ascending=[False, False, True]).reset_index(drop=True)
        total_structures = float(subset["n_structures"].sum())
        total_materials = float(subset["n_materials"].sum())
        signature_count = int(len(subset))
        structure_shares = subset["n_structures"].to_numpy(dtype=float) / total_structures
        material_shares = subset["n_materials"].to_numpy(dtype=float) / total_materials
        structure_ranks = subset["n_structures"].rank(method="dense", ascending=False).astype(int).to_numpy()
        material_ranks = subset["n_materials"].rank(method="dense", ascending=False).astype(int).to_numpy()
        max_structure_share = float(structure_shares.max())
        max_material_share = float(material_shares.max())
        entropy_structures = normalized_entropy(subset["n_structures"].to_numpy(dtype=float))
        entropy_materials = normalized_entropy(subset["n_materials"].to_numpy(dtype=float))

        for idx, row in enumerate(subset.itertuples(index=False)):
            structure_share = float(structure_shares[idx])
            material_share = float(material_shares[idx])
            rows.append(
                {
                    "fractional_composition_json": fraction_key,
                    "n_sites": int(row.n_sites),
                    "exact_formula_signature_count": signature_count,
                    "exact_formula_total_structure_count": int(total_structures),
                    "exact_formula_total_material_count": int(total_materials),
                    "exact_formula_structure_entropy_norm": entropy_structures,
                    "exact_formula_material_entropy_norm": entropy_materials,
                    "exact_formula_nsites_structure_count": int(row.n_structures),
                    "exact_formula_nsites_material_count": int(row.n_materials),
                    "exact_formula_nsites_structure_share": structure_share,
                    "exact_formula_nsites_material_share": material_share,
                    "exact_formula_nsites_structure_rank_desc": int(structure_ranks[idx]),
                    "exact_formula_nsites_material_rank_desc": int(material_ranks[idx]),
                    "exact_formula_nsites_is_structure_mode": bool(structure_share == max_structure_share),
                    "exact_formula_nsites_is_material_mode": bool(material_share == max_material_share),
                    "exact_formula_nsites_structure_share_gap_to_mode": float(max_structure_share - structure_share),
                    "exact_formula_nsites_material_share_gap_to_mode": float(max_material_share - material_share),
                    "exact_formula_is_singleton_signature": bool(signature_count == 1),
                    "exact_formula_nsites_density_tier": assign_density_tier(signature_count, structure_share),
                }
            )
    return pd.DataFrame(rows)


def build_anonymous_density_stats(reference: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        reference.groupby(["anonymous_formula", "n_sites"], as_index=False)
        .agg(
            n_structures=("n_structures", "sum"),
            n_materials=("n_materials", "sum"),
        )
        .sort_values(["anonymous_formula", "n_sites"])
        .reset_index(drop=True)
    )
    rows: list[dict] = []
    for anonymous_formula, subset in aggregated.groupby("anonymous_formula", sort=False):
        subset = subset.sort_values(["n_structures", "n_materials", "n_sites"], ascending=[False, False, True]).reset_index(drop=True)
        total_structures = float(subset["n_structures"].sum())
        total_materials = float(subset["n_materials"].sum())
        signature_count = int(len(subset))
        structure_shares = subset["n_structures"].to_numpy(dtype=float) / total_structures
        material_shares = subset["n_materials"].to_numpy(dtype=float) / total_materials
        structure_ranks = subset["n_structures"].rank(method="dense", ascending=False).astype(int).to_numpy()
        material_ranks = subset["n_materials"].rank(method="dense", ascending=False).astype(int).to_numpy()
        max_structure_share = float(structure_shares.max())
        max_material_share = float(material_shares.max())
        entropy_structures = normalized_entropy(subset["n_structures"].to_numpy(dtype=float))
        entropy_materials = normalized_entropy(subset["n_materials"].to_numpy(dtype=float))

        for idx, row in enumerate(subset.itertuples(index=False)):
            structure_share = float(structure_shares[idx])
            material_share = float(material_shares[idx])
            rows.append(
                {
                    "anonymous_formula": anonymous_formula,
                    "n_sites": int(row.n_sites),
                    "anonymous_formula_signature_count": signature_count,
                    "anonymous_formula_total_structure_count": int(total_structures),
                    "anonymous_formula_total_material_count": int(total_materials),
                    "anonymous_formula_structure_entropy_norm": entropy_structures,
                    "anonymous_formula_material_entropy_norm": entropy_materials,
                    "anonymous_nsites_structure_count": int(row.n_structures),
                    "anonymous_nsites_material_count": int(row.n_materials),
                    "anonymous_nsites_structure_share": structure_share,
                    "anonymous_nsites_material_share": material_share,
                    "anonymous_nsites_structure_rank_desc": int(structure_ranks[idx]),
                    "anonymous_nsites_material_rank_desc": int(material_ranks[idx]),
                    "anonymous_nsites_is_structure_mode": bool(structure_share == max_structure_share),
                    "anonymous_nsites_is_material_mode": bool(material_share == max_material_share),
                    "anonymous_nsites_structure_share_gap_to_mode": float(max_structure_share - structure_share),
                    "anonymous_nsites_material_share_gap_to_mode": float(max_material_share - material_share),
                    "anonymous_formula_is_singleton_signature": bool(signature_count == 1),
                    "anonymous_nsites_density_tier": assign_density_tier(signature_count, structure_share).replace(
                        "formula",
                        "anonymous",
                    ),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build WBM structure-density proxies from training-reference structure signatures.",
    )
    parser.add_argument(
        "--reference-index",
        default=str(PROCESSED_DIR / "mptrj_mp2022_union_structure_signature_index.parquet"),
        help="Processed structure-signature parquet used as the training reference set.",
    )
    parser.add_argument(
        "--structure-overlap-proxy",
        default=str(PROCESSED_DIR / "wbm_mptrj_mp2022_union_material_structure_overlap_proxy.parquet"),
        help="Processed WBM structure-overlap proxy parquet to enrich.",
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

    reference_index = pd.read_parquet(Path(args.reference_index)).copy()
    reference_index["n_sites"] = reference_index["n_sites"].astype(int)

    formula_density = build_formula_density_stats(reference_index)
    anonymous_density = build_anonymous_density_stats(reference_index[["anonymous_formula", "n_sites", "n_structures", "n_materials"]])
    formula_density = formula_density[
        [
            "fractional_composition_json",
            "n_sites",
            "exact_formula_structure_entropy_norm",
            "exact_formula_material_entropy_norm",
            "exact_formula_nsites_structure_share",
            "exact_formula_nsites_material_share",
            "exact_formula_nsites_structure_rank_desc",
            "exact_formula_nsites_material_rank_desc",
            "exact_formula_nsites_is_structure_mode",
            "exact_formula_nsites_is_material_mode",
            "exact_formula_nsites_structure_share_gap_to_mode",
            "exact_formula_nsites_material_share_gap_to_mode",
            "exact_formula_is_singleton_signature",
            "exact_formula_nsites_density_tier",
        ]
    ].copy()

    base_proxy = pd.read_parquet(Path(args.structure_overlap_proxy)).copy()
    base_proxy["n_sites"] = base_proxy["n_sites"].astype(int)
    normalized = base_proxy["formula"].map(normalize_formula)
    base_proxy["anonymous_formula"] = normalized.map(lambda item: item[2])
    base_proxy["n_elements"] = normalized.map(lambda item: item[3])

    merged = (
        base_proxy.merge(
            formula_density,
            on=["fractional_composition_json", "n_sites"],
            how="left",
        )
        .merge(
            anonymous_density,
            on=["anonymous_formula", "n_sites"],
            how="left",
        )
        .sort_values("material_id")
        .reset_index(drop=True)
    )

    bool_columns = [
        "exact_formula_is_singleton_signature",
        "exact_formula_nsites_is_structure_mode",
        "exact_formula_nsites_is_material_mode",
        "anonymous_formula_is_singleton_signature",
        "anonymous_nsites_is_structure_mode",
        "anonymous_nsites_is_material_mode",
    ]
    for column in bool_columns:
        merged[column] = merged[column].fillna(False).astype(bool)

    int_columns = [
        "exact_formula_signature_count",
        "exact_formula_total_structure_count",
        "exact_formula_total_material_count",
        "exact_formula_nsites_structure_count",
        "exact_formula_nsites_material_count",
        "exact_formula_nsites_structure_rank_desc",
        "exact_formula_nsites_material_rank_desc",
        "anonymous_formula_signature_count",
        "anonymous_formula_total_structure_count",
        "anonymous_formula_total_material_count",
        "anonymous_nsites_structure_count",
        "anonymous_nsites_material_count",
        "anonymous_nsites_structure_rank_desc",
        "anonymous_nsites_material_rank_desc",
    ]
    for column in int_columns:
        merged[column] = merged[column].fillna(0).astype(int)

    float_columns = [
        "exact_formula_structure_entropy_norm",
        "exact_formula_material_entropy_norm",
        "exact_formula_nsites_structure_share",
        "exact_formula_nsites_material_share",
        "exact_formula_nsites_structure_share_gap_to_mode",
        "exact_formula_nsites_material_share_gap_to_mode",
        "anonymous_formula_structure_entropy_norm",
        "anonymous_formula_material_entropy_norm",
        "anonymous_nsites_structure_share",
        "anonymous_nsites_material_share",
        "anonymous_nsites_structure_share_gap_to_mode",
        "anonymous_nsites_material_share_gap_to_mode",
    ]
    for column in float_columns:
        merged[column] = merged[column].astype(float)

    merged["exact_formula_nsites_density_tier"] = merged["exact_formula_nsites_density_tier"].fillna("no_match")
    merged["anonymous_nsites_density_tier"] = merged["anonymous_nsites_density_tier"].fillna("no_match")

    summary = {
        "reference_index": str(Path(args.reference_index)),
        "structure_overlap_proxy": str(Path(args.structure_overlap_proxy)),
        "output_prefix": args.output_prefix,
        "n_wbm_materials": int(len(merged)),
        "fraction_exact_formula_nsites_singleton_signature": float(merged["exact_formula_is_singleton_signature"].mean()),
        "fraction_exact_formula_nsites_structure_mode": float(merged["exact_formula_nsites_is_structure_mode"].mean()),
        "fraction_anonymous_nsites_structure_mode": float(merged["anonymous_nsites_is_structure_mode"].mean()),
        "exact_formula_density_tier_distribution": (
            merged["exact_formula_nsites_density_tier"].value_counts(normalize=True).sort_index().to_dict()
        ),
        "anonymous_density_tier_distribution": (
            merged["anonymous_nsites_density_tier"].value_counts(normalize=True).sort_index().to_dict()
        ),
    }

    merged.to_parquet(PROCESSED_DIR / f"{args.output_prefix}_material_structure_density_proxy.parquet", index=False)
    merged.to_csv(PROCESSED_DIR / f"{args.output_prefix}_material_structure_density_proxy.csv", index=False)
    (PROCESSED_DIR / f"{args.output_prefix}_structure_density_proxy_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

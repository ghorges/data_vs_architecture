from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from tqdm import tqdm

from dva_project.settings import PROCESSED_DIR
from dva_project.utils import ensure_dir


@lru_cache(maxsize=400000)
def normalize_formula(formula: str) -> tuple[str, str, int]:
    composition = Composition(formula).fractional_composition
    fraction_map = {str(el): float(amount) for el, amount in composition.as_dict().items()}
    reduced_formula = Composition(fraction_map).reduced_formula
    return reduced_formula, json.dumps(fraction_map, sort_keys=True), len(fraction_map)


def distance_to_reference_range(
    values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    below = np.maximum(lower_bounds - values, 0.0)
    above = np.maximum(values - upper_bounds, 0.0)
    return below + above


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build WBM-to-training structure overlap proxies using formula, n_sites, and volume-per-atom signatures.",
    )
    parser.add_argument(
        "--reference-index",
        default=str(PROCESSED_DIR / "mptrj_mp2022_union_structure_signature_index.parquet"),
        help="Processed structure-signature parquet used as the training reference set.",
    )
    parser.add_argument(
        "--output-prefix",
        default="wbm_mptrj_mp2022_union",
        help="Prefix for the generated processed output files.",
    )
    parser.add_argument(
        "--limit-materials",
        type=int,
        default=None,
        help="Optional cap on the number of WBM materials for debugging.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(PROCESSED_DIR)

    reference_index_path = Path(args.reference_index)
    reference = pd.read_parquet(reference_index_path).copy()
    reference["n_sites"] = reference["n_sites"].astype(int)

    formula_summary = (
        reference.groupby("fractional_composition_json", as_index=False)
        .agg(
            reduced_formula=("reduced_formula", "first"),
            n_elements=("n_elements", "first"),
            exact_formula_signature_count=("n_sites", "size"),
            exact_formula_total_structure_count=("n_structures", "sum"),
            exact_formula_total_material_count=("n_materials", "sum"),
        )
        .sort_values(["exact_formula_total_structure_count", "reduced_formula"], ascending=[False, True])
        .reset_index(drop=True)
    )
    formula_summary["exact_formula_in_reference"] = True

    exact_signature = reference.rename(
        columns={
            "n_structures": "exact_formula_nsites_structure_count",
            "n_materials": "exact_formula_nsites_material_count",
            "mean_volume_per_atom": "exact_formula_nsites_mean_volume_per_atom",
            "min_volume_per_atom": "exact_formula_nsites_min_volume_per_atom",
            "max_volume_per_atom": "exact_formula_nsites_max_volume_per_atom",
        }
    )[
        [
            "fractional_composition_json",
            "n_sites",
            "exact_formula_nsites_structure_count",
            "exact_formula_nsites_material_count",
            "exact_formula_nsites_mean_volume_per_atom",
            "exact_formula_nsites_min_volume_per_atom",
            "exact_formula_nsites_max_volume_per_atom",
        ]
    ].copy()
    exact_signature["exact_formula_nsites_in_reference"] = True

    wbm = pd.read_parquet(
        PROCESSED_DIR / "discovery_prediction_matrix_snapshot_45.parquet",
        columns=["material_id", "formula", "n_sites", "volume"],
    ).copy()
    if args.limit_materials is not None:
        wbm = wbm.head(args.limit_materials).copy()

    normalized = wbm["formula"].map(normalize_formula)
    wbm["reduced_formula"] = normalized.map(lambda item: item[0])
    wbm["fractional_composition_json"] = normalized.map(lambda item: item[1])
    wbm["n_elements"] = normalized.map(lambda item: item[2])
    wbm["n_sites"] = wbm["n_sites"].round().astype(int)
    wbm["volume_per_atom"] = wbm["volume"] / wbm["n_sites"]

    proxy = (
        wbm.merge(
            formula_summary,
            on=["fractional_composition_json", "reduced_formula", "n_elements"],
            how="left",
        )
        .merge(
            exact_signature,
            on=["fractional_composition_json", "n_sites"],
            how="left",
        )
        .sort_values("material_id")
        .reset_index(drop=True)
    )

    proxy["exact_formula_in_reference"] = proxy["exact_formula_in_reference"].fillna(False).astype(bool)
    proxy["exact_formula_nsites_in_reference"] = proxy["exact_formula_nsites_in_reference"].fillna(False).astype(bool)
    count_columns = [
        "exact_formula_signature_count",
        "exact_formula_total_structure_count",
        "exact_formula_total_material_count",
        "exact_formula_nsites_structure_count",
        "exact_formula_nsites_material_count",
    ]
    for column in count_columns:
        proxy[column] = proxy[column].fillna(0).astype(int)

    proxy["exact_formula_nsites_mean_volume_per_atom"] = proxy["exact_formula_nsites_mean_volume_per_atom"].astype(float)
    proxy["exact_formula_nsites_min_volume_per_atom"] = proxy["exact_formula_nsites_min_volume_per_atom"].astype(float)
    proxy["exact_formula_nsites_max_volume_per_atom"] = proxy["exact_formula_nsites_max_volume_per_atom"].astype(float)

    n_rows = len(proxy)
    min_abs_nsites_diff_exact_formula = np.full(n_rows, np.nan, dtype=float)
    nearest_nsites_exact_formula = np.full(n_rows, np.nan, dtype=float)
    min_abs_volume_per_atom_diff_exact_formula_mean = np.full(n_rows, np.nan, dtype=float)
    min_volume_range_distance_exact_formula = np.full(n_rows, np.nan, dtype=float)
    exact_formula_volume_in_reference_range = np.zeros(n_rows, dtype=bool)
    min_abs_volume_per_atom_diff_exact_formula_nsites_mean = np.full(n_rows, np.nan, dtype=float)
    min_volume_range_distance_exact_formula_nsites = np.full(n_rows, np.nan, dtype=float)
    exact_formula_nsites_volume_in_reference_range = np.zeros(n_rows, dtype=bool)

    reference_groups = {
        fraction_key: subset.reset_index(drop=True)
        for fraction_key, subset in reference.groupby("fractional_composition_json", sort=False)
    }

    grouped_iter = proxy.groupby("fractional_composition_json", sort=True)
    if not args.no_progress:
        grouped_iter = tqdm(
            grouped_iter,
            total=proxy["fractional_composition_json"].nunique(),
            desc="Computing WBM structure overlap proxy",
        )

    for fraction_key, subset in grouped_iter:
        reference_subset = reference_groups.get(fraction_key)
        if reference_subset is None:
            continue

        row_indices = subset.index.to_numpy(dtype=int)
        query_n_sites = subset["n_sites"].to_numpy(dtype=int)
        query_vpa = subset["volume_per_atom"].to_numpy(dtype=float)

        ref_n_sites = reference_subset["n_sites"].to_numpy(dtype=int)
        ref_mean_vpa = reference_subset["mean_volume_per_atom"].to_numpy(dtype=float)
        ref_min_vpa = reference_subset["min_volume_per_atom"].to_numpy(dtype=float)
        ref_max_vpa = reference_subset["max_volume_per_atom"].to_numpy(dtype=float)

        n_site_differences = np.abs(query_n_sites[:, None] - ref_n_sites[None, :])
        nearest_indices = n_site_differences.argmin(axis=1)
        min_abs_nsites_diff_exact_formula[row_indices] = n_site_differences[np.arange(len(subset)), nearest_indices]
        nearest_nsites_exact_formula[row_indices] = ref_n_sites[nearest_indices]

        volume_mean_differences = np.abs(query_vpa[:, None] - ref_mean_vpa[None, :])
        min_abs_volume_per_atom_diff_exact_formula_mean[row_indices] = volume_mean_differences.min(axis=1)

        volume_range_differences = distance_to_reference_range(
            query_vpa[:, None],
            ref_min_vpa[None, :],
            ref_max_vpa[None, :],
        )
        min_volume_range_distance_exact_formula[row_indices] = volume_range_differences.min(axis=1)
        exact_formula_volume_in_reference_range[row_indices] = (
            min_volume_range_distance_exact_formula[row_indices] == 0
        )

        for n_sites_value in np.unique(query_n_sites):
            query_mask = query_n_sites == n_sites_value
            reference_mask = ref_n_sites == n_sites_value
            if not reference_mask.any():
                continue

            subset_row_indices = row_indices[query_mask]
            subset_query_vpa = query_vpa[query_mask]
            nsites_mean_diff = np.abs(subset_query_vpa[:, None] - ref_mean_vpa[None, reference_mask]).min(axis=1)
            nsites_range_diff = distance_to_reference_range(
                subset_query_vpa[:, None],
                ref_min_vpa[None, reference_mask],
                ref_max_vpa[None, reference_mask],
            ).min(axis=1)

            min_abs_volume_per_atom_diff_exact_formula_nsites_mean[subset_row_indices] = nsites_mean_diff
            min_volume_range_distance_exact_formula_nsites[subset_row_indices] = nsites_range_diff
            exact_formula_nsites_volume_in_reference_range[subset_row_indices] = nsites_range_diff == 0

    proxy["min_abs_nsites_diff_exact_formula"] = min_abs_nsites_diff_exact_formula
    proxy["nearest_nsites_exact_formula"] = nearest_nsites_exact_formula
    proxy["min_abs_volume_per_atom_diff_exact_formula_mean"] = min_abs_volume_per_atom_diff_exact_formula_mean
    proxy["min_volume_range_distance_exact_formula"] = min_volume_range_distance_exact_formula
    proxy["exact_formula_volume_in_reference_range"] = exact_formula_volume_in_reference_range
    proxy["min_abs_volume_per_atom_diff_exact_formula_nsites_mean"] = (
        min_abs_volume_per_atom_diff_exact_formula_nsites_mean
    )
    proxy["min_volume_range_distance_exact_formula_nsites"] = min_volume_range_distance_exact_formula_nsites
    proxy["exact_formula_nsites_volume_in_reference_range"] = exact_formula_nsites_volume_in_reference_range

    summary = {
        "reference_index": str(reference_index_path),
        "output_prefix": args.output_prefix,
        "n_wbm_materials": int(len(proxy)),
        "n_wbm_unique_formula_nsites": int(proxy[["fractional_composition_json", "n_sites"]].drop_duplicates().shape[0]),
        "fraction_exact_formula_in_reference": float(proxy["exact_formula_in_reference"].mean()),
        "fraction_exact_formula_nsites_in_reference": float(proxy["exact_formula_nsites_in_reference"].mean()),
        "fraction_exact_formula_volume_in_reference_range": float(proxy["exact_formula_volume_in_reference_range"].mean()),
        "fraction_exact_formula_nsites_volume_in_reference_range": float(
            proxy["exact_formula_nsites_volume_in_reference_range"].mean()
        ),
        "median_min_abs_nsites_diff_exact_formula": float(
            proxy.loc[proxy["exact_formula_in_reference"], "min_abs_nsites_diff_exact_formula"].median()
        ),
        "median_min_volume_range_distance_exact_formula": float(
            proxy.loc[proxy["exact_formula_in_reference"], "min_volume_range_distance_exact_formula"].median()
        ),
        "median_min_volume_range_distance_exact_formula_nsites": float(
            proxy.loc[
                proxy["exact_formula_nsites_in_reference"],
                "min_volume_range_distance_exact_formula_nsites",
            ].median()
        ),
    }

    suffix = f"_sample_{args.limit_materials}" if args.limit_materials is not None else ""
    proxy.to_parquet(PROCESSED_DIR / f"{args.output_prefix}_material_structure_overlap_proxy{suffix}.parquet", index=False)
    proxy.to_csv(PROCESSED_DIR / f"{args.output_prefix}_material_structure_overlap_proxy{suffix}.csv", index=False)
    (PROCESSED_DIR / f"{args.output_prefix}_structure_overlap_proxy_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

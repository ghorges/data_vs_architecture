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
def normalize_formula(formula: str) -> tuple[str, str, str, int]:
    composition = Composition(formula).fractional_composition
    fraction_map = {str(el): float(amount) for el, amount in composition.as_dict().items()}
    reduced_formula = Composition(fraction_map).reduced_formula
    element_set_key = "|".join(sorted(fraction_map))
    return (
        reduced_formula,
        element_set_key,
        json.dumps(fraction_map, sort_keys=True),
        len(fraction_map),
    )


def parse_fraction_json(payload: str) -> dict[str, float]:
    return {str(key): float(value) for key, value in json.loads(payload).items()}


def build_formula_groups(frame: pd.DataFrame) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for element_set_key, subset in frame.groupby("element_set_key"):
        formulas = subset["reduced_formula"].to_numpy(dtype=object)
        fraction_maps = [parse_fraction_json(value) for value in subset["fractional_composition_json"]]
        elements = element_set_key.split("|")
        matrix = np.array(
            [[fraction_map.get(element, 0.0) for element in elements] for fraction_map in fraction_maps],
            dtype=float,
        )
        groups[element_set_key] = {
            "elements": elements,
            "formulas": formulas,
            "matrix": matrix,
            "n_formulas": len(formulas),
        }
    return groups


def compute_formula_proxy(
    wbm_formula_frame: pd.DataFrame,
    mptrj_formula_frame: pd.DataFrame,
    show_progress: bool = True,
) -> pd.DataFrame:
    mptrj_formula_key_set = set(mptrj_formula_frame["fractional_composition_json"])
    mptrj_groups = build_formula_groups(mptrj_formula_frame)

    unique_wbm = (
        wbm_formula_frame[["reduced_formula", "element_set_key", "fractional_composition_json", "n_elements"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    rows: list[dict] = []
    grouped_iter = unique_wbm.groupby("element_set_key", sort=True)
    if show_progress:
        grouped_iter = tqdm(
            grouped_iter,
            total=unique_wbm["element_set_key"].nunique(),
            desc="Computing WBM->MPtrj coverage proxy",
        )

    for element_set_key, subset in grouped_iter:
        ref_group = mptrj_groups.get(element_set_key)
        query_maps = [parse_fraction_json(value) for value in subset["fractional_composition_json"]]
        query_elements = element_set_key.split("|")
        query_matrix = np.array(
            [[fraction_map.get(element, 0.0) for element in query_elements] for fraction_map in query_maps],
            dtype=float,
        )

        nearest_distance = np.full(len(subset), np.nan, dtype=float)
        nearest_formula = np.array([""] * len(subset), dtype=object)
        same_element_set_count = 0

        if ref_group is not None:
            ref_matrix = ref_group["matrix"]
            l1_distances = np.abs(query_matrix[:, None, :] - ref_matrix[None, :, :]).sum(axis=2)
            nearest_indices = l1_distances.argmin(axis=1)
            nearest_distance = l1_distances[np.arange(len(subset)), nearest_indices]
            nearest_formula = ref_group["formulas"][nearest_indices]
            same_element_set_count = ref_group["n_formulas"]

        for idx, row in enumerate(subset.itertuples(index=False)):
            exact_formula_hit = row.fractional_composition_json in mptrj_formula_key_set
            rows.append(
                {
                    "reduced_formula": row.reduced_formula,
                    "n_elements": int(row.n_elements),
                    "element_set_key": element_set_key,
                    "fractional_composition_json": row.fractional_composition_json,
                    "exact_formula_in_mptrj": exact_formula_hit,
                    "same_element_set_in_mptrj": same_element_set_count > 0,
                    "same_element_set_formula_count": int(same_element_set_count),
                    "nearest_same_element_formula": row.reduced_formula if exact_formula_hit else str(nearest_formula[idx]),
                    "min_l1_distance_same_element_set": 0.0 if exact_formula_hit else float(nearest_distance[idx]),
                }
            )

    return pd.DataFrame(rows).sort_values("reduced_formula").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build WBM-to-MPtrj composition coverage proxies.",
    )
    parser.add_argument(
        "--reference-index",
        default=str(PROCESSED_DIR / "mptrj_formula_index.parquet"),
        help="Processed formula-index parquet used as the training reference set.",
    )
    parser.add_argument(
        "--output-prefix",
        default="wbm_mptrj",
        help="Prefix for the generated processed output files.",
    )
    parser.add_argument(
        "--limit-formulas",
        type=int,
        default=None,
        help="Optional cap on the number of WBM rows for debugging.",
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
    mptrj_formula_index = pd.read_parquet(reference_index_path)
    mptrj_formula_index = mptrj_formula_index.copy()
    mptrj_formula_index["element_set_key"] = mptrj_formula_index["fractional_composition_json"].apply(
        lambda value: "|".join(sorted(json.loads(value).keys()))
    )

    wbm_formula_frame = pd.read_parquet(
        PROCESSED_DIR / "discovery_prediction_matrix_snapshot_45.parquet",
        columns=["material_id", "formula"],
    ).copy()
    if args.limit_formulas is not None:
        wbm_formula_frame = wbm_formula_frame.head(args.limit_formulas).copy()
    normalized = wbm_formula_frame["formula"].map(normalize_formula)
    wbm_formula_frame["reduced_formula"] = normalized.map(lambda item: item[0])
    wbm_formula_frame["element_set_key"] = normalized.map(lambda item: item[1])
    wbm_formula_frame["fractional_composition_json"] = normalized.map(lambda item: item[2])
    wbm_formula_frame["n_elements"] = normalized.map(lambda item: item[3])

    formula_proxy = compute_formula_proxy(
        wbm_formula_frame=wbm_formula_frame,
        mptrj_formula_frame=mptrj_formula_index,
        show_progress=not args.no_progress,
    )
    material_proxy = wbm_formula_frame[
        ["material_id", "formula", "reduced_formula", "fractional_composition_json"]
    ].merge(
        formula_proxy,
        on=["reduced_formula", "fractional_composition_json"],
        how="left",
    )

    summary = {
        "reference_index": str(reference_index_path),
        "output_prefix": args.output_prefix,
        "n_wbm_materials": int(len(material_proxy)),
        "n_wbm_unique_reduced_formulas": int(formula_proxy["reduced_formula"].nunique()),
        "fraction_exact_formula_in_reference": float(formula_proxy["exact_formula_in_mptrj"].mean()),
        "fraction_same_element_set_in_reference": float(formula_proxy["same_element_set_in_mptrj"].mean()),
        "median_min_l1_distance_same_element_set": float(
            formula_proxy.loc[
                formula_proxy["same_element_set_in_mptrj"],
                "min_l1_distance_same_element_set",
            ].median()
        ),
    }

    suffix = f"_sample_{args.limit_formulas}" if args.limit_formulas is not None else ""

    formula_proxy.to_csv(PROCESSED_DIR / f"{args.output_prefix}_formula_coverage_proxy{suffix}.csv", index=False)
    formula_proxy.to_parquet(
        PROCESSED_DIR / f"{args.output_prefix}_formula_coverage_proxy{suffix}.parquet",
        index=False,
    )
    material_proxy.to_parquet(
        PROCESSED_DIR / f"{args.output_prefix}_material_coverage_proxy{suffix}.parquet",
        index=False,
    )
    (PROCESSED_DIR / f"{args.output_prefix}_coverage_proxy_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

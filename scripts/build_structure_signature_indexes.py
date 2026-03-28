from __future__ import annotations

import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import ijson
import pandas as pd
from pymatgen.core import Composition
from tqdm import tqdm

from dva_project.settings import PROCESSED_DIR, RAW_DIR
from dva_project.utils import ensure_dir


MPTRJ_SOURCE = RAW_DIR / "training_refs" / "MPtrj_2022.9_full.json"
MP2022_SOURCE = RAW_DIR / "training_refs" / "2023-02-07-mp-computed-structure-entries.json.gz"


def normalize_composition_from_sites(sites: list[dict]) -> tuple[str, str, str, int]:
    composition_dict: dict[str, float] = defaultdict(float)
    for site in sites:
        for species in site.get("species", []):
            composition_dict[str(species["element"])] += float(species.get("occu", 1.0))

    composition = Composition(dict(composition_dict)).fractional_composition
    fraction_map = {str(el): float(amount) for el, amount in composition.as_dict().items()}
    reduced_formula = Composition(fraction_map).reduced_formula
    anonymous_formula = Composition(fraction_map).anonymized_formula
    fraction_json = json.dumps(fraction_map, sort_keys=True)
    return reduced_formula, anonymous_formula, fraction_json, len(fraction_map)


def normalize_composition_dict(composition_dict: dict[str, float]) -> tuple[str, str, str, int]:
    cleaned = {str(key): float(value) for key, value in composition_dict.items() if float(value) > 0}
    composition = Composition(cleaned).fractional_composition
    fraction_map = {str(el): float(amount) for el, amount in composition.as_dict().items()}
    reduced_formula = Composition(fraction_map).reduced_formula
    anonymous_formula = Composition(fraction_map).anonymized_formula
    fraction_json = json.dumps(fraction_map, sort_keys=True)
    return reduced_formula, anonymous_formula, fraction_json, len(fraction_map)


def aggregate_rows(rows: list[dict], dataset_label: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["source_dataset"] = dataset_label
    return (
        frame.groupby(
            ["fractional_composition_json", "n_sites"],
            as_index=False,
        )
        .agg(
            reduced_formula=("reduced_formula", "first"),
            anonymous_formula=("anonymous_formula", "first"),
            n_elements=("n_elements", "first"),
            n_structures=("n_structures", "sum"),
            n_materials=("n_materials", "sum"),
            mean_volume_per_atom=("weighted_volume_sum", lambda values: float(sum(values))),
            min_volume_per_atom=("min_volume_per_atom", "min"),
            max_volume_per_atom=("max_volume_per_atom", "max"),
            source_dataset=("source_dataset", "first"),
        )
        .assign(mean_volume_per_atom=lambda frame_: frame_["mean_volume_per_atom"] / frame_["n_structures"])
        .sort_values(["n_structures", "n_materials", "reduced_formula", "n_sites"], ascending=[False, False, True, True])
        .reset_index(drop=True)
    )


def build_mptrj_index() -> pd.DataFrame:
    if not MPTRJ_SOURCE.exists():
        raise FileNotFoundError(f"Missing cached MPtrj source: {MPTRJ_SOURCE}")

    aggregate: dict[tuple[str, int], dict] = {}
    with MPTRJ_SOURCE.open("rb") as handle:
        for mp_id, task_map in tqdm(ijson.kvitems(handle, ""), desc="Parsing MPtrj structures"):
            material_keys_seen: set[tuple[str, int]] = set()
            if not isinstance(task_map, dict):
                continue
            for task_payload in task_map.values():
                structure = task_payload.get("structure")
                if not isinstance(structure, dict):
                    continue
                sites = structure.get("sites", [])
                n_sites = len(sites)
                if n_sites == 0:
                    continue
                reduced_formula, anonymous_formula, fraction_json, n_elements = normalize_composition_from_sites(sites)
                lattice = structure.get("lattice", {})
                volume = lattice.get("volume")
                volume_per_atom = float(volume) / n_sites if volume is not None else float("nan")
                key = (fraction_json, n_sites)
                if key not in aggregate:
                    aggregate[key] = {
                        "fractional_composition_json": fraction_json,
                        "reduced_formula": reduced_formula,
                        "anonymous_formula": anonymous_formula,
                        "n_elements": n_elements,
                        "n_sites": n_sites,
                        "n_structures": 0,
                        "n_materials": 0,
                        "weighted_volume_sum": 0.0,
                        "min_volume_per_atom": volume_per_atom,
                        "max_volume_per_atom": volume_per_atom,
                    }
                record = aggregate[key]
                record["n_structures"] += 1
                if key not in material_keys_seen:
                    record["n_materials"] += 1
                    material_keys_seen.add(key)
                if pd.notna(volume_per_atom):
                    record["weighted_volume_sum"] += volume_per_atom
                    record["min_volume_per_atom"] = min(record["min_volume_per_atom"], volume_per_atom)
                    record["max_volume_per_atom"] = max(record["max_volume_per_atom"], volume_per_atom)
    return aggregate_rows(list(aggregate.values()), "mptrj")


def build_mp2022_index() -> pd.DataFrame:
    if not MP2022_SOURCE.exists():
        raise FileNotFoundError(f"Missing cached MP 2022 source: {MP2022_SOURCE}")

    frame = pd.read_json(MP2022_SOURCE, compression="gzip")
    aggregate: dict[tuple[str, int], dict] = {}
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="Parsing MP 2022 structures"):
        entry = row.entry
        if not isinstance(entry, dict):
            continue
        composition_dict = entry.get("composition")
        structure = entry.get("structure")
        if not isinstance(composition_dict, dict) or not isinstance(structure, dict):
            continue
        sites = structure.get("sites", [])
        n_sites = len(sites)
        if n_sites == 0:
            continue
        reduced_formula, anonymous_formula, fraction_json, n_elements = normalize_composition_dict(composition_dict)
        lattice = structure.get("lattice", {})
        volume = lattice.get("volume")
        volume_per_atom = float(volume) / n_sites if volume is not None else float("nan")
        key = (fraction_json, n_sites)
        if key not in aggregate:
            aggregate[key] = {
                "fractional_composition_json": fraction_json,
                "reduced_formula": reduced_formula,
                "anonymous_formula": anonymous_formula,
                "n_elements": n_elements,
                "n_sites": n_sites,
                "n_structures": 0,
                "n_materials": 0,
                "weighted_volume_sum": 0.0,
                "min_volume_per_atom": volume_per_atom,
                "max_volume_per_atom": volume_per_atom,
            }
        record = aggregate[key]
        record["n_structures"] += 1
        record["n_materials"] += 1
        if pd.notna(volume_per_atom):
            record["weighted_volume_sum"] += volume_per_atom
            record["min_volume_per_atom"] = min(record["min_volume_per_atom"], volume_per_atom)
            record["max_volume_per_atom"] = max(record["max_volume_per_atom"], volume_per_atom)
    return aggregate_rows(list(aggregate.values()), "mp2022")


def build_union_index(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    union = (
        combined.groupby(["fractional_composition_json", "n_sites"], as_index=False)
        .agg(
            reduced_formula=("reduced_formula", "first"),
            anonymous_formula=("anonymous_formula", "first"),
            n_elements=("n_elements", "first"),
            n_structures=("n_structures", "sum"),
            n_materials=("n_materials", "sum"),
            weighted_volume_sum=("mean_volume_per_atom", lambda values: 0.0),
            min_volume_per_atom=("min_volume_per_atom", "min"),
            max_volume_per_atom=("max_volume_per_atom", "max"),
            n_source_datasets=("source_dataset", "nunique"),
            source_dataset=("source_dataset", lambda values: "__".join(sorted(set(values)))),
        )
        .reset_index(drop=True)
    )
    weighted_means: list[float] = []
    for row in union.itertuples(index=False):
        mask = (
            (combined["fractional_composition_json"] == row.fractional_composition_json)
            & (combined["n_sites"] == row.n_sites)
        )
        subset = combined.loc[mask]
        weighted_means.append(float((subset["mean_volume_per_atom"] * subset["n_structures"]).sum() / subset["n_structures"].sum()))
    union["mean_volume_per_atom"] = weighted_means
    union = union.drop(columns=["weighted_volume_sum"])
    return union.sort_values(
        ["n_source_datasets", "n_structures", "n_materials", "reduced_formula", "n_sites"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)


def write_outputs(frame: pd.DataFrame, prefix: str) -> None:
    frame.to_csv(PROCESSED_DIR / f"{prefix}_structure_signature_index.csv", index=False)
    frame.to_parquet(PROCESSED_DIR / f"{prefix}_structure_signature_index.parquet", index=False)
    summary = {
        "prefix": prefix,
        "n_rows": int(len(frame)),
        "n_unique_formulas": int(frame["fractional_composition_json"].nunique()),
        "n_unique_formula_nsites_signatures": int(len(frame)),
        "top_rows": frame.head(10)[
            [
                "reduced_formula",
                "n_sites",
                "n_structures",
                "n_materials",
                "source_dataset" if "source_dataset" in frame.columns else "source_dataset",
            ]
        ].to_dict(orient="records"),
    }
    (PROCESSED_DIR / f"{prefix}_structure_signature_index_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dir(PROCESSED_DIR)
    mptrj = build_mptrj_index()
    mp2022 = build_mp2022_index()
    union = build_union_index([mptrj, mp2022])

    write_outputs(mptrj, "mptrj")
    write_outputs(mp2022, "mp2022")
    write_outputs(union, "mptrj_mp2022_union")


if __name__ == "__main__":
    main()

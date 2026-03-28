from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import ijson
import pandas as pd
from pymatgen.core import Composition

from dva_project.settings import MANIFEST_DIR, PROCESSED_DIR, RAW_DIR
from dva_project.utils import download_file, ensure_dir, load_yaml


DATASET_NAME = "MPtrj"
DEFAULT_LOCAL_JSON_PATH = RAW_DIR / "training_refs" / "MPtrj_2022.9_full.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact reduced-formula index for the MPtrj training dataset."
    )
    parser.add_argument(
        "--limit-mp-ids",
        type=int,
        default=None,
        help="Only parse the first N top-level MP ids for validation/debugging.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Network timeout in seconds.",
    )
    parser.add_argument(
        "--download-json",
        action="store_true",
        help="Download and cache the full raw MPtrj JSON locally before parsing.",
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        default=DEFAULT_LOCAL_JSON_PATH,
        help="Local path of the cached MPtrj JSON file.",
    )
    return parser.parse_args()


def get_mptrj_url() -> str:
    datasets = load_yaml(MANIFEST_DIR / "datasets.yml")
    url = datasets[DATASET_NAME]["download_url"]
    match = re.search(r"/files/(\d+)", url)
    if match:
        return f"https://ndownloader.figshare.com/files/{match.group(1)}"
    return url


def structure_to_formula(structure_payload: dict) -> tuple[str, str, int]:
    composition_dict: dict[str, float] = {}
    for site in structure_payload.get("sites", []):
        for species in site.get("species", []):
            element = species["element"]
            occupancy = float(species.get("occu", 1.0))
            composition_dict[element] = composition_dict.get(element, 0.0) + occupancy

    composition = Composition(composition_dict)
    reduced_formula = composition.reduced_formula
    fractional = composition.fractional_composition
    fraction_json = json.dumps(
        {str(el): float(amount) for el, amount in fractional.as_dict().items()},
        sort_keys=True,
    )
    return reduced_formula, fraction_json, len(composition.elements)


def ensure_local_source(source_file: Path, timeout: int, download_json: bool) -> Path:
    ensure_dir(source_file.parent)
    if source_file.exists() and source_file.stat().st_size > 0:
        return source_file
    if source_file.exists() and source_file.stat().st_size == 0:
        source_file.unlink()
    if not download_json:
        raise FileNotFoundError(
            f"MPtrj source file not found at {source_file}. Re-run with --download-json."
        )
    download_file(
        get_mptrj_url(),
        source_file,
        overwrite=False,
        timeout=timeout,
        retries=3,
    )
    return source_file


def build_formula_index(
    source_file: Path,
    limit_mp_ids: int | None,
) -> tuple[pd.DataFrame, dict]:
    url = get_mptrj_url()
    formula_counts: Counter[str] = Counter()
    formula_mp_counts: Counter[str] = Counter()
    formula_payloads: dict[str, dict] = {}

    n_mp_ids = 0
    n_structures = 0

    with source_file.open("rb") as handle:
        for mp_id, task_map in ijson.kvitems(handle, ""):
            n_mp_ids += 1
            formulas_seen_for_mp: set[str] = set()

            if isinstance(task_map, dict):
                for task_payload in task_map.values():
                    structure_payload = task_payload.get("structure")
                    if not isinstance(structure_payload, dict):
                        continue
                    reduced_formula, fraction_json, n_elements = structure_to_formula(structure_payload)
                    formula_counts[reduced_formula] += 1
                    if reduced_formula not in formulas_seen_for_mp:
                        formula_mp_counts[reduced_formula] += 1
                        formulas_seen_for_mp.add(reduced_formula)
                    if reduced_formula not in formula_payloads:
                        formula_payloads[reduced_formula] = {
                            "fractional_composition_json": fraction_json,
                            "n_elements": n_elements,
                        }
                    n_structures += 1

            if limit_mp_ids is not None and n_mp_ids >= limit_mp_ids:
                break

    rows: list[dict] = []
    for reduced_formula, n_formula_structures in formula_counts.items():
        rows.append(
            {
                "reduced_formula": reduced_formula,
                "n_structures": int(n_formula_structures),
                "n_mp_ids": int(formula_mp_counts[reduced_formula]),
                "fractional_composition_json": formula_payloads[reduced_formula][
                    "fractional_composition_json"
                ],
                "n_elements": int(formula_payloads[reduced_formula]["n_elements"]),
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        ["n_structures", "n_mp_ids", "reduced_formula"],
        ascending=[False, False, True],
    )
    summary = {
        "dataset": DATASET_NAME,
        "source_url": url,
        "source_file": str(source_file),
        "limit_mp_ids": limit_mp_ids,
        "n_mp_ids_processed": n_mp_ids,
        "n_structures_processed": n_structures,
        "n_unique_reduced_formulas": int(len(frame)),
        "top_formulas": frame.head(10)[["reduced_formula", "n_structures", "n_mp_ids"]].to_dict(
            orient="records"
        ),
    }
    return frame, summary


def main() -> None:
    args = parse_args()
    output_dir = PROCESSED_DIR
    ensure_dir(output_dir)
    source_file = ensure_local_source(
        source_file=args.source_file,
        timeout=args.timeout,
        download_json=args.download_json,
    )

    frame, summary = build_formula_index(
        source_file=source_file,
        limit_mp_ids=args.limit_mp_ids,
    )

    suffix = f"_sample_{args.limit_mp_ids}" if args.limit_mp_ids is not None else ""
    frame.to_csv(output_dir / f"mptrj_formula_index{suffix}.csv", index=False)
    frame.to_parquet(output_dir / f"mptrj_formula_index{suffix}.parquet", index=False)
    (output_dir / f"mptrj_formula_index_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

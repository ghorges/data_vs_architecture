from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

from dva_project.settings import MANIFEST_DIR, PROCESSED_DIR, RAW_DIR
from dva_project.utils import download_file, ensure_dir, load_yaml


DATASET_NAME = "MP 2022"
DEFAULT_LOCAL_GZ_PATH = RAW_DIR / "training_refs" / "2023-02-07-mp-computed-structure-entries.json.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact reduced-formula index for the MP 2022 training dataset."
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Only parse the first N rows for validation/debugging.",
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
        help="Download and cache the gzipped MP 2022 JSON locally before parsing.",
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        default=DEFAULT_LOCAL_GZ_PATH,
        help="Local path of the cached MP 2022 JSON.gz file.",
    )
    return parser.parse_args()


def get_mp2022_url() -> str:
    datasets = load_yaml(MANIFEST_DIR / "datasets.yml")
    url = datasets[DATASET_NAME]["download_url"]
    match = re.search(r"/files/(\d+)", url)
    if match:
        return f"https://ndownloader.figshare.com/files/{match.group(1)}"
    return url


def composition_dict_to_formula(composition_dict: dict[str, float]) -> tuple[str, str, int]:
    cleaned = {str(key): float(value) for key, value in composition_dict.items() if float(value) > 0}
    composition = Composition(cleaned)
    fractional = composition.fractional_composition
    reduced_formula = Composition(fractional.as_dict()).reduced_formula
    fraction_json = json.dumps(
        {str(el): float(amount) for el, amount in fractional.as_dict().items()},
        sort_keys=True,
    )
    return reduced_formula, fraction_json, len(cleaned)


def ensure_local_source(source_file: Path, timeout: int, download_json: bool) -> Path:
    ensure_dir(source_file.parent)
    if source_file.exists() and source_file.stat().st_size > 0:
        return source_file
    if source_file.exists() and source_file.stat().st_size == 0:
        source_file.unlink()
    if not download_json:
        raise FileNotFoundError(
            f"MP 2022 source file not found at {source_file}. Re-run with --download-json."
        )
    download_file(
        get_mp2022_url(),
        source_file,
        overwrite=False,
        timeout=timeout,
        retries=3,
    )
    return source_file


def build_formula_index(
    source_file: Path,
    limit_rows: int | None,
) -> tuple[pd.DataFrame, dict]:
    url = get_mp2022_url()
    frame = pd.read_json(source_file, compression="gzip")
    if limit_rows is not None:
        frame = frame.head(limit_rows).copy()

    formula_counts: Counter[str] = Counter()
    formula_material_counts: Counter[str] = Counter()
    formula_payloads: dict[str, dict] = {}

    n_rows = 0
    n_material_ids = 0
    for row in frame.itertuples(index=False):
        entry = row.entry
        if not isinstance(entry, dict):
            continue
        composition_dict = entry.get("composition")
        if not isinstance(composition_dict, dict):
            continue

        reduced_formula, fraction_json, n_elements = composition_dict_to_formula(composition_dict)
        formula_counts[reduced_formula] += 1
        formula_material_counts[reduced_formula] += 1
        if reduced_formula not in formula_payloads:
            formula_payloads[reduced_formula] = {
                "fractional_composition_json": fraction_json,
                "n_elements": n_elements,
            }
        n_rows += 1
        if isinstance(row.material_id, str) and row.material_id:
            n_material_ids += 1

    rows: list[dict] = []
    for reduced_formula, n_formula_rows in formula_counts.items():
        rows.append(
            {
                "reduced_formula": reduced_formula,
                "n_structures": int(n_formula_rows),
                "n_mp_ids": int(formula_material_counts[reduced_formula]),
                "fractional_composition_json": formula_payloads[reduced_formula][
                    "fractional_composition_json"
                ],
                "n_elements": int(formula_payloads[reduced_formula]["n_elements"]),
            }
        )

    formula_index = pd.DataFrame(rows).sort_values(
        ["n_structures", "n_mp_ids", "reduced_formula"],
        ascending=[False, False, True],
    )
    summary = {
        "dataset": DATASET_NAME,
        "source_url": url,
        "source_file": str(source_file),
        "limit_rows": limit_rows,
        "n_rows_processed": n_rows,
        "n_material_ids_processed": n_material_ids,
        "n_unique_reduced_formulas": int(len(formula_index)),
        "top_formulas": formula_index.head(10)[["reduced_formula", "n_structures", "n_mp_ids"]].to_dict(
            orient="records"
        ),
    }
    return formula_index, summary


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
        limit_rows=args.limit_rows,
    )

    suffix = f"_sample_{args.limit_rows}" if args.limit_rows is not None else ""
    frame.to_csv(output_dir / f"mp2022_formula_index{suffix}.csv", index=False)
    frame.to_parquet(output_dir / f"mp2022_formula_index{suffix}.parquet", index=False)
    (output_dir / f"mp2022_formula_index_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

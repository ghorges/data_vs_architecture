from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dva_project.settings import PROCESSED_DIR
from dva_project.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple processed formula-index tables into a union reference index."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            str(PROCESSED_DIR / "mptrj_formula_index.parquet"),
            str(PROCESSED_DIR / "mp2022_formula_index.parquet"),
        ],
        help="Input parquet formula indexes to combine.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional source labels matching --inputs.",
    )
    parser.add_argument(
        "--output-prefix",
        default="mptrj_mp2022_union",
        help="Prefix for the generated union index files.",
    )
    return parser.parse_args()


def infer_labels(inputs: list[str], labels: list[str] | None) -> list[str]:
    if labels is None:
        return [Path(path).stem.replace("_formula_index", "") for path in inputs]
    if len(labels) != len(inputs):
        raise ValueError("--labels must have the same length as --inputs.")
    return labels


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    labels = infer_labels(args.inputs, args.labels)
    ensure_dir(PROCESSED_DIR)

    frames: list[pd.DataFrame] = []
    for path, label in zip(input_paths, labels, strict=True):
        frame = pd.read_parquet(path).copy()
        frame["source_dataset"] = label
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    union_index = (
        combined.groupby("fractional_composition_json", as_index=False)
        .agg(
            reduced_formula=("reduced_formula", "first"),
            n_structures=("n_structures", "sum"),
            n_mp_ids=("n_mp_ids", "sum"),
            n_elements=("n_elements", "first"),
            n_source_datasets=("source_dataset", "nunique"),
            source_datasets=("source_dataset", lambda values: "__".join(sorted(set(values)))),
        )
        .sort_values(
            ["n_source_datasets", "n_structures", "n_mp_ids", "reduced_formula"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )

    summary = {
        "output_prefix": args.output_prefix,
        "input_paths": [str(path) for path in input_paths],
        "source_labels": labels,
        "n_input_rows": int(len(combined)),
        "n_unique_union_formulas": int(len(union_index)),
        "n_formulas_present_in_multiple_sources": int((union_index["n_source_datasets"] > 1).sum()),
        "top_formulas": union_index.head(10)[
            ["reduced_formula", "n_structures", "n_mp_ids", "source_datasets"]
        ].to_dict(orient="records"),
    }

    union_index.to_csv(PROCESSED_DIR / f"{args.output_prefix}_formula_index.csv", index=False)
    union_index.to_parquet(PROCESSED_DIR / f"{args.output_prefix}_formula_index.parquet", index=False)
    (PROCESSED_DIR / f"{args.output_prefix}_formula_index_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from dva_project.model_inventory import MODEL_YAML_PATHS
from dva_project.settings import (
    DATASETS_YAML_NAME,
    DISCOVERY_PREDICTIONS_DIR,
    MANIFEST_DIR,
    MATBENCH_DISCOVERY_RAW_BASE_URL,
    MODEL_YAML_DIR,
    WBM_DIR,
    WBM_SUMMARY_FILE_NAME,
)
from dva_project.utils import download_file, ensure_dir, load_yaml, make_local_prediction_name


def normalize_figshare_download_url(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"/files/(\d+)", url)
    if match:
        return f"https://ndownloader.figshare.com/files/{match.group(1)}"
    return url


def download_raw_reference(relative_path: str, destination: Path, overwrite: bool = False) -> Path:
    url = f"{MATBENCH_DISCOVERY_RAW_BASE_URL}/{relative_path}"
    return download_file(url, destination, overwrite=overwrite)


def collect_discovery_model_specs(overwrite: bool = False) -> list[dict]:
    specs: list[dict] = []
    for relative_path in MODEL_YAML_PATHS:
        local_yaml_path = MODEL_YAML_DIR / Path(relative_path).relative_to("models")
        download_raw_reference(relative_path, local_yaml_path, overwrite=overwrite)
        payload = load_yaml(local_yaml_path)
        discovery = payload.get("metrics", {}).get("discovery")
        if not isinstance(discovery, dict):
            continue
        prediction_name = Path(discovery["pred_file"]).name
        family = Path(relative_path).parts[1]
        specs.append(
            {
                "model_key": payload["model_key"],
                "family": family,
                "yaml_path": str(local_yaml_path.relative_to(MODEL_YAML_DIR.parent.parent)),
                "prediction_file": prediction_name,
                "prediction_column": discovery.get("pred_col"),
                "prediction_source_url": normalize_figshare_download_url(discovery.get("pred_file_url")),
            }
        )
    return sorted(specs, key=lambda item: item["model_key"])


def copy_static_reference_files(overwrite: bool = False) -> None:
    download_raw_reference(f"data/{DATASETS_YAML_NAME}", MANIFEST_DIR / DATASETS_YAML_NAME, overwrite=overwrite)
    download_raw_reference(
        f"data/wbm/{WBM_SUMMARY_FILE_NAME}",
        WBM_DIR / WBM_SUMMARY_FILE_NAME,
        overwrite=overwrite,
    )
    ensure_dir(MANIFEST_DIR)
    (MANIFEST_DIR / "matbench_discovery_source.txt").write_text(
        f"{MATBENCH_DISCOVERY_RAW_BASE_URL}\n",
        encoding="utf-8",
    )


def download_prediction_files(specs: list[dict], overwrite: bool = False, limit: int | None = None) -> None:
    selected_specs = specs[:limit] if limit else specs

    manifest_rows: list[dict] = []
    for spec in selected_specs:
        source_url = spec.get("prediction_source_url")
        if not source_url:
            raise KeyError(f"prediction source URL missing for {spec['model_key']}")
        local_prediction_file = make_local_prediction_name(spec["model_key"], spec["prediction_file"])
        destination = DISCOVERY_PREDICTIONS_DIR / local_prediction_file
        download_file(source_url, destination, overwrite=overwrite)
        manifest_rows.append(
            {
                **spec,
                "local_prediction_file": local_prediction_file,
                "download_url": source_url,
                "size_bytes": destination.stat().st_size,
                "local_prediction_path": str(destination.relative_to(destination.parents[3])),
            }
        )

    ensure_dir(MANIFEST_DIR)
    manifest_file = MANIFEST_DIR / "discovery_prediction_manifest.csv"
    with manifest_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download benchmark inputs and model predictions.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload files even if present.")
    parser.add_argument("--limit", type=int, default=None, help="Only download the first N model predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_static_reference_files(overwrite=args.overwrite)
    specs = collect_discovery_model_specs(overwrite=args.overwrite)
    download_prediction_files(specs, overwrite=args.overwrite, limit=args.limit)


if __name__ == "__main__":
    main()

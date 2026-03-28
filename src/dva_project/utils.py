from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterable

import requests
import yaml


REQUEST_HEADERS = {"User-Agent": "data-vs-architecture/0.1"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_if_missing(source: Path, destination: Path, overwrite: bool = False) -> Path:
    ensure_dir(destination.parent)
    if destination.exists() and not overwrite:
        return destination
    shutil.copy2(source, destination)
    return destination


def download_file(
    url: str,
    destination: Path,
    *,
    overwrite: bool = False,
    timeout: int = 120,
    retries: int = 3,
) -> Path:
    ensure_dir(destination.parent)
    if destination.exists() and not overwrite:
        return destination

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        tmp_path = None
        try:
            with requests.get(
                url,
                stream=True,
                timeout=timeout,
                headers=REQUEST_HEADERS,
            ) as response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    dir=destination.parent,
                    suffix=".tmp",
                ) as handle:
                    tmp_path = Path(handle.name)
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp_path.replace(destination)
            return destination
        except Exception as exc:  # pragma: no cover - network retries
            last_error = exc
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                break
            time.sleep(attempt)

    raise RuntimeError(f"failed to download {url}") from last_error


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def iter_model_descriptor_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".yml", ".yaml"}:
            continue
        if "training" in path.parts:
            continue
        yield path


def make_local_prediction_name(model_key: str, prediction_file: str) -> str:
    return f"{model_key}__{prediction_file}"

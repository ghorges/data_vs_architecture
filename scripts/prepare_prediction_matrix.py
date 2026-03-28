from __future__ import annotations

from pathlib import Path

import pandas as pd

from dva_project.settings import DISCOVERY_PREDICTIONS_DIR, PROCESSED_DIR, WBM_DIR, WBM_SUMMARY_FILE_NAME
from dva_project.utils import ensure_dir


BASE_WBM_COLUMNS = [
    "material_id",
    "formula",
    "n_sites",
    "volume",
    "e_form_per_atom_wbm",
    "e_above_hull_wbm",
    "e_form_per_atom_mp2020_corrected",
    "e_above_hull_mp2020_corrected_ppd_mp",
    "wyckoff_spglib",
    "unique_prototype",
]


def load_metadata() -> pd.DataFrame:
    metadata_path = PROCESSED_DIR / "model_metadata.csv"
    if not metadata_path.exists():
        raise RuntimeError("model metadata missing; run scripts/build_model_table.py first")
    return pd.read_csv(metadata_path)


def main() -> None:
    metadata = load_metadata()
    wbm = pd.read_csv(WBM_DIR / WBM_SUMMARY_FILE_NAME, usecols=BASE_WBM_COLUMNS)
    matrix = wbm.copy()

    for row in metadata.itertuples(index=False):
        local_prediction_file = getattr(row, "local_prediction_file", row.prediction_file)
        prediction_path = DISCOVERY_PREDICTIONS_DIR / local_prediction_file
        pred = pd.read_csv(prediction_path, usecols=["material_id", row.prediction_column])
        pred = pred.rename(columns={row.prediction_column: row.model_key})
        matrix = matrix.merge(pred, on="material_id", how="left", validate="one_to_one")

    error_matrix = matrix[["material_id"]].copy()
    for model_key in metadata["model_key"]:
        error_matrix[model_key] = matrix[model_key] - matrix["e_form_per_atom_wbm"]

    ensure_dir(PROCESSED_DIR)
    matrix.to_parquet(PROCESSED_DIR / "discovery_prediction_matrix.parquet", index=False)
    error_matrix.to_parquet(PROCESSED_DIR / "discovery_error_matrix.parquet", index=False)


if __name__ == "__main__":
    main()

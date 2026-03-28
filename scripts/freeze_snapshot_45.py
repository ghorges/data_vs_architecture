from __future__ import annotations

import json

import pandas as pd

from dva_project.settings import PROCESSED_DIR
from dva_project.utils import ensure_dir


SNAPSHOT_LABEL = "snapshot_45"
SNAPSHOT_CUTOFF = "2025-09-08"
BASE_COLUMNS = [
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


def main() -> None:
    ensure_dir(PROCESSED_DIR)

    metadata = pd.read_csv(PROCESSED_DIR / "model_metadata.csv", parse_dates=["date_added"])
    snapshot_meta = metadata.loc[metadata["date_added"] <= SNAPSHOT_CUTOFF].copy()
    snapshot_meta = snapshot_meta.sort_values("model_key").reset_index(drop=True)

    prediction_matrix = pd.read_parquet(PROCESSED_DIR / "discovery_prediction_matrix.parquet")
    error_matrix = pd.read_parquet(PROCESSED_DIR / "discovery_error_matrix.parquet")

    model_columns = snapshot_meta["model_key"].tolist()
    snapshot_prediction_matrix = prediction_matrix[BASE_COLUMNS + model_columns].copy()
    snapshot_error_matrix = error_matrix[["material_id"] + model_columns].copy()

    snapshot_meta.to_csv(PROCESSED_DIR / f"model_metadata_{SNAPSHOT_LABEL}.csv", index=False)
    snapshot_meta.to_parquet(PROCESSED_DIR / f"model_metadata_{SNAPSHOT_LABEL}.parquet", index=False)
    snapshot_prediction_matrix.to_parquet(
        PROCESSED_DIR / f"discovery_prediction_matrix_{SNAPSHOT_LABEL}.parquet",
        index=False,
    )
    snapshot_error_matrix.to_parquet(
        PROCESSED_DIR / f"discovery_error_matrix_{SNAPSHOT_LABEL}.parquet",
        index=False,
    )

    summary = {
        "snapshot_label": SNAPSHOT_LABEL,
        "snapshot_cutoff": SNAPSHOT_CUTOFF,
        "n_models": len(snapshot_meta),
        "n_materials": len(snapshot_prediction_matrix),
        "model_keys": model_columns,
    }
    (PROCESSED_DIR / f"{SNAPSHOT_LABEL}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

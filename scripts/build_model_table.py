from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dva_project.settings import MANIFEST_DIR, MODEL_YAML_DIR, PROCESSED_DIR
from dva_project.utils import ensure_dir, iter_model_descriptor_paths, load_yaml, make_local_prediction_name


ARCHITECTURE_GROUP_BY_FAMILY = {
    "alchembert": "non_gnn",
    "alignn": "invariant_gnn",
    "alignn_ff": "invariant_gnn",
    "allegro": "equivariant_gnn",
    "alphanet": "invariant_gnn",
    "bowsr": "hybrid_ensemble",
    "cgcnn": "non_gnn",
    "chgnet": "invariant_gnn",
    "deepmd": "invariant_gnn",
    "eSEN": "transformer",
    "eqV2": "transformer",
    "eqnorm": "equivariant_gnn",
    "equflash": "equivariant_gnn",
    "esnet": "invariant_gnn",
    "gnome": "invariant_gnn",
    "grace": "equivariant_gnn",
    "hienet": "invariant_gnn",
    "m3gnet": "invariant_gnn",
    "mace": "equivariant_gnn",
    "matris": "invariant_gnn",
    "mattersim": "invariant_gnn",
    "megnet": "invariant_gnn",
    "nequip": "equivariant_gnn",
    "nequix": "equivariant_gnn",
    "orb": "hybrid_ensemble",
    "pet": "hybrid_ensemble",
    "sevennet": "equivariant_gnn",
    "tace": "equivariant_gnn",
    "voronoi_rf": "non_gnn",
    "wrenformer": "non_gnn",
}


NON_ADDITIVE_TRAINING_SETS = {
    "OpenLAM": {"OMat24", "MPtrj", "Alex", "sAlex"},
    "COSMOSDataset": {"OMat24", "MPtrj", "sAlex"},
}


def load_dataset_sizes() -> dict[str, int]:
    datasets = load_yaml(MANIFEST_DIR / "datasets.yml")
    sizes: dict[str, int] = {}
    for name, payload in datasets.items():
        count = payload.get("n_structures")
        if count is None:
            continue
        if isinstance(count, str):
            count = int(str(count).replace("_", "").replace(",", ""))
        sizes[name] = int(count)
    return sizes


def normalize_training_set(raw_training_set: list[str] | None) -> list[str]:
    if raw_training_set is None:
        return []
    return sorted(dict.fromkeys(raw_training_set))


def infer_training_group(training_set: list[str]) -> str:
    key = tuple(training_set)
    if key == ("MPtrj",):
        return "A_MPtrj_only"
    if key == ("MPtrj", "sAlex"):
        return "B_MPtrj_sAlex"
    if key == ("MPtrj", "OMat24"):
        return "C_OMat24_MPtrj"
    if key == ("MPtrj", "OMat24", "sAlex"):
        return "D_OMat24_sAlex_MPtrj"
    return "E_other"


def infer_training_combo_label(training_set: list[str]) -> str:
    if not training_set:
        return "unknown"
    return "__".join(training_set)


def estimate_effective_training_size(training_set: list[str], dataset_sizes: dict[str, int]) -> int | None:
    if not training_set:
        return None

    training_names = set(training_set)
    for compound_name, covered_sets in NON_ADDITIVE_TRAINING_SETS.items():
        if compound_name in training_names:
            return dataset_sizes.get(compound_name)

    total = 0
    if "OMat24" in training_names:
        total += dataset_sizes.get("OMat24", 0)
        training_names -= {"OMat24", "Alex", "sAlex"}
    elif "sAlex" in training_names:
        total += dataset_sizes.get("sAlex", 0)
        training_names -= {"Alex", "sAlex"}
    elif "Alex" in training_names:
        total += dataset_sizes.get("Alex", 0)
        training_names -= {"Alex"}

    for name in sorted(training_names):
        total += dataset_sizes.get(name, 0)

    return total or None


def normalize_scalar(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return value


def build_rows() -> list[dict]:
    dataset_sizes = load_dataset_sizes()
    rows: list[dict] = []

    for path in iter_model_descriptor_paths(MODEL_YAML_DIR):
        payload = load_yaml(path)
        discovery = payload.get("metrics", {}).get("discovery")
        if not isinstance(discovery, dict):
            continue

        relative_path = path.relative_to(MODEL_YAML_DIR)
        family = relative_path.parts[0]
        full_test_set = discovery.get("full_test_set", {})
        training_set = normalize_training_set(payload.get("training_set"))
        rows.append(
            {
                "model_key": payload.get("model_key"),
                "model_name": payload.get("model_name"),
                "family": family,
                "architecture_group": ARCHITECTURE_GROUP_BY_FAMILY.get(family, "unclassified"),
                "training_set": json.dumps(training_set),
                "training_group": infer_training_group(training_set),
                "training_combo": infer_training_combo_label(training_set),
                "effective_training_structures": estimate_effective_training_size(training_set, dataset_sizes),
                "model_params": payload.get("model_params"),
                "n_estimators": payload.get("n_estimators"),
                "model_type": payload.get("model_type"),
                "openness": payload.get("openness"),
                "trained_for_benchmark": payload.get("trained_for_benchmark"),
                "training_cost_raw": normalize_scalar(payload.get("training_cost")),
                "date_added": payload.get("date_added"),
                "date_published": payload.get("date_published"),
                "yaml_path": str(path.relative_to(path.parents[2])),
                "prediction_file": Path(discovery["pred_file"]).name,
                "local_prediction_file": make_local_prediction_name(
                    payload.get("model_key"),
                    Path(discovery["pred_file"]).name,
                ),
                "prediction_column": discovery.get("pred_col"),
                "f1_full_test": full_test_set.get("F1"),
                "mae_full_test": full_test_set.get("MAE"),
                "daf_full_test": full_test_set.get("DAF"),
                "r2_full_test": full_test_set.get("R2"),
                "precision_full_test": full_test_set.get("Precision"),
                "recall_full_test": full_test_set.get("Recall"),
                "accuracy_full_test": full_test_set.get("Accuracy"),
                "missing_preds_full_test": full_test_set.get("missing_preds"),
            }
        )

    return sorted(rows, key=lambda row: row["model_key"])


def main() -> None:
    rows = build_rows()
    if not rows:
        raise RuntimeError("no discovery model descriptors found; run scripts/download_inputs.py first")

    ensure_dir(PROCESSED_DIR)
    frame = pd.DataFrame(rows)
    frame.to_csv(PROCESSED_DIR / "model_metadata.csv", index=False)
    frame.to_parquet(PROCESSED_DIR / "model_metadata.parquet", index=False)


if __name__ == "__main__":
    main()

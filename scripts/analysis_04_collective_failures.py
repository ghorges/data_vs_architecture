from __future__ import annotations

import json
import math
import re
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from pymatgen.core import Composition, Element
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


SNAPSHOT_LABEL = "snapshot_45"
OUTLIER_ABS_THRESHOLD = 10.0
FAILURE_VOTE_THRESHOLD = 0.8
SUCCESS_VOTE_THRESHOLD = 1.0
COVERAGE_DISTANCE_FILL = 2.0
RADAR_FEATURES = [
    "n_elements",
    "composition_entropy",
    "std_electronegativity",
    "std_atomic_radius",
    "n_sites",
    "volume_per_atom",
]


def crystal_system_from_spacegroup(spacegroup_number: float | int | None) -> str:
    if spacegroup_number is None or math.isnan(float(spacegroup_number)):
        return "unknown"
    n = int(spacegroup_number)
    if n <= 2:
        return "triclinic"
    if n <= 15:
        return "monoclinic"
    if n <= 74:
        return "orthorhombic"
    if n <= 142:
        return "tetragonal"
    if n <= 167:
        return "trigonal"
    if n <= 194:
        return "hexagonal"
    return "cubic"


def parse_spacegroup_number(label: str) -> float:
    match = re.search(r"_(\d+)_", label)
    return float(match.group(1)) if match else np.nan


@lru_cache(maxsize=300000)
def composition_feature_dict(formula: str) -> dict:
    composition = Composition(formula)
    fractions = np.array(list(composition.fractional_composition.get_el_amt_dict().values()), dtype=float)
    elements = [Element(symbol) for symbol in composition.as_dict()]

    def _collect(attr_name: str, fallback=np.nan):
        values = []
        for el in elements:
            value = getattr(el, attr_name, fallback)
            if value is None:
                value = fallback
            values.append(float(value) if value is not None and not isinstance(value, str) else np.nan)
        return np.array(values, dtype=float)

    atomic_numbers = np.array([float(el.Z) for el in elements], dtype=float)
    electronegativities = np.array(
        [float(el.X) if el.X is not None else np.nan for el in elements],
        dtype=float,
    )
    atomic_radii = np.array(
        [
            float(el.atomic_radius) if el.atomic_radius is not None else np.nan
            for el in elements
        ],
        dtype=float,
    )
    rows = np.array([float(el.row) if el.row is not None else np.nan for el in elements], dtype=float)
    groups = np.array([float(el.group) if el.group is not None else np.nan for el in elements], dtype=float)

    entropy = -float(np.sum(fractions * np.log(fractions)))
    return {
        "elements": sorted(str(el) for el in composition.as_dict()),
        "n_elements": len(elements),
        "max_fraction": float(fractions.max()),
        "composition_entropy": entropy,
        "mean_atomic_number": float(np.nanmean(atomic_numbers)),
        "std_atomic_number": float(np.nanstd(atomic_numbers)),
        "mean_electronegativity": float(np.nanmean(electronegativities)),
        "std_electronegativity": float(np.nanstd(electronegativities)),
        "mean_atomic_radius": float(np.nanmean(atomic_radii)),
        "std_atomic_radius": float(np.nanstd(atomic_radii)),
        "mean_row": float(np.nanmean(rows)),
        "std_row": float(np.nanstd(rows)),
        "mean_group": float(np.nanmean(groups)),
        "std_group": float(np.nanstd(groups)),
    }


def periodic_table_position(symbol: str) -> tuple[int, int]:
    el = Element(symbol)
    if el.is_lanthanoid:
        lanth_index = [str(e) for e in Element if e.is_lanthanoid].index(symbol)
        return 3 + lanth_index, 8
    if el.is_actinoid:
        act_index = [str(e) for e in Element if e.is_actinoid].index(symbol)
        return 3 + act_index, 9
    return int(el.group), int(el.row)


def build_collective_outcomes(matrix: pd.DataFrame, model_keys: list[str]) -> pd.DataFrame:
    cleaned = matrix.copy()
    for model in model_keys:
        cleaned.loc[cleaned[model].abs() > OUTLIER_ABS_THRESHOLD, model] = np.nan

    reference_hull_energy = (
        cleaned["e_form_per_atom_mp2020_corrected"]
        - cleaned["e_above_hull_mp2020_corrected_ppd_mp"]
    )
    true_stable = cleaned["e_above_hull_mp2020_corrected_ppd_mp"] <= 0
    pred_stable = pd.DataFrame(index=cleaned.index)
    for model in model_keys:
        pred_stable[model] = (cleaned[model] - reference_hull_energy) <= 0
    vote_rate = pred_stable.mean(axis=1)

    result = cleaned[
        [
            "material_id",
            "formula",
            "n_sites",
            "volume",
            "wyckoff_spglib",
            "unique_prototype",
            "e_above_hull_mp2020_corrected_ppd_mp",
        ]
    ].copy()
    result["true_stable"] = true_stable
    result["stable_vote_rate"] = vote_rate
    result["collective_false_negative"] = true_stable & (vote_rate <= (1 - FAILURE_VOTE_THRESHOLD))
    result["collective_false_positive"] = (~true_stable) & (vote_rate >= FAILURE_VOTE_THRESHOLD)
    result["collective_failure"] = result["collective_false_negative"] | result["collective_false_positive"]
    result["collective_success"] = (
        (true_stable & (vote_rate >= SUCCESS_VOTE_THRESHOLD))
        | ((~true_stable) & (vote_rate <= (1 - SUCCESS_VOTE_THRESHOLD)))
    )
    result["failure_type"] = np.select(
        [
            result["collective_false_negative"],
            result["collective_false_positive"],
        ],
        [
            "false_negative",
            "false_positive",
        ],
        default="other",
    )
    return result


def merge_coverage_proxy(outcomes: pd.DataFrame) -> pd.DataFrame:
    coverage = pd.read_parquet(
        PROCESSED_DIR / "wbm_mptrj_material_coverage_proxy.parquet",
        columns=[
            "material_id",
            "exact_formula_in_mptrj",
            "same_element_set_in_mptrj",
            "same_element_set_formula_count",
            "min_l1_distance_same_element_set",
        ],
    ).drop_duplicates(subset=["material_id"])

    merged = outcomes.merge(coverage, on="material_id", how="left", validate="one_to_one")
    merged["exact_formula_in_mptrj"] = merged["exact_formula_in_mptrj"].fillna(False).astype(bool)
    merged["same_element_set_in_mptrj"] = merged["same_element_set_in_mptrj"].fillna(False).astype(bool)
    merged["same_element_set_formula_count"] = merged["same_element_set_formula_count"].fillna(0).astype(int)
    merged["min_l1_distance_same_element_set"] = merged["min_l1_distance_same_element_set"].astype(float)
    merged["min_l1_distance_same_element_set_filled"] = merged["min_l1_distance_same_element_set"].fillna(
        COVERAGE_DISTANCE_FILL
    )
    return merged


def enrich_features(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for row in outcomes.itertuples(index=False):
        comp_features = composition_feature_dict(row.formula)
        volume_per_atom = row.volume / row.n_sites if row.n_sites else np.nan
        spacegroup_number = parse_spacegroup_number(row.wyckoff_spglib)
        rows.append(
            {
                **row._asdict(),
                **{k: v for k, v in comp_features.items() if k != "elements"},
                "elements": json.dumps(comp_features["elements"]),
                "volume_per_atom": volume_per_atom,
                "spacegroup_number": spacegroup_number,
                "crystal_system": crystal_system_from_spacegroup(spacegroup_number),
                "exact_formula_in_mptrj": bool(row.exact_formula_in_mptrj),
                "same_element_set_in_mptrj": bool(row.same_element_set_in_mptrj),
                "same_element_set_formula_count": int(row.same_element_set_formula_count),
                "min_l1_distance_same_element_set": float(row.min_l1_distance_same_element_set)
                if not pd.isna(row.min_l1_distance_same_element_set)
                else np.nan,
                "min_l1_distance_same_element_set_filled": float(row.min_l1_distance_same_element_set_filled),
            }
        )
    return pd.DataFrame(rows)


def compare_failure_success(feature_frame: pd.DataFrame) -> pd.DataFrame:
    subset = feature_frame.loc[feature_frame["collective_failure"] | feature_frame["collective_success"]].copy()
    subset["label"] = np.where(subset["collective_failure"], "failure", "success")
    numeric_cols = [
        "n_elements",
        "max_fraction",
        "composition_entropy",
        "mean_atomic_number",
        "std_atomic_number",
        "mean_electronegativity",
        "std_electronegativity",
        "mean_atomic_radius",
        "std_atomic_radius",
        "n_sites",
        "volume_per_atom",
        "spacegroup_number",
        "exact_formula_in_mptrj",
        "same_element_set_in_mptrj",
        "same_element_set_formula_count",
        "min_l1_distance_same_element_set_filled",
    ]

    rows: list[dict] = []
    for feature in numeric_cols:
        grouped = subset.groupby("label")[feature].agg(["mean", "median", "std"]).reset_index()
        grouped["feature"] = feature
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def train_failure_tree(
    feature_frame: pd.DataFrame,
) -> tuple[DecisionTreeClassifier, pd.DataFrame, float, list[str], pd.DataFrame]:
    subset = feature_frame.loc[feature_frame["collective_failure"] | feature_frame["collective_success"]].copy()
    subset["label"] = subset["collective_failure"].astype(int)
    subset["crystal_system"] = subset["crystal_system"].fillna("unknown")

    base_feature_columns = [
        "n_elements",
        "max_fraction",
        "composition_entropy",
        "mean_atomic_number",
        "std_atomic_number",
        "mean_electronegativity",
        "std_electronegativity",
        "mean_atomic_radius",
        "std_atomic_radius",
        "n_sites",
        "volume_per_atom",
        "spacegroup_number",
        "unique_prototype",
        "crystal_system",
    ]
    coverage_feature_columns = [
        "exact_formula_in_mptrj",
        "same_element_set_in_mptrj",
        "same_element_set_formula_count",
        "min_l1_distance_same_element_set_filled",
    ]
    feature_columns = base_feature_columns + coverage_feature_columns
    model_frame = subset[feature_columns + ["label"]].copy()
    model_frame["unique_prototype"] = model_frame["unique_prototype"].astype(int)
    model_frame["exact_formula_in_mptrj"] = model_frame["exact_formula_in_mptrj"].astype(int)
    model_frame["same_element_set_in_mptrj"] = model_frame["same_element_set_in_mptrj"].astype(int)
    numeric_fill_columns = [column for column in feature_columns if column != "crystal_system"]
    for column in numeric_fill_columns:
        model_frame[column] = model_frame[column].fillna(model_frame[column].median())
    model_frame = pd.get_dummies(model_frame, columns=["crystal_system"], drop_first=False)

    X = model_frame.drop(columns=["label"])
    y = model_frame["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    clf = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=100,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    balanced_acc = balanced_accuracy_score(y_test, pred)

    importances = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    coverage_only_importances = importances.loc[
        importances["feature"].isin(coverage_feature_columns)
    ].reset_index(drop=True)
    return clf, importances, balanced_acc, X.columns.tolist(), coverage_only_importances


def summarize_coverage_gap(feature_frame: pd.DataFrame) -> pd.DataFrame:
    subset = feature_frame.loc[feature_frame["collective_failure"] | feature_frame["collective_success"]].copy()
    subset["label"] = np.where(subset["collective_failure"], "failure", "success")

    rows: list[dict] = []
    for label, group in subset.groupby("label"):
        same_set = group.loc[group["same_element_set_in_mptrj"]]
        rows.append(
            {
                "label": label,
                "n_materials": int(len(group)),
                "exact_formula_hit_rate": float(group["exact_formula_in_mptrj"].mean()),
                "same_element_set_hit_rate": float(group["same_element_set_in_mptrj"].mean()),
                "median_same_element_set_formula_count": float(group["same_element_set_formula_count"].median()),
                "mean_same_element_set_formula_count": float(group["same_element_set_formula_count"].mean()),
                "median_min_l1_distance_same_element_set": float(
                    same_set["min_l1_distance_same_element_set"].median()
                )
                if len(same_set)
                else np.nan,
                "mean_min_l1_distance_same_element_set": float(
                    same_set["min_l1_distance_same_element_set"].mean()
                )
                if len(same_set)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def build_element_failure_rates(feature_frame: pd.DataFrame) -> pd.DataFrame:
    total_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}

    for row in feature_frame.itertuples(index=False):
        elements = json.loads(row.elements)
        for el in elements:
            total_counts[el] = total_counts.get(el, 0) + 1
            if row.collective_failure:
                failure_counts[el] = failure_counts.get(el, 0) + 1

    rows: list[dict] = []
    for symbol, total in total_counts.items():
        failures = failure_counts.get(symbol, 0)
        group, period = periodic_table_position(symbol)
        rows.append(
            {
                "element": symbol,
                "total_materials": total,
                "collective_failures": failures,
                "failure_rate": failures / total,
                "group": group,
                "period": period,
            }
        )
    return pd.DataFrame(rows).sort_values(["period", "group"])


def make_periodic_table_plot(element_rates: pd.DataFrame, output_path) -> None:
    fig, ax = plt.subplots(figsize=(18, 5.5))
    cmap = plt.cm.OrRd
    max_rate = element_rates["failure_rate"].max()

    for row in element_rates.itertuples(index=False):
        x = row.group
        y = row.period
        color = cmap(row.failure_rate / max_rate if max_rate else 0.0)
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=1.5))
        ax.text(x + 0.5, y + 0.42, row.element, ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x + 0.5, y + 0.72, f"{row.failure_rate:.2%}", ha="center", va="center", fontsize=6)

    ax.set_xlim(1, 19)
    ax.set_ylim(10, 0)
    ax.set_xticks(range(1, 19))
    ax.set_yticks(range(1, 10))
    ax.set_xlabel("Group")
    ax.set_ylabel("Period / series")
    ax.set_title("Collective failure rate by element")
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f4ed")
    for spine in ax.spines.values():
        spine.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_rate))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Failure rate among materials containing the element")

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_radar_plot(feature_frame: pd.DataFrame, output_path) -> None:
    subset = feature_frame.loc[feature_frame["collective_failure"] | feature_frame["collective_success"]].copy()
    subset["label"] = np.where(subset["collective_failure"], "failure", "success")

    values = []
    for feature in RADAR_FEATURES:
        grouped = subset.groupby("label")[feature].mean()
        low = grouped.min()
        high = grouped.max()
        span = high - low if high != low else 1.0
        values.append(
            {
                "feature": feature,
                "failure": (grouped["failure"] - low) / span,
                "success": (grouped["success"] - low) / span,
            }
        )
    radar = pd.DataFrame(values)

    angles = np.linspace(0, 2 * np.pi, len(radar), endpoint=False).tolist()
    angles += angles[:1]
    failure_values = radar["failure"].tolist() + [radar["failure"].iloc[0]]
    success_values = radar["success"].tolist() + [radar["success"].iloc[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.plot(angles, failure_values, color="#d95f02", linewidth=2, label="Collective failure")
    ax.fill(angles, failure_values, color="#d95f02", alpha=0.18)
    ax.plot(angles, success_values, color="#1b9e77", linewidth=2, label="Collective success")
    ax.fill(angles, success_values, color="#1b9e77", alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar["feature"].tolist())
    ax.set_yticklabels([])
    ax.set_title("Failure vs success feature profile")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), frameon=False)

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_tree_plot(clf: DecisionTreeClassifier, feature_names: list[str], output_path) -> None:
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["success", "failure"],
        filled=True,
        rounded=True,
        impurity=False,
        ax=ax,
        fontsize=8,
    )
    ax.set_title("Decision tree for collective failures")
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_table_dir = RESULTS_DIR / "tables" / "analysis_04"
    output_figure_dir = RESULTS_DIR / "figures" / "analysis_04"
    ensure_dir(output_table_dir)
    ensure_dir(output_figure_dir)

    metadata = pd.read_csv(PROCESSED_DIR / f"model_metadata_{SNAPSHOT_LABEL}.csv")
    matrix = pd.read_parquet(PROCESSED_DIR / f"discovery_prediction_matrix_{SNAPSHOT_LABEL}.parquet")

    outcomes = build_collective_outcomes(matrix, metadata["model_key"].tolist())
    outcomes = merge_coverage_proxy(outcomes)
    features = enrich_features(outcomes)
    comparison = compare_failure_success(features)
    coverage_gap = summarize_coverage_gap(features)
    clf, importances, balanced_acc, tree_feature_names, coverage_importances = train_failure_tree(features)
    element_rates = build_element_failure_rates(features)

    failure_row = coverage_gap.loc[coverage_gap["label"] == "failure"].iloc[0]
    success_row = coverage_gap.loc[coverage_gap["label"] == "success"].iloc[0]

    summary = {
        "collective_false_negative": int(features["collective_false_negative"].sum()),
        "collective_false_positive": int(features["collective_false_positive"].sum()),
        "collective_failure_total": int(features["collective_failure"].sum()),
        "collective_success_total": int(features["collective_success"].sum()),
        "decision_tree_balanced_accuracy": balanced_acc,
        "failure_exact_formula_hit_rate": float(failure_row["exact_formula_hit_rate"]),
        "success_exact_formula_hit_rate": float(success_row["exact_formula_hit_rate"]),
        "failure_same_element_set_hit_rate": float(failure_row["same_element_set_hit_rate"]),
        "success_same_element_set_hit_rate": float(success_row["same_element_set_hit_rate"]),
        "failure_median_min_l1_distance_same_element_set": float(
            failure_row["median_min_l1_distance_same_element_set"]
        ),
        "success_median_min_l1_distance_same_element_set": float(
            success_row["median_min_l1_distance_same_element_set"]
        ),
    }

    outcomes.to_csv(output_table_dir / "collective_outcomes.csv", index=False)
    features.to_csv(output_table_dir / "collective_failure_features.csv", index=False)
    comparison.to_csv(output_table_dir / "failure_success_feature_summary.csv", index=False)
    coverage_gap.to_csv(output_table_dir / "failure_success_coverage_summary.csv", index=False)
    importances.to_csv(output_table_dir / "decision_tree_feature_importance.csv", index=False)
    coverage_importances.to_csv(output_table_dir / "decision_tree_coverage_feature_importance.csv", index=False)
    element_rates.to_csv(output_table_dir / "element_failure_rates.csv", index=False)
    (output_table_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_periodic_table_plot(element_rates, output_figure_dir / "element_failure_periodic_table.png")
    make_radar_plot(features, output_figure_dir / "failure_success_radar.png")
    make_tree_plot(clf, tree_feature_names, output_figure_dir / "decision_tree.png")


if __name__ == "__main__":
    main()

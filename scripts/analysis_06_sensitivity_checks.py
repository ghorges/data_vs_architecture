from __future__ import annotations

import json
import math
import re
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from pymatgen.core import Composition, Element
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.anova import anova_lm

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


SNAPSHOT_CONFIGS = [
    {
        "dataset_key": "snapshot_45",
        "dataset_label": "Frozen 45-model snapshot",
        "metadata_path": PROCESSED_DIR / "model_metadata_snapshot_45.csv",
        "prediction_matrix_path": PROCESSED_DIR / "discovery_prediction_matrix_snapshot_45.parquet",
        "error_matrix_path": PROCESSED_DIR / "discovery_error_matrix_snapshot_45.parquet",
    },
    {
        "dataset_key": "full_53",
        "dataset_label": "Current 53-model source state",
        "metadata_path": PROCESSED_DIR / "model_metadata.csv",
        "prediction_matrix_path": PROCESSED_DIR / "discovery_prediction_matrix.parquet",
        "error_matrix_path": PROCESSED_DIR / "discovery_error_matrix.parquet",
    },
]
METRICS = {
    "f1_full_test": "F1",
    "mae_full_test": "MAE",
    "daf_full_test": "DAF",
    "r2_full_test": "R2",
}
FAILURE_THRESHOLDS = [0.7, 0.8, 0.9]
OUTLIER_ABS_THRESHOLD = 10.0
SUCCESS_VOTE_THRESHOLD = 1.0


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def exact_anova_sensitivity(metadata: pd.DataFrame, dataset_key: str, dataset_label: str) -> pd.DataFrame:
    frame = metadata.copy()
    frame["log_model_params"] = np.log10(frame["model_params"].astype(float))

    rows: list[dict] = []
    for metric, metric_label in METRICS.items():
        metric_frame = frame.dropna(
            subset=[metric, "training_combo", "architecture_group", "log_model_params"]
        ).copy()
        model = smf.ols(
            f"{metric} ~ C(training_combo) + C(architecture_group) + log_model_params",
            data=metric_frame,
        ).fit()
        anova = anova_lm(model, typ=3).reset_index().rename(columns={"index": "term"})
        residual_ss = float(anova.loc[anova["term"] == "Residual", "sum_sq"].iloc[0])
        factor_map = {
            "C(training_combo)": "Training data",
            "C(architecture_group)": "Architecture",
            "log_model_params": "Parameters",
        }
        for row in anova.itertuples(index=False):
            if row.term not in factor_map:
                continue
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_label": dataset_label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "factor": factor_map[row.term],
                    "partial_eta_sq": row.sum_sq / (row.sum_sq + residual_ss),
                    "n_models": int(model.nobs),
                    "model_r2": model.rsquared,
                    "model_adj_r2": model.rsquared_adj,
                }
            )
    return pd.DataFrame(rows)


def error_cluster_sensitivity(
    metadata: pd.DataFrame,
    error_matrix_path,
    dataset_key: str,
    dataset_label: str,
) -> pd.DataFrame:
    errors = pd.read_parquet(error_matrix_path)
    model_order = metadata["model_key"].tolist()
    error_matrix = errors[model_order].copy()
    corr = error_matrix.corr()
    filled = error_matrix.apply(lambda col: col.fillna(col.mean()), axis=0)
    model_vectors = zscore_rows(filled.to_numpy(dtype=float).T)
    linkage_matrix = linkage(model_vectors, method="ward")

    exact_clusters = fcluster(
        linkage_matrix,
        t=metadata["training_combo"].nunique(),
        criterion="maxclust",
    )
    coarse_clusters = fcluster(
        linkage_matrix,
        t=metadata["training_group"].nunique(),
        criterion="maxclust",
    )
    arch_clusters = fcluster(
        linkage_matrix,
        t=metadata["architecture_group"].nunique(),
        criterion="maxclust",
    )

    lookup = metadata.set_index("model_key")
    pair_rows: list[dict] = []
    for index, left in enumerate(model_order):
        for right in model_order[index + 1 :]:
            pair_rows.append(
                {
                    "same_training_combo": lookup.loc[left, "training_combo"]
                    == lookup.loc[right, "training_combo"],
                    "same_architecture_group": lookup.loc[left, "architecture_group"]
                    == lookup.loc[right, "architecture_group"],
                    "correlation": float(corr.loc[left, right]),
                }
            )
    pairwise = pd.DataFrame(pair_rows)

    return pd.DataFrame(
        [
            {
                "dataset_key": dataset_key,
                "dataset_label": dataset_label,
                "n_models": int(len(metadata)),
                "n_materials": int(len(errors)),
                "ari_training_exact": adjusted_rand_score(
                    metadata["training_combo"], exact_clusters
                ),
                "ari_training_coarse": adjusted_rand_score(
                    metadata["training_group"], coarse_clusters
                ),
                "ari_architecture": adjusted_rand_score(
                    metadata["architecture_group"], arch_clusters
                ),
                "mean_corr_same_training_combo": pairwise.loc[
                    pairwise["same_training_combo"], "correlation"
                ].mean(),
                "mean_corr_diff_training_combo": pairwise.loc[
                    ~pairwise["same_training_combo"], "correlation"
                ].mean(),
                "mean_corr_same_architecture_group": pairwise.loc[
                    pairwise["same_architecture_group"], "correlation"
                ].mean(),
                "mean_corr_diff_architecture_group": pairwise.loc[
                    ~pairwise["same_architecture_group"], "correlation"
                ].mean(),
            }
        ]
    )


def parse_spacegroup_number(label: str) -> float:
    match = re.search(r"_(\d+)_", label)
    return float(match.group(1)) if match else np.nan


def crystal_system_from_spacegroup(spacegroup_number: float | int | None) -> str:
    if spacegroup_number is None or math.isnan(float(spacegroup_number)):
        return "unknown"
    number = int(spacegroup_number)
    if number <= 2:
        return "triclinic"
    if number <= 15:
        return "monoclinic"
    if number <= 74:
        return "orthorhombic"
    if number <= 142:
        return "tetragonal"
    if number <= 167:
        return "trigonal"
    if number <= 194:
        return "hexagonal"
    return "cubic"


@lru_cache(maxsize=300000)
def composition_feature_dict(formula: str) -> dict:
    composition = Composition(formula)
    fractions = np.array(
        list(composition.fractional_composition.get_el_amt_dict().values()),
        dtype=float,
    )
    elements = [Element(symbol) for symbol in composition.as_dict()]
    atomic_numbers = np.array([float(el.Z) for el in elements], dtype=float)
    electronegativities = np.array(
        [float(el.X) if el.X is not None else np.nan for el in elements],
        dtype=float,
    )
    atomic_radii = np.array(
        [float(el.atomic_radius) if el.atomic_radius is not None else np.nan for el in elements],
        dtype=float,
    )

    return {
        "n_elements": len(elements),
        "max_fraction": float(fractions.max()),
        "composition_entropy": -float(np.sum(fractions * np.log(fractions))),
        "mean_atomic_number": float(np.nanmean(atomic_numbers)),
        "std_atomic_number": float(np.nanstd(atomic_numbers)),
        "mean_electronegativity": float(np.nanmean(electronegativities)),
        "std_electronegativity": float(np.nanstd(electronegativities)),
        "mean_atomic_radius": float(np.nanmean(atomic_radii)),
        "std_atomic_radius": float(np.nanstd(atomic_radii)),
    }


def build_feature_base(metadata: pd.DataFrame, prediction_matrix_path) -> pd.DataFrame:
    matrix = pd.read_parquet(prediction_matrix_path).copy()
    model_keys = metadata["model_key"].tolist()
    for model_key in model_keys:
        matrix.loc[matrix[model_key].abs() > OUTLIER_ABS_THRESHOLD, model_key] = np.nan

    reference_hull_energy = (
        matrix["e_form_per_atom_mp2020_corrected"] - matrix["e_above_hull_mp2020_corrected_ppd_mp"]
    )
    true_stable = matrix["e_above_hull_mp2020_corrected_ppd_mp"] <= 0
    pred_stable = pd.DataFrame(index=matrix.index)
    for model_key in model_keys:
        pred_stable[model_key] = (matrix[model_key] - reference_hull_energy) <= 0
    stable_vote_rate = pred_stable.mean(axis=1).to_numpy(dtype=float)
    true_stable_values = true_stable.to_numpy(dtype=bool)

    feature_rows: list[dict] = []
    base_columns = [
        "material_id",
        "formula",
        "n_sites",
        "volume",
        "wyckoff_spglib",
        "unique_prototype",
    ]
    for index, row in enumerate(matrix[base_columns].itertuples(index=False)):
        composition_features = composition_feature_dict(row.formula)
        spacegroup_number = parse_spacegroup_number(row.wyckoff_spglib)
        feature_rows.append(
            {
                **row._asdict(),
                **composition_features,
                "true_stable": bool(true_stable_values[index]),
                "stable_vote_rate": float(stable_vote_rate[index]),
                "volume_per_atom": row.volume / row.n_sites if row.n_sites else np.nan,
                "spacegroup_number": spacegroup_number,
                "crystal_system": crystal_system_from_spacegroup(spacegroup_number),
            }
        )

    return pd.DataFrame(feature_rows)


def train_threshold_tree(feature_frame: pd.DataFrame) -> tuple[float, list[str]]:
    subset = feature_frame.loc[
        feature_frame["collective_failure"] | feature_frame["collective_success"]
    ].copy()
    subset["label"] = subset["collective_failure"].astype(int)

    feature_columns = [
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
    model_frame = subset[feature_columns + ["label"]].copy()
    model_frame["crystal_system"] = model_frame["crystal_system"].fillna("unknown")
    model_frame["unique_prototype"] = model_frame["unique_prototype"].astype(int)
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
    predictions = clf.predict(X_test)
    importances = (
        pd.DataFrame({"feature": X.columns, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return float(balanced_accuracy_score(y_test, predictions)), importances["feature"].head(5).tolist()


def failure_threshold_sensitivity(feature_base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    top_feature_rows: list[dict] = []

    for threshold in FAILURE_THRESHOLDS:
        frame = feature_base.copy()
        frame["collective_false_negative"] = frame["true_stable"] & (
            frame["stable_vote_rate"] <= (1 - threshold)
        )
        frame["collective_false_positive"] = (~frame["true_stable"]) & (
            frame["stable_vote_rate"] >= threshold
        )
        frame["collective_failure"] = (
            frame["collective_false_negative"] | frame["collective_false_positive"]
        )
        frame["collective_success"] = (
            (frame["true_stable"] & (frame["stable_vote_rate"] >= SUCCESS_VOTE_THRESHOLD))
            | ((~frame["true_stable"]) & (frame["stable_vote_rate"] <= (1 - SUCCESS_VOTE_THRESHOLD)))
        )
        balanced_acc, top_features = train_threshold_tree(frame)
        summary_rows.append(
            {
                "failure_vote_threshold": threshold,
                "collective_false_negative": int(frame["collective_false_negative"].sum()),
                "collective_false_positive": int(frame["collective_false_positive"].sum()),
                "collective_failure_total": int(frame["collective_failure"].sum()),
                "collective_success_total": int(frame["collective_success"].sum()),
                "decision_tree_balanced_accuracy": balanced_acc,
            }
        )
        for rank, feature_name in enumerate(top_features, start=1):
            top_feature_rows.append(
                {
                    "failure_vote_threshold": threshold,
                    "rank": rank,
                    "feature": feature_name,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(top_feature_rows)


def make_sensitivity_plot(
    anova_comparison: pd.DataFrame,
    cluster_comparison: pd.DataFrame,
    threshold_comparison: pd.DataFrame,
    output_path,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    panel_a = anova_comparison[anova_comparison["factor"].isin(["Training data", "Architecture"])].copy()
    sns.lineplot(
        data=panel_a,
        x="metric_label",
        y="partial_eta_sq",
        hue="factor",
        style="dataset_label",
        markers=True,
        dashes=False,
        palette={"Training data": "#0b6e4f", "Architecture": "#f4a259"},
        ax=axes[0],
    )
    axes[0].set_title("A. Exact ANOVA is stable from 45 to 53 models")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Partial eta squared")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(frameon=False, fontsize=8, title="")

    ari_frame = cluster_comparison.melt(
        id_vars=["dataset_label"],
        value_vars=["ari_training_exact", "ari_architecture"],
        var_name="label_type",
        value_name="adjusted_rand_index",
    )
    ari_frame["label_type"] = ari_frame["label_type"].map(
        {
            "ari_training_exact": "Training data",
            "ari_architecture": "Architecture",
        }
    )
    sns.barplot(
        data=ari_frame,
        x="dataset_label",
        y="adjusted_rand_index",
        hue="label_type",
        palette={"Training data": "#0b6e4f", "Architecture": "#f4a259"},
        ax=axes[1],
    )
    axes[1].set_title("B. Error clustering still tracks data")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Adjusted Rand index")
    axes[1].tick_params(axis="x", rotation=16)
    axes[1].legend(frameon=False, fontsize=8, title="")

    line_counts = sns.lineplot(
        data=threshold_comparison,
        x="failure_vote_threshold",
        y="collective_failure_total",
        marker="o",
        color="#d95f02",
        ax=axes[2],
        label="Collective failures",
    )
    acc_ax = axes[2].twinx()
    sns.lineplot(
        data=threshold_comparison,
        x="failure_vote_threshold",
        y="decision_tree_balanced_accuracy",
        marker="s",
        color="#1b9e77",
        ax=acc_ax,
        label="Balanced accuracy",
    )
    axes[2].set_title("C. Blind spots shrink but stay learnable")
    axes[2].set_xlabel("Failure vote threshold")
    axes[2].set_ylabel("Collective failure count")
    acc_ax.set_ylabel("Balanced accuracy")
    axes[2].set_xticks(FAILURE_THRESHOLDS)
    handles_1, labels_1 = line_counts.get_legend_handles_labels()
    handles_2, labels_2 = acc_ax.get_legend_handles_labels()
    axes[2].legend(handles_1 + handles_2, labels_1 + labels_2, frameon=False, fontsize=8)

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_table_dir = RESULTS_DIR / "tables" / "sensitivity"
    output_figure_dir = RESULTS_DIR / "figures" / "sensitivity"
    ensure_dir(output_table_dir)
    ensure_dir(output_figure_dir)

    anova_frames: list[pd.DataFrame] = []
    cluster_frames: list[pd.DataFrame] = []
    metadata_lookup: dict[str, pd.DataFrame] = {}

    for config in SNAPSHOT_CONFIGS:
        metadata = pd.read_csv(config["metadata_path"])
        metadata_lookup[config["dataset_key"]] = metadata
        anova_frames.append(
            exact_anova_sensitivity(
                metadata,
                dataset_key=config["dataset_key"],
                dataset_label=config["dataset_label"],
            )
        )
        cluster_frames.append(
            error_cluster_sensitivity(
                metadata,
                config["error_matrix_path"],
                dataset_key=config["dataset_key"],
                dataset_label=config["dataset_label"],
            )
        )

    anova_comparison = pd.concat(anova_frames, ignore_index=True)
    cluster_comparison = pd.concat(cluster_frames, ignore_index=True)

    feature_base = build_feature_base(
        metadata_lookup["snapshot_45"],
        prediction_matrix_path=SNAPSHOT_CONFIGS[0]["prediction_matrix_path"],
    )
    threshold_comparison, threshold_top_features = failure_threshold_sensitivity(feature_base)

    summary = {
        "anova_exact": {},
        "error_clustering": cluster_comparison.set_index("dataset_key")
        .drop(columns=["dataset_label"])
        .to_dict(orient="index"),
        "collective_failure_thresholds": threshold_comparison.set_index(
            "failure_vote_threshold"
        ).to_dict(orient="index"),
    }
    for dataset_key, dataset_frame in anova_comparison.groupby("dataset_key"):
        summary["anova_exact"][dataset_key] = {}
        for metric, metric_frame in dataset_frame.groupby("metric_label"):
            summary["anova_exact"][dataset_key][metric] = {
                row.factor: row.partial_eta_sq for row in metric_frame.itertuples(index=False)
            }

    anova_comparison.to_csv(output_table_dir / "analysis_01_snapshot_comparison.csv", index=False)
    cluster_comparison.to_csv(output_table_dir / "analysis_02_snapshot_comparison.csv", index=False)
    threshold_comparison.to_csv(
        output_table_dir / "analysis_04_threshold_sensitivity.csv",
        index=False,
    )
    threshold_top_features.to_csv(
        output_table_dir / "analysis_04_threshold_top_features.csv",
        index=False,
    )
    (output_table_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_sensitivity_plot(
        anova_comparison,
        cluster_comparison,
        threshold_comparison,
        output_figure_dir / "sensitivity_checks.png",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from dva_project.figure_style import (
    FACTOR_COLORS,
    add_panel_label,
    configure_publication_style,
    create_figure_mm,
    save_figure_to_many,
    style_axes,
)
from dva_project.settings import PROJECT_ROOT, RESULTS_DIR
from dva_project.utils import ensure_dir


SUPP_PACKAGE_DIR = PROJECT_ROOT / "paper" / "submission_package" / "figures" / "supplementary"
SUPP_RESULTS_DIR = RESULTS_DIR / "figures"
METRIC_ORDER = ["F1", "MAE", "DAF", "R2"]

FEATURE_NAME_MAP = {
    "unique_prototype": "Unique prototype",
    "spacegroup_number": "Space group number",
    "n_sites": "Number of sites",
    "mean_atomic_number": "Mean atomic number",
    "std_atomic_radius": "Atomic radius std",
    "std_atomic_number": "Atomic number std",
    "std_electronegativity": "Electronegativity std",
    "min_l1_distance_same_element_set_filled": "Nearest-set distance",
    "volume_per_atom": "Volume per atom",
}
S04_FOCUS_FEATURES = [
    "unique_prototype",
    "spacegroup_number",
    "n_sites",
    "mean_atomic_number",
    "std_electronegativity",
]


def output_dirs(figure_name: str) -> tuple[list[Path], Path]:
    results_dir = SUPP_RESULTS_DIR / figure_name
    package_dir = SUPP_PACKAGE_DIR / figure_name
    for directory in [results_dir, package_dir]:
        ensure_dir(directory)
    return [results_dir, package_dir], package_dir


def save_panel(fig, destinations: list[Path], stem_name: str) -> None:
    save_figure_to_many(fig, [destination / stem_name for destination in destinations])


def compose_preview(
    package_dir: Path,
    destinations: list[Path],
    stem_name: str,
    placements: list[tuple[str, list[float]]],
    size_mm: tuple[float, float],
) -> None:
    fig = create_figure_mm(*size_mm)
    fig.patch.set_facecolor("white")
    for file_name, position in placements:
        axis = fig.add_axes(position)
        axis.imshow(plt.imread(package_dir / file_name))
        axis.axis("off")
    save_figure_to_many(fig, [destination / stem_name for destination in destinations])


def write_notes(destinations: list[Path], title: str, lines: list[str]) -> None:
    contents = "\n".join([f"# {title}", "", *lines, ""])
    for directory in destinations:
        (directory / "assembly_notes.md").write_text(contents, encoding="utf-8")


def format_feature_name(name: str) -> str:
    return FEATURE_NAME_MAP.get(name, name.replace("_", " ").title())


def make_s04_decision_tree() -> None:
    destinations, package_dir = output_dirs("figure_s04_decision_tree")
    importances = pd.read_csv(RESULTS_DIR / "tables" / "analysis_04" / "decision_tree_feature_importance.csv").sort_values("importance", ascending=False).head(8)
    threshold_top = pd.read_csv(RESULTS_DIR / "tables" / "sensitivity" / "analysis_04_threshold_top_features.csv")
    summary = json.loads((RESULTS_DIR / "tables" / "analysis_04" / "summary.json").read_text(encoding="utf-8"))

    fig = create_figure_mm(86.0, 70.0)
    ax = fig.add_subplot(111)
    style_axes(ax, grid=False)
    y = np.arange(len(importances))
    ax.barh(y, importances["importance"], color="#4C78A8", edgecolor="#1F3552", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([format_feature_name(value) for value in importances["feature"]], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_title("")
    ax.text(
        0.58,
        0.10,
        f"Balanced accuracy = {summary['decision_tree_balanced_accuracy']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.2,
        color="#333333",
    )
    add_panel_label(ax, "a", x=-0.11, y=1.03)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.88, bottom=0.14)
    save_panel(fig, destinations, "panel_a_tree_importance")

    pivot = threshold_top.pivot(index="feature", columns="failure_vote_threshold", values="rank")
    pivot = pivot.reindex(S04_FOCUS_FEATURES)

    fig = create_figure_mm(86.0, 70.0)
    ax = fig.add_subplot(111)
    style_axes(ax, grid=False)
    thresholds = [0.7, 0.8, 0.9]
    x = np.arange(len(thresholds))
    feature_styles = {
        "unique_prototype": {"color": "#14A37F", "lw": 2.4, "ms": 5.0, "zorder": 4},
        "spacegroup_number": {"color": "#B8B8B8", "lw": 1.15, "ms": 4.3, "zorder": 3},
        "n_sites": {"color": "#B8B8B8", "lw": 1.15, "ms": 4.3, "zorder": 3},
        "mean_atomic_number": {"color": "#B8B8B8", "lw": 1.15, "ms": 4.3, "zorder": 3},
        "std_electronegativity": {"color": "#D0D0D0", "lw": 1.0, "ms": 4.0, "zorder": 2},
    }
    marker_map = {
        "unique_prototype": "o",
        "spacegroup_number": "s",
        "n_sites": "D",
        "mean_atomic_number": "^",
        "std_electronegativity": "v",
    }
    for feature, row in pivot.iterrows():
        if row.isna().all():
            continue
        yvals = np.array([row.get(value, np.nan) for value in thresholds], dtype=float)
        if np.isfinite(yvals).sum() < 2:
            continue
        style = feature_styles[feature]
        ax.plot(
            x,
            yvals,
            marker=marker_map[feature],
            ms=style["ms"],
            lw=style["lw"],
            color=style["color"],
            zorder=style["zorder"],
        )
        valid_idx = np.where(np.isfinite(yvals))[0]
        label_x = x[valid_idx[-1]] + 0.12
        label_y = yvals[valid_idx[-1]]
        if feature == "n_sites":
            label_y = yvals[valid_idx[-1]] - 0.12
        elif feature == "mean_atomic_number":
            label_y = yvals[valid_idx[-1]] - 0.02
        elif feature == "spacegroup_number":
            label_y = yvals[valid_idx[-1]] - 0.00
        ax.text(
            label_x,
            label_y,
            format_feature_name(feature),
            fontsize=6.0,
            color="#14A37F" if feature == "unique_prototype" else "#6F6F6F",
            fontweight="bold" if feature == "unique_prototype" else "normal",
            ha="left",
            va="center",
            clip_on=False,
        )
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([f"{value:.1f}" for value in thresholds])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(5.2, 0.8)
    ax.set_xlim(-0.10, 2.72)
    ax.set_xlabel("Failure vote threshold")
    ax.set_ylabel("Feature rank")
    ax.set_title("")
    add_panel_label(ax, "b", x=-0.11, y=1.03)
    fig.subplots_adjust(left=0.16, right=0.95, top=0.88, bottom=0.16)
    save_panel(fig, destinations, "panel_b_threshold_rank_stability")

    compose_preview(
        package_dir,
        destinations,
        "figure_s04_preview",
        [
            ("panel_a_tree_importance.png", [0.00, 0.06, 0.48, 0.88]),
            ("panel_b_threshold_rank_stability.png", [0.52, 0.06, 0.48, 0.88]),
        ],
        (180.0, 76.0),
    )
    write_notes(
        destinations,
        "Figure S4 Assembly Notes",
        [
            "Two-panel supplementary figure.",
            "Panel a summarizes the fitted decision tree using exported feature importances.",
            "Panel b shows how top-ranked signals persist when the collective-failure threshold is varied; features appearing at only one threshold are omitted from the stability trajectory panel.",
        ],
    )


def make_s05_sensitivity_overview() -> None:
    destinations, package_dir = output_dirs("figure_s05_sensitivity_overview")
    anova = pd.read_csv(RESULTS_DIR / "tables" / "sensitivity" / "analysis_01_snapshot_comparison.csv")
    cluster = pd.read_csv(RESULTS_DIR / "tables" / "sensitivity" / "analysis_02_snapshot_comparison.csv")
    thresholds = pd.read_csv(RESULTS_DIR / "tables" / "sensitivity" / "analysis_04_threshold_sensitivity.csv")

    panel_size_mm = (58.0, 62.0)

    fig = create_figure_mm(*panel_size_mm)
    ax = fig.add_subplot(111)
    style_axes(ax, grid=False)
    metrics = METRIC_ORDER
    x = np.arange(len(metrics))
    dataset_offsets = {"snapshot_45": -0.04, "full_53": 0.04}
    markers = {"snapshot_45": "o", "full_53": "s"}
    for factor, color, base_offset in [
        ("Training data", FACTOR_COLORS["Training data"], -0.14),
        ("Architecture", FACTOR_COLORS["Architecture"], 0.14),
    ]:
        for dataset_key in ["snapshot_45", "full_53"]:
            subset = anova.loc[(anova["factor"] == factor) & (anova["dataset_key"] == dataset_key)].copy()
            subset["metric_label"] = pd.Categorical(subset["metric_label"], categories=metrics, ordered=True)
            subset = subset.sort_values("metric_label")
            xpos = x + base_offset + dataset_offsets[dataset_key]
            ax.plot(
                xpos,
                subset["partial_eta_sq"],
                marker=markers[dataset_key],
                ms=4,
                lw=1.1,
                color=color,
                alpha=0.9 if dataset_key == "snapshot_45" else 0.65,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["F1", "MAE", "DAF", r"R$^2$"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Partial η²")
    ax.set_title("")
    ax.legend(
        [
            Line2D([0], [0], color=FACTOR_COLORS["Training data"], marker="o", lw=1.1),
            Line2D([0], [0], color=FACTOR_COLORS["Architecture"], marker="o", lw=1.1),
            Line2D([0], [0], color="#444444", marker="o", lw=0),
            Line2D([0], [0], color="#444444", marker="s", lw=0),
        ],
        ["Training data", "Architecture", "Frozen 45", "Current 53"],
        loc="lower left",
        bbox_to_anchor=(0.00, 0.03),
        fontsize=5.8,
        ncol=2,
    )
    add_panel_label(ax, "a", x=-0.11, y=1.03)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.88, bottom=0.16)
    save_panel(fig, destinations, "panel_a_anova_sensitivity")

    fig = create_figure_mm(*panel_size_mm)
    ax = fig.add_subplot(111)
    style_axes(ax, grid=False)
    cluster_metrics = [
        ("ari_training_exact", "Training exact", FACTOR_COLORS["Training data"]),
        ("ari_training_coarse", "Training coarse", "#4C78A8"),
        ("ari_architecture", "Architecture", FACTOR_COLORS["Architecture"]),
    ]
    y_positions = np.arange(len(cluster_metrics))
    frozen_row = cluster.loc[cluster["dataset_key"] == "snapshot_45"].iloc[0]
    current_row = cluster.loc[cluster["dataset_key"] == "full_53"].iloc[0]
    label_offsets = {
        "ari_training_exact": 0.040,
        "ari_training_coarse": 0.030,
        "ari_architecture": 0.030,
    }
    for ypos, (column, label, color) in zip(y_positions, cluster_metrics):
        ax.plot([frozen_row[column], current_row[column]], [ypos, ypos], color="#B5B5B5", lw=1.2)
        ax.scatter(frozen_row[column], ypos, s=30, color=color, edgecolor="#222222", linewidth=0.4, zorder=3)
        ax.scatter(current_row[column], ypos, s=30, color="white", edgecolor=color, linewidth=1.0, zorder=3)
        ax.text(
            current_row[column] + label_offsets[column],
            ypos,
            f"{current_row[column]:.3f}",
            fontsize=5.5,
            color="#555555",
            va="center",
            ha="left",
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([item[1] for item in cluster_metrics], fontsize=6.5)
    ax.set_xlabel("ARI")
    ax.set_xlim(0.0, 0.8)
    ax.set_title("")
    ax.legend(
        [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="#222222"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555"),
        ],
        ["Frozen 45", "Current 53"],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=5.8,
    )
    add_panel_label(ax, "b", x=-0.11, y=1.03)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.88, bottom=0.16)
    save_panel(fig, destinations, "panel_b_cluster_sensitivity")

    fig = create_figure_mm(*panel_size_mm)
    ax = fig.add_subplot(111)
    style_axes(ax, grid=False)
    thresholds = thresholds.sort_values("failure_vote_threshold")
    xpos = np.arange(len(thresholds))
    width = 0.32
    ax.bar(xpos - width / 2, thresholds["collective_false_negative"], width=width, color="#C62828", alpha=0.75, label="False negative")
    ax.bar(xpos + width / 2, thresholds["collective_false_positive"], width=width, color="#F28E85", alpha=0.85, label="False positive")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{value:.1f}" for value in thresholds["failure_vote_threshold"]])
    ax.set_xlabel("Failure vote threshold")
    ax.set_ylabel("Count", color="#222222")
    ax.tick_params(axis="y", colors="#222222")
    ax.set_title("")
    ax2 = ax.twinx()
    ax2.plot(xpos, thresholds["decision_tree_balanced_accuracy"], color="#1F4E79", marker="o", ms=4, lw=1.3)
    ax2.set_ylim(0.60, 0.75)
    ax2.set_ylabel("Balanced accuracy", color="#222222")
    ax2.tick_params(axis="y", colors="#222222", labelsize=6.4)
    ax2.spines["right"].set_color("#222222")
    ax.legend(loc="upper right", fontsize=5.7)
    add_panel_label(ax, "c", x=-0.11, y=1.03)
    fig.subplots_adjust(left=0.18, right=0.82, top=0.88, bottom=0.16)
    save_panel(fig, destinations, "panel_c_threshold_sensitivity")

    compose_preview(
        package_dir,
        destinations,
        "figure_s05_preview",
        [
            ("panel_a_anova_sensitivity.png", [0.00, 0.07, 0.32, 0.86]),
            ("panel_b_cluster_sensitivity.png", [0.34, 0.07, 0.32, 0.86]),
            ("panel_c_threshold_sensitivity.png", [0.68, 0.07, 0.32, 0.86]),
        ],
        (180.0, 72.0),
    )
    write_notes(
        destinations,
        "Figure S5 Assembly Notes",
        [
            "Three-panel supplementary figure.",
            "Panels compare frozen-versus-current sensitivity for ANOVA, cluster alignment, and failure-threshold selection.",
        ],
    )


def main() -> None:
    configure_publication_style()
    make_s04_decision_tree()
    make_s05_sensitivity_overview()


if __name__ == "__main__":
    main()

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

from dva_project.figure_style import (
    add_panel_label,
    configure_publication_style,
    create_figure_mm,
    save_figure_to_many,
    style_axes,
)
from dva_project.settings import PROJECT_ROOT, RESULTS_DIR
from dva_project.utils import ensure_dir

from figure_supplementary_batch_b import FEATURE_NAME_MAP, S04_FOCUS_FEATURES


VARIANT_RESULTS_DIR = RESULTS_DIR / "figures"
VARIANT_PACKAGE_DIR = (
    PROJECT_ROOT / "paper" / "submission_package" / "figures" / "supplementary_variants"
)


def output_dirs(figure_name: str) -> tuple[list[Path], Path]:
    results_dir = VARIANT_RESULTS_DIR / figure_name
    package_dir = VARIANT_PACKAGE_DIR / figure_name
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


def make_s04_decision_tree_nature_variant() -> None:
    destinations, package_dir = output_dirs("figure_s04_decision_tree_nature_variant")
    importances = (
        pd.read_csv(RESULTS_DIR / "tables" / "analysis_04" / "decision_tree_feature_importance.csv")
        .sort_values("importance", ascending=False)
        .head(8)
    )
    threshold_top = pd.read_csv(
        RESULTS_DIR / "tables" / "sensitivity" / "analysis_04_threshold_top_features.csv"
    )
    summary = json.loads(
        (RESULTS_DIR / "tables" / "analysis_04" / "summary.json").read_text(encoding="utf-8")
    )

    panel_size_mm = (86.0, 70.0)

    fig = create_figure_mm(*panel_size_mm)
    ax = fig.add_subplot(111)
    style_axes(ax, grid=False)
    y = np.arange(len(importances))
    ax.barh(
        y,
        importances["importance"],
        color="#6288B6",
        edgecolor="#294B6B",
        linewidth=0.45,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([format_feature_name(value) for value in importances["feature"]], fontsize=6.2)
    ax.tick_params(axis="y", pad=4)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.37)
    ax.set_xlabel("Feature importance")
    ax.set_title("")
    ax.text(
        0.60,
        0.11,
        f"Balanced accuracy = {summary['decision_tree_balanced_accuracy']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.0,
        color="#333333",
    )
    add_panel_label(ax, "a", x=-0.11, y=1.03)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.88, bottom=0.14)
    save_panel(fig, destinations, "panel_a_tree_importance")

    pivot = threshold_top.pivot(index="feature", columns="failure_vote_threshold", values="rank")
    pivot = pivot.reindex(S04_FOCUS_FEATURES)

    fig = create_figure_mm(*panel_size_mm)
    ax = fig.add_subplot(111)
    style_axes(ax, grid=False)
    thresholds = [0.7, 0.8, 0.9]
    x = np.arange(len(thresholds))
    feature_styles = {
        "unique_prototype": {"color": "#14A37F", "lw": 2.1, "ms": 4.8, "zorder": 4},
        "spacegroup_number": {"color": "#AFAFAF", "lw": 0.95, "ms": 4.0, "zorder": 3},
        "n_sites": {"color": "#B7B7B7", "lw": 0.95, "ms": 4.0, "zorder": 3},
        "mean_atomic_number": {"color": "#B7B7B7", "lw": 0.95, "ms": 4.0, "zorder": 3},
        "std_electronegativity": {"color": "#D0D0D0", "lw": 0.85, "ms": 3.8, "zorder": 2},
    }
    marker_map = {
        "unique_prototype": "o",
        "spacegroup_number": "s",
        "n_sites": "D",
        "mean_atomic_number": "^",
        "std_electronegativity": "v",
    }
    label_y_adjust = {
        "unique_prototype": 0.00,
        "spacegroup_number": 0.00,
        "n_sites": -0.22,
        "mean_atomic_number": 0.08,
    }
    label_x = x[-1] + 0.14

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
        anchor_y = yvals[valid_idx[-1]] + label_y_adjust.get(feature, 0.0)
        ax.text(
            label_x,
            anchor_y,
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
    ax.set_xlim(-0.10, 2.78)
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
        "Figure S4 Nature Variant Notes",
        [
            "Independent supplementary variant generated without overwriting the original Figure S4.",
            "Layout follows the project Nature plot standardizer: equal panel sizes, no non-semantic grid lines, consistent panel-label anchors, lighter secondary lines, and right-column label alignment.",
        ],
    )


def main() -> None:
    configure_publication_style()
    make_s04_decision_tree_nature_variant()


if __name__ == "__main__":
    main()

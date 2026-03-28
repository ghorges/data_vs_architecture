from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from dva_project.figure_style import (
    FACTOR_COLORS,
    add_panel_label,
    add_underbar_marker,
    configure_publication_style,
    create_figure_mm,
    save_figure_to_many,
    style_axes,
)
from dva_project.settings import PROJECT_ROOT, RESULTS_DIR
from dva_project.utils import ensure_dir


PANEL_SIZE_MM = (108.0, 82.0)
PREVIEW_SIZE_MM = (220.0, 88.0)
FACTOR_ORDER = ["Training data", "Architecture", "Parameters"]
METRIC_ORDER = ["F1", "MAE", "DAF", "R2"]
METRIC_DISPLAY = {"F1": "F1", "MAE": "MAE", "DAF": "DAF", "R2": r"$R^2$"}


def load_variance_table() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS_DIR / "tables" / "sensitivity" / "analysis_01_snapshot_comparison.csv")
    frame = frame.loc[frame["factor"].isin(FACTOR_ORDER)].copy()
    frame["snapshot"] = frame["dataset_key"].map({"snapshot_45": "Frozen 45", "full_53": "Live 53"})
    return frame


def load_permutation_pvalues() -> dict[str, float]:
    frame = pd.read_csv(RESULTS_DIR / "tables" / "analysis_24" / "anova_permutation_summary.csv")
    frame = frame.loc[frame["scope"] == "snapshot_45"].copy()
    return {row.metric_label: float(row.train_minus_arch_p_one_sided_ge) for row in frame.itertuples(index=False)}


def format_p_value(value: float) -> str:
    if value >= 0.01:
        return f"{value:.3f}"
    if value >= 0.001:
        return f"{value:.4f}"
    return f"{value:.2e}"


def get_values(variance: pd.DataFrame, snapshot: str, factor: str) -> np.ndarray:
    subset = variance.loc[
        (variance["snapshot"] == snapshot) & (variance["factor"] == factor),
        ["metric_label", "partial_eta_sq"],
    ].set_index("metric_label")
    return np.array([subset.loc[metric, "partial_eta_sq"] for metric in METRIC_ORDER], dtype=float)


def draw_simplified(ax, variance: pd.DataFrame, p_values: dict[str, float]) -> None:
    style_axes(ax, grid=False)
    metric_centers = np.arange(len(METRIC_ORDER), dtype=float)
    offsets = np.array([-0.24, 0.0, 0.24], dtype=float)
    bar_width = 0.18

    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-0.65, len(METRIC_ORDER) - 0.35)
    ax.set_ylabel(r"Partial $\eta^2$")
    ax.set_xticks(metric_centers)
    ax.set_xticklabels([METRIC_DISPLAY[metric] for metric in METRIC_ORDER])
    ax.set_title("Figure 2a variant: simplified bars + live markers", loc="left")

    for factor_index, factor in enumerate(FACTOR_ORDER):
        frozen_values = get_values(variance, "Frozen 45", factor)
        live_values = get_values(variance, "Live 53", factor)
        positions = metric_centers + offsets[factor_index]

        bars = ax.bar(
            positions,
            frozen_values,
            width=bar_width,
            color=FACTOR_COLORS[factor],
            edgecolor="#222222",
            linewidth=0.5,
            zorder=3,
        )
        for x_pos, live_value in zip(positions, live_values):
            add_underbar_marker(ax, x=float(x_pos), y=float(live_value), width=bar_width * 0.92, color="#111111", linewidth=1.3)
        for bar in bars:
            bar.set_joinstyle("miter")

    for index, metric in enumerate(METRIC_ORDER):
        frozen_value = variance.loc[
            (variance["snapshot"] == "Frozen 45") & (variance["factor"] == "Training data") & (variance["metric_label"] == metric),
            "partial_eta_sq",
        ].iloc[0]
        live_value = variance.loc[
            (variance["snapshot"] == "Live 53") & (variance["factor"] == "Training data") & (variance["metric_label"] == metric),
            "partial_eta_sq",
        ].iloc[0]
        ax.text(
            metric_centers[index] + offsets[0],
            max(frozen_value, live_value) + 0.045,
            f"p={format_p_value(p_values[metric])}",
            fontsize=6.7,
            ha="center",
            va="bottom",
            color="#333333",
        )

    ax.text(0.02, 0.98, "Bars = Frozen 45 | black marker = Live 53", transform=ax.transAxes, fontsize=6.8, ha="left", va="top", color="#333333")
    factor_handles = [Patch(facecolor=FACTOR_COLORS[factor], edgecolor="#222222", linewidth=0.5, label=factor) for factor in FACTOR_ORDER]
    ax.legend(
        handles=factor_handles,
        loc="upper center",
        bbox_to_anchor=(0.44, -0.14),
        fontsize=6.6,
        ncol=3,
        handlelength=1.2,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    add_panel_label(ax, "a")


def draw_grouped(ax, variance: pd.DataFrame, p_values: dict[str, float]) -> None:
    style_axes(ax, grid=False)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel(r"Partial $\eta^2$")
    ax.set_title("Figure 2a variant: grouped frozen/live bars", loc="left")

    metric_centers = np.arange(len(METRIC_ORDER), dtype=float) * 1.45
    group_offsets = {"Frozen 45": -0.26, "Live 53": 0.26}
    factor_offsets = np.array([-0.16, 0.0, 0.16], dtype=float)
    bar_width = 0.12

    for snapshot, hatch, line_width in [("Frozen 45", None, 0.5), ("Live 53", "///", 0.8)]:
        for factor_index, factor in enumerate(FACTOR_ORDER):
            heights = get_values(variance, snapshot, factor)
            positions = metric_centers + group_offsets[snapshot] + factor_offsets[factor_index]
            ax.bar(
                positions,
                heights,
                width=bar_width,
                color=FACTOR_COLORS[factor],
                edgecolor="#222222",
                linewidth=line_width,
                hatch=hatch,
                zorder=3,
            )

    for index, metric in enumerate(METRIC_ORDER):
        frozen_training = variance.loc[
            (variance["snapshot"] == "Frozen 45") & (variance["factor"] == "Training data") & (variance["metric_label"] == metric),
            "partial_eta_sq",
        ].iloc[0]
        ax.text(
            metric_centers[index] + group_offsets["Frozen 45"] + factor_offsets[0],
            frozen_training + 0.045,
            f"p={format_p_value(p_values[metric])}",
            fontsize=6.5,
            ha="center",
            va="bottom",
            color="#333333",
        )
        ax.text(metric_centers[index] + group_offsets["Frozen 45"], -0.075, "Frozen", fontsize=6.1, ha="center", va="top", color="#666666")
        ax.text(metric_centers[index] + group_offsets["Live 53"], -0.075, "Live", fontsize=6.1, ha="center", va="top", color="#666666")

    ax.set_xticks(metric_centers)
    ax.set_xticklabels([METRIC_DISPLAY[metric] for metric in METRIC_ORDER])
    ax.set_xlim(metric_centers[0] - 0.55, metric_centers[-1] + 0.55)

    factor_handles = [Patch(facecolor=FACTOR_COLORS[factor], edgecolor="#222222", linewidth=0.5, label=factor) for factor in FACTOR_ORDER]
    snapshot_handles = [
        Patch(facecolor="white", edgecolor="#222222", linewidth=0.8, label="Frozen 45"),
        Patch(facecolor="white", edgecolor="#222222", linewidth=0.8, hatch="///", label="Live 53"),
    ]
    ax.legend(
        handles=factor_handles + snapshot_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.14),
        fontsize=6.3,
        ncol=5,
        handlelength=1.4,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    add_panel_label(ax, "a")


def create_panel(output_stems: list[Path], draw_fn, variance: pd.DataFrame, p_values: dict[str, float]) -> None:
    fig = create_figure_mm(*PANEL_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_fn(ax, variance, p_values)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.24)
    save_figure_to_many(fig, output_stems)


def create_preview(output_stems: list[Path], package_dir: Path) -> None:
    fig = create_figure_mm(*PREVIEW_SIZE_MM)
    fig.patch.set_facecolor("white")
    placements = [
        ("figure_02a_simplified.png", [0.00, 0.04, 0.49, 0.92]),
        ("figure_02a_grouped_dual.png", [0.51, 0.04, 0.49, 0.92]),
    ]
    for file_name, position in placements:
        axis = fig.add_axes(position)
        axis.imshow(plt.imread(package_dir / file_name))
        axis.axis("off")
    save_figure_to_many(fig, output_stems)


def write_notes(output_dirs: list[Path]) -> None:
    contents = "\n".join(
        [
            "# Figure 2a Variant Notes",
            "",
            "Two standalone variants exported for visual comparison.",
            "",
            "- `figure_02a_simplified.pdf`: frozen bars + live markers",
            "- `figure_02a_grouped_dual.pdf`: frozen/live grouped bars with hatching",
            "- `figure_02a_variant_preview.pdf`: side-by-side comparison preview",
            "",
        ]
    )
    for output_dir in output_dirs:
        (output_dir / "notes.md").write_text(contents, encoding="utf-8")


def main() -> None:
    configure_publication_style()

    results_dir = RESULTS_DIR / "figures" / "figure_02a_variants"
    package_dir = PROJECT_ROOT / "paper" / "submission_package" / "figures" / "main" / "figure_02a_variants"
    output_dirs = [results_dir, package_dir]
    for output_dir in output_dirs:
        ensure_dir(output_dir)

    variance = load_variance_table()
    p_values = load_permutation_pvalues()

    create_panel([output_dir / "figure_02a_simplified" for output_dir in output_dirs], draw_simplified, variance, p_values)
    create_panel([output_dir / "figure_02a_grouped_dual" for output_dir in output_dirs], draw_grouped, variance, p_values)
    create_preview([output_dir / "figure_02a_variant_preview" for output_dir in output_dirs], package_dir)
    write_notes(output_dirs)


if __name__ == "__main__":
    main()

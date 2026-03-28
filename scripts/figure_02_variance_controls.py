from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from PIL import Image

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


PANEL_A_SIZE_MM = (182.88, 106.68)
PANEL_B_SIZE_MM = (86.36, 76.20)
PANEL_C_SIZE_MM = (86.36, 76.20)
PREVIEW_SIZE_MM = (180.0, 80.0)

FACTOR_ORDER = ["Training data", "Architecture", "Parameters"]
METRIC_ORDER = ["F1", "MAE", "DAF", "R2"]
PAIR_METRIC_ORDER = ["F1", "MAE", "DAF"]
METRIC_DISPLAY = {
    "F1": "F1",
    "MAE": "MAE",
    "DAF": "DAF",
    "R2": "R²",
}
PAIR_DOT_COLORS = {
    "same_data_diff_architecture": "#0072B2",
    "same_family_diff_data": "#D55E00",
}
PAIR_LABELS = {
    "same_data_diff_architecture": "Same data, different architecture (n=197)",
    "same_family_diff_data": "Same family, different data (n=22)",
}
VERTICAL_LAYOUT_GAP_X = 44
VERTICAL_LAYOUT_GAP_Y = 50
HORIZONTAL_LAYOUT_GAP_X = 50
HORIZONTAL_LAYOUT_GAP_Y = 41


def load_variance_table() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS_DIR / "tables" / "sensitivity" / "analysis_01_snapshot_comparison.csv")
    frame = frame.loc[frame["factor"].isin(FACTOR_ORDER)].copy()
    frame["snapshot"] = frame["dataset_key"].map(
        {
            "snapshot_45": "Frozen 45",
            "full_53": "Live 53",
        }
    )
    return frame


def load_permutation_pvalues() -> dict[str, float]:
    frame = pd.read_csv(RESULTS_DIR / "tables" / "analysis_24" / "anova_permutation_summary.csv")
    frame = frame.loc[frame["scope"] == "snapshot_45"].copy()
    return {
        row.metric_label: float(row.train_minus_arch_p_one_sided_ge)
        for row in frame.itertuples(index=False)
    }


def load_pair_summary() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS_DIR / "tables" / "analysis_01_control" / "pair_delta_summary.csv")
    return frame.loc[frame["metric_label"].isin(PAIR_METRIC_ORDER)].copy()


def load_permutation_panel() -> tuple[pd.Series, dict[str, float]]:
    draws = pd.read_csv(RESULTS_DIR / "tables" / "analysis_24" / "anova_permutation_draws.csv")
    observed = pd.read_csv(RESULTS_DIR / "tables" / "analysis_24" / "anova_observed.csv")
    summary = pd.read_csv(RESULTS_DIR / "tables" / "analysis_24" / "anova_permutation_summary.csv")

    draws = draws.loc[
        (draws["scope"] == "snapshot_45") & (draws["metric"] == "mae_full_test"),
        "train_minus_arch",
    ].copy()
    observed_row = observed.loc[
        (observed["scope"] == "snapshot_45") & (observed["metric"] == "mae_full_test")
    ].iloc[0]
    summary_row = summary.loc[
        (summary["scope"] == "snapshot_45") & (summary["metric"] == "mae_full_test")
    ].iloc[0]
    return draws, {
        "observed": float(observed_row["train_minus_arch"]),
        "p_value": float(summary_row["train_minus_arch_p_one_sided_ge"]),
        "null_mean": float(summary_row["train_minus_arch_null_mean"]),
        "n_permutations": int(len(draws)),
    }


def format_p_value(value: float) -> str:
    if value >= 0.0001:
        return f"{value:.4f}"
    return f"{value:.2e}"


def draw_panel_a(ax, variance: pd.DataFrame, p_values: dict[str, float]) -> None:
    style_axes(ax, grid=False)
    ax.axhline(0.0, color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Partial η²")
    metric_centers = np.arange(len(METRIC_ORDER), dtype=float) * 1.20
    factor_offsets = np.array([-0.18, 0.0, 0.18], dtype=float)
    bar_width = 0.13

    for factor_index, factor in enumerate(FACTOR_ORDER):
        frozen_subset = variance.loc[
            (variance["snapshot"] == "Frozen 45") & (variance["factor"] == factor),
            ["metric_label", "partial_eta_sq"],
        ].set_index("metric_label")
        live_subset = variance.loc[
            (variance["snapshot"] == "Live 53") & (variance["factor"] == factor),
            ["metric_label", "partial_eta_sq"],
        ].set_index("metric_label")
        frozen_values = np.array([frozen_subset.loc[metric, "partial_eta_sq"] for metric in METRIC_ORDER], dtype=float)
        live_values = np.array([live_subset.loc[metric, "partial_eta_sq"] for metric in METRIC_ORDER], dtype=float)
        positions = metric_centers + factor_offsets[factor_index]
        ax.bar(
            positions,
            frozen_values,
            width=bar_width,
            color=FACTOR_COLORS[factor],
            edgecolor="#222222",
            linewidth=0.5,
            zorder=3,
        )
        for xpos, live_value in zip(positions, live_values):
            inset = bar_width * 0.012
            live_line = ax.plot(
                [float(xpos) - (bar_width / 2.0) + inset, float(xpos) + (bar_width / 2.0) - inset],
                [float(live_value), float(live_value)],
                color="#222222",
                linewidth=1.8,
                solid_capstyle="butt",
                zorder=4,
            )[0]
            live_line.set_path_effects([pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()])
        if factor == "Parameters":
            for xpos, frozen_value, live_value in zip(positions, frozen_values, live_values):
                display_value = max(frozen_value, live_value)
                label = "<0.01" if display_value < 0.01 else f"{display_value:.2f}"
                ax.text(
                    xpos,
                    display_value + 0.012,
                    label,
                    fontsize=5.8,
                    ha="center",
                    va="bottom",
                    color="#666666",
                )

    bracket_y = 0.915
    bracket_drop = 0.012
    p_y = bracket_y + 0.010
    for index, metric in enumerate(METRIC_ORDER):
        train_x = metric_centers[index] + factor_offsets[0]
        arch_x = metric_centers[index] + factor_offsets[1]
        ax.plot(
            [train_x, train_x, arch_x, arch_x],
            [bracket_y - bracket_drop, bracket_y, bracket_y, bracket_y - bracket_drop],
            color="#444444",
            linewidth=0.9,
            zorder=4,
        )
        ax.text(
            (train_x + arch_x) / 2.0,
            p_y,
            f"p={format_p_value(p_values[metric])}",
            fontsize=6.8,
            ha="center",
            va="bottom",
            color="#333333",
        )
    ax.set_xticks(metric_centers)
    ax.set_xticklabels([""] * len(METRIC_ORDER))
    label_transform = ax.get_xaxis_transform()
    for xpos, metric in zip(metric_centers, METRIC_ORDER):
        ax.text(
            xpos,
            -0.052,
            METRIC_DISPLAY[metric],
            transform=label_transform,
            fontsize=8,
            ha="center",
            va="top",
            color="#111111",
        )
    ax.set_xlim(metric_centers[0] - 0.45, metric_centers[-1] + 0.45)
    ax.tick_params(axis="x", pad=2)
    add_panel_label(ax, "a")


def draw_panel_b(axes: list, comparison: pd.DataFrame) -> None:
    for axis, metric in zip(axes, PAIR_METRIC_ORDER):
        style_axes(axis, show_top=False, show_right=False, grid=False)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        row_same_arch = comparison.loc[
            (comparison["experiment_type"] == "same_data_diff_architecture")
            & (comparison["metric_label"] == metric)
        ].iloc[0]
        row_same_data = comparison.loc[
            (comparison["experiment_type"] == "same_family_diff_data")
            & (comparison["metric_label"] == metric)
        ].iloc[0]
        x_left = float(row_same_arch["mean_abs_delta"])
        x_right = float(row_same_data["mean_abs_delta"])
        ratio = x_right / x_left
        x_limit = max(x_left, x_right) * 1.40

        axis.set_xlim(0.0, x_limit)
        axis.set_ylim(-0.7, 0.7)
        axis.set_yticks([])
        axis.hlines(0.0, x_left, x_right, color="#BDBDBD", linewidth=0.7, zorder=1)
        axis.scatter(
            [x_left],
            [0.0],
            s=32,
            color=PAIR_DOT_COLORS["same_data_diff_architecture"],
            zorder=3,
        )
        axis.scatter(
            [x_right],
            [0.0],
            s=32,
            color=PAIR_DOT_COLORS["same_family_diff_data"],
            zorder=3,
        )
        axis.text(
            -0.03,
            0.5,
            metric,
            transform=axis.transAxes,
            fontsize=8,
            ha="right",
            va="center",
            color="#111111",
        )
        axis.text(
            x_left,
            0.20,
            f"{x_left:.4f}",
            fontsize=7.0,
            ha="center",
            va="bottom",
            color=PAIR_DOT_COLORS["same_data_diff_architecture"],
        )
        axis.text(
            x_right,
            0.20,
            f"{x_right:.4f}",
            fontsize=7.0,
            ha="center",
            va="bottom",
            color=PAIR_DOT_COLORS["same_family_diff_data"],
        )
        axis.text(
            x_limit * 0.98,
            0.0,
            f"{ratio:.1f}x",
            fontsize=7.2,
            fontweight="bold",
            ha="right",
            va="center",
            color="#222222",
        )

    axes[-1].set_xlabel("Mean absolute difference")


def draw_panel_c(ax, draws: pd.Series, summary: dict[str, float]) -> None:
    style_axes(ax, grid=False)
    observed = summary["observed"]
    max_x = max(float(draws.max()), observed) * 1.05
    min_x = min(float(draws.min()), 0.0)
    ax.hist(
        draws,
        bins=30,
        color="#AAAAAA",
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
    )
    ax.set_xlim(min_x, max_x)
    ax.set_xlabel(r"Training $\eta^2$ - Architecture $\eta^2$")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.text(
        0.85,
        0.95,
        "\n".join(
            [
                f"p = {format_p_value(summary['p_value'])}",
                f"Observed = {observed:.3f}",
                f"{summary['n_permutations']:,} permutations",
            ]
        ),
        transform=ax.transAxes,
        fontsize=6.8,
        ha="right",
        va="top",
        linespacing=1.38,
        color="#222222",
        bbox={"facecolor": "none", "edgecolor": "none", "pad": 0.0},
    )
    add_panel_label(ax, "c")


def create_panel_a(output_stems: list[Path], variance: pd.DataFrame, p_values: dict[str, float]) -> None:
    fig = create_figure_mm(*PANEL_A_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_a(ax, variance, p_values)
    legend_handles = [
        Patch(facecolor=FACTOR_COLORS[factor], edgecolor="#222222", linewidth=0.5, label=factor)
        for factor in FACTOR_ORDER
    ] + [
        Patch(facecolor="white", edgecolor="#222222", linewidth=0.8, label="Frozen 45"),
        Line2D([0], [0], color="#222222", marker="_", markersize=12, linewidth=0.0, label="Live 53"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.955),
        fontsize=6.6,
        ncol=5,
        handlelength=1.3,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.18)
    save_figure_to_many(fig, output_stems)


def create_panel_b(output_stems: list[Path], comparison: pd.DataFrame) -> None:
    fig = create_figure_mm(*PANEL_B_SIZE_MM)
    axes = fig.subplots(3, 1, sharex=False)
    draw_panel_b(list(axes), comparison)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAIR_DOT_COLORS[key], markersize=5, label=label)
        for key in ["same_family_diff_data", "same_data_diff_architecture"]
        for label in [PAIR_LABELS[key]]
    ]
    fig.text(
        0.02,
        0.965,
        "b",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111111",
    )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.14, 0.94),
        fontsize=6.5,
        handletextpad=0.5,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.80, bottom=0.14, hspace=0.88)
    save_figure_to_many(fig, output_stems)


def create_panel_c(output_stems: list[Path], draws: pd.Series, summary: dict[str, float]) -> None:
    fig = create_figure_mm(*PANEL_C_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_c(ax, draws, summary)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.88, bottom=0.20)
    save_figure_to_many(fig, output_stems)


def create_preview(output_stems: list[Path], package_dir: Path) -> None:
    fig = create_figure_mm(*PREVIEW_SIZE_MM)
    fig.patch.set_facecolor("white")

    placements = [
        ("panel_a_grouped_variance.png", [0.01, 0.05, 0.55, 0.90]),
        ("panel_b_paired_controls.png", [0.60, 0.56, 0.38, 0.34]),
        ("panel_c_permutation_null_mae.png", [0.60, 0.10, 0.38, 0.30]),
    ]
    for file_name, position in placements:
        image_path = package_dir / file_name
        axis = fig.add_axes(position)
        axis.imshow(plt.imread(image_path), aspect="auto")
        axis.axis("off")

    save_figure_to_many(fig, output_stems)


def create_composite_layouts(output_dirs: list[Path]) -> None:
    for output_dir in output_dirs:
        panel_a = Image.open(output_dir / "panel_a_grouped_variance.png").convert("RGB")
        panel_b = Image.open(output_dir / "panel_b_paired_controls.png").convert("RGB")
        panel_c = Image.open(output_dir / "panel_c_permutation_null_mae.png").convert("RGB")

        vertical_row_width = panel_b.width + VERTICAL_LAYOUT_GAP_X + panel_c.width
        vertical_bottom_height = max(panel_b.height, panel_c.height)
        vertical_canvas_width = max(panel_a.width, vertical_row_width)
        vertical_canvas = Image.new(
            "RGB",
            (
                vertical_canvas_width,
                panel_a.height + VERTICAL_LAYOUT_GAP_Y + vertical_bottom_height,
            ),
            "white",
        )
        vertical_a_x = (vertical_canvas_width - panel_a.width) // 2
        vertical_canvas.paste(panel_a, (vertical_a_x, 0))
        vertical_y = panel_a.height + VERTICAL_LAYOUT_GAP_Y
        vertical_row_x = (vertical_canvas_width - vertical_row_width) // 2
        vertical_canvas.paste(
            panel_b,
            (vertical_row_x, vertical_y + vertical_bottom_height - panel_b.height),
        )
        vertical_canvas.paste(
            panel_c,
            (
                vertical_row_x + panel_b.width + VERTICAL_LAYOUT_GAP_X,
                vertical_y + vertical_bottom_height - panel_c.height,
            ),
        )
        vertical_canvas.save(output_dir / "figure_02_layout_vertical.png")

        horizontal_column_width = max(panel_b.width, panel_c.width)
        horizontal_stack_height = panel_b.height + HORIZONTAL_LAYOUT_GAP_Y + panel_c.height
        horizontal_canvas = Image.new(
            "RGB",
            (
                panel_a.width + HORIZONTAL_LAYOUT_GAP_X + horizontal_column_width,
                max(panel_a.height, horizontal_stack_height),
            ),
            "white",
        )
        horizontal_a_y = (horizontal_canvas.height - panel_a.height) // 2
        horizontal_canvas.paste(panel_a, (0, horizontal_a_y))
        horizontal_x = panel_a.width + HORIZONTAL_LAYOUT_GAP_X
        horizontal_stack_y = (horizontal_canvas.height - horizontal_stack_height) // 2
        horizontal_b_x = horizontal_x + (horizontal_column_width - panel_b.width) // 2
        horizontal_c_x = horizontal_x + (horizontal_column_width - panel_c.width) // 2
        horizontal_canvas.paste(panel_b, (horizontal_b_x, horizontal_stack_y))
        horizontal_canvas.paste(
            panel_c,
            (horizontal_c_x, horizontal_stack_y + panel_b.height + HORIZONTAL_LAYOUT_GAP_Y),
        )
        horizontal_canvas.save(output_dir / "figure_02_layout_horizontal.png")


def write_assembly_notes(output_dirs: list[Path]) -> None:
    contents = "\n".join(
        [
            "# Figure 2 Assembly Notes",
            "",
            "Panels exported as individual PDF and PNG assets.",
            "",
            "Suggested layout:",
            "- Top row: `panel_a_grouped_variance.pdf`",
            "- Bottom left: `panel_b_paired_controls.pdf`",
            "- Bottom right: `panel_c_permutation_null_mae.pdf`",
            "",
            "Suggested proportions from the guide:",
            "- panel a: full-width top row",
            "- panel b: lower row left half",
            "- panel c: lower row right half",
            "",
            "Panel target sizes used during export:",
            f"- panel a: {PANEL_A_SIZE_MM[0]:.0f} mm x {PANEL_A_SIZE_MM[1]:.0f} mm",
            f"- panel b: {PANEL_B_SIZE_MM[0]:.0f} mm x {PANEL_B_SIZE_MM[1]:.0f} mm",
            f"- panel c: {PANEL_C_SIZE_MM[0]:.0f} mm x {PANEL_C_SIZE_MM[1]:.0f} mm",
            "",
            "Preview file:",
            "- `figure_02_preview.png` / `figure_02_preview.pdf`",
            "",
            "Composite PNG files:",
            "- `figure_02_layout_vertical.png`",
            "- `figure_02_layout_horizontal.png`",
            "- panels are pasted at native exported size without resizing",
            "",
            "Data sources:",
            "- `results/tables/sensitivity/analysis_01_snapshot_comparison.csv`",
            "- `results/tables/analysis_01_control/pair_delta_summary.csv`",
            "- `results/tables/analysis_24/anova_observed.csv`",
            "- `results/tables/analysis_24/anova_permutation_draws.csv`",
            "- `results/tables/analysis_24/anova_permutation_summary.csv`",
            "",
        ]
    )
    for output_dir in output_dirs:
        ensure_dir(output_dir)
        (output_dir / "assembly_notes.md").write_text(contents, encoding="utf-8")


def main() -> None:
    configure_publication_style()

    results_dir = RESULTS_DIR / "figures" / "figure_02_variance_controls"
    package_dir = (
        PROJECT_ROOT
        / "paper"
        / "submission_package"
        / "figures"
        / "main"
        / "figure_02_variance_controls"
    )
    output_dirs = [results_dir, package_dir]
    for output_dir in output_dirs:
        ensure_dir(output_dir)

    variance = load_variance_table()
    p_values = load_permutation_pvalues()
    comparison = load_pair_summary()
    draws, permutation_summary = load_permutation_panel()

    create_panel_a(
        [output_dir / "panel_a_grouped_variance" for output_dir in output_dirs],
        variance,
        p_values,
    )
    create_panel_b(
        [output_dir / "panel_b_paired_controls" for output_dir in output_dirs],
        comparison,
    )
    create_panel_c(
        [output_dir / "panel_c_permutation_null_mae" for output_dir in output_dirs],
        draws,
        permutation_summary,
    )
    create_preview(
        [output_dir / "figure_02_preview" for output_dir in output_dirs],
        package_dir,
    )
    create_composite_layouts(output_dirs)
    write_assembly_notes(output_dirs)


if __name__ == "__main__":
    main()

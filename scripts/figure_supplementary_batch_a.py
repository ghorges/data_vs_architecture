from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter

from dva_project.figure_style import (
    FACTOR_COLORS,
    TRAINING_REGIME_COLORS,
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

FEATURE_NAME_MAP = {
    "min_l1_distance_same_element_set_filled": "Nearest-set distance",
    "exact_formula_in_mptrj": "Exact formula (MPtrj)",
    "same_element_set_in_mptrj": "Same elements (MPtrj)",
    "n_sites": "Number of sites",
    "spacegroup_number": "Space group number",
    "std_electronegativity": "Electronegativity std",
    "mean_atomic_number": "Mean atomic number",
    "std_atomic_radius": "Atomic radius std",
}
TRAINING_LABEL_MAP = {
    "MPtrj": "MPtrj only",
    "MPtrj__sAlex": "MPtrj+sAlex",
    "MPtrj__OMat24": "MPtrj+Alex+OMat24",
    "Alex__MPtrj": "MPtrj+Alex+OMat24",
    "Alex__MPtrj__OMat24": "MPtrj+Alex+OMat24",
    "MPtrj__OMat24__sAlex": "OMat24+sAlex+MPtrj",
    "MatterSim": "MatterSim",
    "OpenLAM": "OpenLAM",
    "COSMOSDataset": "COSMOSDataset",
    "MPF": "MPF/MP2022",
    "MP 2022": "MPF/MP2022",
    "MP Graphs": "MPF/MP2022",
    "GNoME": "MPF/MP2022",
}
TRAINING_LEGEND_ORDER = [
    "OMat24+sAlex+MPtrj",
    "MPtrj+Alex+OMat24",
    "MPtrj+sAlex",
    "MPtrj only",
    "MatterSim",
    "OpenLAM",
    "COSMOSDataset",
    "MPF/MP2022",
]
ARCHITECTURE_COLOR_MAP = {
    "equivariant_gnn": "#14A37F",
    "invariant_gnn": "#4C78A8",
    "transformer": "#E69F00",
    "hybrid_ensemble": "#7A5195",
    "non_gnn": "#8F8F8F",
}
ARCHITECTURE_LABEL_MAP = {
    "equivariant_gnn": "Equivariant GNN",
    "invariant_gnn": "Invariant GNN",
    "transformer": "Transformer",
    "hybrid_ensemble": "Hybrid",
    "non_gnn": "Non-GNN",
}
ARCHITECTURE_LEGEND_ORDER = [
    "equivariant_gnn",
    "invariant_gnn",
    "transformer",
    "hybrid_ensemble",
    "non_gnn",
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


def get_s01_legend_items(assignments: pd.DataFrame) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    training_items = [
        (label, TRAINING_REGIME_COLORS[label])
        for label in TRAINING_LEGEND_ORDER
        if label in assignments["training_regime_label"].unique()
    ]
    architecture_items = [
        (ARCHITECTURE_LABEL_MAP[key], ARCHITECTURE_COLOR_MAP[key])
        for key in ARCHITECTURE_LEGEND_ORDER
        if key in assignments["architecture_group"].unique()
    ]
    return training_items, architecture_items


def wrap_s01_legend_label(label: str) -> str:
    wrapped = {
        "OMat24+sAlex+MPtrj": "OMat24+sAlex+\nMPtrj",
        "MPtrj+Alex+OMat24": "MPtrj+Alex+\nOMat24",
    }
    return wrapped.get(label, label)


def compact_tick_positions(candidates: list[int], *, min_gap: int, last_index: int) -> list[int]:
    kept: list[int] = []
    for candidate in sorted(set(candidates)):
        if not kept or candidate - kept[-1] >= min_gap:
            kept.append(candidate)
    if not kept or kept[0] != 0:
        kept.insert(0, 0)
    if kept[-1] != last_index:
        kept.append(last_index)
    return kept


def draw_s01_legend_section(
    ax,
    title: str,
    items: list[tuple[str, str]],
    *,
    y_top: float,
    ncols: int = 2,
) -> float:
    ax.text(
        0.0,
        y_top,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        fontweight="bold",
        color="#111111",
        fontfamily="Arial",
    )
    y_cursor = y_top - 0.11
    row_gap = 0.034
    square_w = 0.028
    square_h = 0.038
    text_dx = 0.038
    col_x = np.array([0.0, 0.60]) if ncols == 2 else np.linspace(0.0, 0.72, num=ncols)
    nrows = math.ceil(len(items) / ncols)
    wrapped_items = [(wrap_s01_legend_label(label), color) for label, color in items]
    for row in range(nrows):
        row_items = wrapped_items[row * ncols : (row + 1) * ncols]
        row_lines = max(label.count("\n") + 1 for label, _ in row_items)
        y_center = y_cursor - 0.022 * (row_lines - 1)
        for col, (label, color) in enumerate(row_items):
            x_pos = float(col_x[col])
            ax.add_patch(
                Rectangle(
                    (x_pos, y_center - square_h / 2),
                    square_w,
                    square_h,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="#777777",
                    linewidth=0.25,
                    clip_on=False,
                )
            )
            ax.text(
                x_pos + text_dx,
                y_center,
                label,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=6.0,
                linespacing=1.0,
                color="#222222",
                fontfamily="Arial",
            )
        y_cursor -= row_lines * row_gap + 0.060
    return y_cursor - 0.010


def draw_s01_compact_legends(ax, assignments: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    training_items, architecture_items = get_s01_legend_items(assignments)
    next_y = draw_s01_legend_section(ax, "Training data", training_items, y_top=0.98, ncols=2)
    draw_s01_legend_section(ax, "Architecture", architecture_items, y_top=next_y, ncols=2)


def draw_s01_ari_block(ax, summary: dict[str, float]) -> None:
    entries = [
        ("ARI training exact", float(summary["ari_training_exact"])),
        ("ARI training coarse", float(summary["ari_training_coarse"])),
        ("ARI architecture", float(summary["ari_architecture"])),
    ]
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.0,
        0.98,
        "Adjusted Rand index",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        fontweight="bold",
        color="#111111",
        fontfamily="Arial",
    )
    for (label, value), y_pos in zip(entries, [0.72, 0.42, 0.12]):
        ax.text(
            0.0,
            y_pos,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.5,
            color="#333333",
            fontfamily="Arial",
        )
        ax.text(
            0.92,
            y_pos,
            f"{value:0.3f}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=6.5,
            color="#333333",
            fontfamily="Arial",
        )


def make_s01_error_clustermap() -> None:
    destinations, package_dir = output_dirs("figure_s01_error_clustermap")
    matrix = pd.read_csv(RESULTS_DIR / "tables" / "analysis_02" / "error_correlation_matrix.csv", index_col=0)
    assignments = pd.read_csv(RESULTS_DIR / "tables" / "analysis_02" / "cluster_assignments.csv")
    summary = json.loads((RESULTS_DIR / "tables" / "analysis_02" / "summary.json").read_text(encoding="utf-8"))

    assignments["training_regime_label"] = assignments["training_combo"].map(TRAINING_LABEL_MAP).fillna("MPF/MP2022")
    assignments = assignments.sort_values(["cluster_training_exact", "cluster_order", "f1_full_test"], ascending=[True, True, False])
    order = assignments["model_key"].tolist()
    matrix = matrix.loc[order, order]
    labels = assignments["model_name"].fillna(assignments["model_key"]).tolist()

    corr_min = float(matrix.to_numpy().min())
    corr_floor = min(-0.01, corr_min)
    vmax = 1.0
    zero_fraction = (0.0 - corr_floor) / (vmax - corr_floor)
    cmap = LinearSegmentedColormap.from_list(
        "corr_diverging_refined",
        [
            (0.0, "#496A8A"),
            (zero_fraction * 0.40, "#89A8C4"),
            (zero_fraction, "#F4F1EC"),
            (0.35, "#E2B9BF"),
            (0.68, "#C96E79"),
            (1.0, "#A32136"),
        ],
    )
    norm = Normalize(vmin=corr_floor, vmax=vmax)
    size_mm = (184.0, 118.0)
    fig = create_figure_mm(*size_mm)
    fig_w_mm, fig_h_mm = size_mm

    def mm_rect(left_mm: float, bottom_mm: float, width_mm: float, height_mm: float) -> list[float]:
        return [
            left_mm / fig_w_mm,
            bottom_mm / fig_h_mm,
            width_mm / fig_w_mm,
            height_mm / fig_h_mm,
        ]

    heatmap_left_mm = 40.0
    heatmap_bottom_mm = 18.0
    heatmap_side_mm = 80.0
    strip_gap_mm = 1.4
    strip_thickness_mm = 4.0
    left_strip_gap_mm = 1.6
    left_strip_width_mm = 4.0
    colorbar_gap_mm = 4.0
    colorbar_width_mm = 7.0
    note_gap_mm = 7.0
    right_margin_mm = 6.0
    note_left_mm = heatmap_left_mm + heatmap_side_mm + colorbar_gap_mm + colorbar_width_mm + note_gap_mm
    note_width_mm = fig_w_mm - note_left_mm - right_margin_mm
    ari_height_mm = 18.0
    legend_gap_mm = 4.0

    ax = fig.add_axes(mm_rect(heatmap_left_mm, heatmap_bottom_mm, heatmap_side_mm, heatmap_side_mm))
    ax_top = fig.add_axes(
        mm_rect(
            heatmap_left_mm,
            heatmap_bottom_mm + heatmap_side_mm + strip_gap_mm,
            heatmap_side_mm,
            strip_thickness_mm,
        )
    )
    ax_left = fig.add_axes(
        mm_rect(
            heatmap_left_mm - left_strip_gap_mm - left_strip_width_mm,
            heatmap_bottom_mm,
            left_strip_width_mm,
            heatmap_side_mm,
        )
    )
    cax = fig.add_axes(
        mm_rect(
            heatmap_left_mm + heatmap_side_mm + colorbar_gap_mm,
            heatmap_bottom_mm,
            colorbar_width_mm,
            heatmap_side_mm,
        )
    )
    ari_ax = fig.add_axes(
        mm_rect(
            note_left_mm,
            heatmap_bottom_mm + heatmap_side_mm - ari_height_mm,
            note_width_mm,
            ari_height_mm,
        )
    )
    legend_ax = fig.add_axes(
        mm_rect(
            note_left_mm,
            heatmap_bottom_mm,
            note_width_mm,
            heatmap_side_mm - ari_height_mm - legend_gap_mm,
        )
    )

    im = ax.imshow(matrix.to_numpy(), cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title("")
    clusters = assignments["cluster_training_exact"].to_numpy()
    boundary_positions = [idx for idx in range(1, len(clusters)) if clusters[idx] != clusters[idx - 1]]
    ax.set_xticks(np.arange(-0.5, len(labels), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1.0), minor=True)
    x_tick_positions = compact_tick_positions([0, *boundary_positions], min_gap=5, last_index=len(labels) - 1)
    if len(x_tick_positions) >= 2 and x_tick_positions[-1] - x_tick_positions[-2] < 7:
        x_tick_positions.pop(-2)
    y_tick_positions = compact_tick_positions([0, *boundary_positions], min_gap=4, last_index=len(labels) - 1)
    ax.set_xticks(x_tick_positions)
    ax.set_yticks(y_tick_positions)
    ax.set_xticklabels([labels[idx] for idx in x_tick_positions], rotation=24, fontsize=4.8)
    ax.set_yticklabels([])
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
        tick.set_fontfamily("Arial")
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=1.8,
        width=0.5,
        top=True,
        right=True,
        pad=0.8,
    )
    ax.tick_params(axis="y", which="major", labelleft=False)
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=0.55,
        width=0.25,
        top=True,
        right=True,
    )
    style_axes(ax, show_top=True, show_right=True, grid=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#222222")

    for idx in range(1, len(clusters)):
        if clusters[idx] != clusters[idx - 1]:
            boundary = idx - 0.5
            ax.axhline(boundary, color="white", lw=1.0, alpha=0.9)
            ax.axvline(boundary, color="white", lw=1.0, alpha=0.9)

    training_labels = [label for label in TRAINING_LEGEND_ORDER if label in assignments["training_regime_label"].unique()]
    training_code_map = {label: idx for idx, label in enumerate(training_labels)}
    training_codes = assignments["training_regime_label"].map(training_code_map).to_numpy(dtype=int)
    training_cmap = ListedColormap([TRAINING_REGIME_COLORS[label] for label in training_labels])
    ax_top.imshow(
        training_codes[None, :],
        aspect="auto",
        interpolation="nearest",
        cmap=training_cmap,
        vmin=-0.5,
        vmax=len(training_labels) - 0.5,
    )
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for spine in ax_top.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#222222")

    architecture_keys = [key for key in ARCHITECTURE_LEGEND_ORDER if key in assignments["architecture_group"].unique()]
    architecture_code_map = {label: idx for idx, label in enumerate(architecture_keys)}
    architecture_codes = assignments["architecture_group"].map(architecture_code_map).to_numpy(dtype=int)
    architecture_cmap = ListedColormap([ARCHITECTURE_COLOR_MAP[key] for key in architecture_keys])
    ax_left.imshow(
        architecture_codes[:, None],
        aspect="auto",
        interpolation="nearest",
        cmap=architecture_cmap,
        vmin=-0.5,
        vmax=len(architecture_keys) - 0.5,
    )
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    for spine in ax_left.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#222222")

    for idx in range(1, len(clusters)):
        if clusters[idx] != clusters[idx - 1]:
            boundary = idx - 0.5
            ax_top.axvline(boundary, color="white", lw=1.0, alpha=0.9)
            ax_left.axhline(boundary, color="white", lw=1.0, alpha=0.9)

    draw_s01_ari_block(ari_ax, summary)

    colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
    colorbar.set_ticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.set_label("")
    colorbar.ax.set_title("Pairwise error\ncorrelation", fontsize=6.4, pad=4.0, fontfamily="Arial", color="#222222")
    colorbar.ax.tick_params(axis="y", which="major", direction="in", length=2.0, width=0.55, labelsize=7.0, pad=2.0)
    colorbar.outline.set_linewidth(0.8)
    colorbar.outline.set_edgecolor("#222222")
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontfamily("Arial")

    draw_s01_compact_legends(legend_ax, assignments)

    save_panel(fig, destinations, "panel_a_error_clustermap")
    compose_preview(
        package_dir,
        destinations,
        "figure_s01_preview",
        [("panel_a_error_clustermap.png", [0.00, 0.00, 1.0, 1.0])],
        size_mm,
    )
    write_notes(
        destinations,
        "Figure S1 Assembly Notes",
        [
            "Single-panel supplementary figure.",
            "White boundaries indicate changes in the exact training-data cluster assignment.",
            "Right-side annotation column separates the ARI summary, color scale, and compact legend grids.",
        ],
    )


def plot_control_distribution(ax, values_a: np.ndarray, values_b: np.ndarray, ylabel: str) -> None:
    style_axes(ax, grid=False)
    values = [values_a, values_b]
    positions = np.array([0.0, 1.0], dtype=float)
    violin_width = 0.82
    parts = ax.violinplot(
        values,
        positions=positions,
        widths=violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    colors = [FACTOR_COLORS["Architecture"], FACTOR_COLORS["Training data"]]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
        body.set_linewidth(1.2)
        body.set_zorder(1)

    rng = np.random.default_rng(7)
    ymax = max(float(np.max(values_a)), float(np.max(values_b)))
    ymin = min(float(np.min(values_a)), float(np.min(values_b)))
    span = max(ymax - ymin, 1e-6)
    summary_color = "#333333"
    for xpos, series, color, body in zip(positions, values, colors, parts["bodies"]):
        jitter = rng.uniform(-0.065, 0.065, size=len(series))
        points = ax.scatter(
            np.full(len(series), xpos) + jitter,
            series,
            s=28,
            color=color,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.55,
            zorder=3,
        )
        points.set_clip_path(body.get_paths()[0], body.get_transform())
        q25, median, q75 = np.quantile(series, [0.25, 0.5, 0.75])
        ax.vlines(xpos, q25, q75, color=summary_color, lw=0.95, zorder=4)
        ax.hlines(median, xpos - 0.135, xpos + 0.135, color=summary_color, lw=1.25, zorder=4)
        y_text = float(np.max(series) + 0.065 * span)
        ax.text(
            float(xpos),
            y_text,
            f"mean = {np.mean(series):.3f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#4A4A4A",
            fontfamily="Arial",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(["Arch change\nsame data", "Data change\nsame family"], fontsize=6.4, fontfamily="Arial")
    ax.set_ylabel(ylabel, fontfamily="Arial")
    ax.set_xlim(-0.48, 1.48)
    ax.set_ylim(ymin - 0.05 * span, ymax + 0.21 * span)


def make_s02_paired_controls_distribution() -> None:
    destinations, package_dir = output_dirs("figure_s02_paired_controls_distribution")
    arch_pairs = pd.read_csv(RESULTS_DIR / "tables" / "analysis_01_control" / "same_data_diff_architecture_pairs.csv")
    data_pairs = pd.read_csv(RESULTS_DIR / "tables" / "analysis_01_control" / "same_family_diff_data_pairs.csv")

    specs = [
        ("delta_f1_full_test", r"$\Delta$ F1", "panel_a_f1_distribution"),
        ("delta_mae_full_test", r"$\Delta$ MAE", "panel_b_mae_distribution"),
        ("delta_daf_full_test", r"$\Delta$ DAF", "panel_c_daf_distribution"),
    ]
    for idx, (column, ylabel, stem_name) in enumerate(specs):
        fig = create_figure_mm(58.0, 62.0)
        ax = fig.add_subplot(111)
        plot_control_distribution(ax, arch_pairs[column].to_numpy(), data_pairs[column].to_numpy(), ylabel)
        add_panel_label(ax, chr(ord("a") + idx), x=-0.16, y=1.02)
        fig.subplots_adjust(left=0.20, right=0.98, top=0.90, bottom=0.20)
        save_panel(fig, destinations, stem_name)

    compose_preview(
        package_dir,
        destinations,
        "figure_s02_preview",
        [
            ("panel_a_f1_distribution.png", [0.00, 0.05, 0.33, 0.90]),
            ("panel_b_mae_distribution.png", [0.335, 0.05, 0.33, 0.90]),
            ("panel_c_daf_distribution.png", [0.67, 0.05, 0.33, 0.90]),
        ],
        (180.0, 70.0),
    )
    write_notes(
        destinations,
        "Figure S2 Assembly Notes",
        [
            "Three-panel supplementary figure.",
            "Panel order: F1, MAE, DAF.",
            "Orange distributions quantify architecture changes at fixed data; green distributions quantify data changes within the same family.",
        ],
    )


def make_s03_feature_radar() -> None:
    destinations, package_dir = output_dirs("figure_s03_feature_radar")
    summary = pd.read_csv(RESULTS_DIR / "tables" / "analysis_04" / "failure_success_feature_summary.csv")
    pivot = summary.pivot(index="feature", columns="label", values=["mean", "std"])

    rows = []
    for feature in pivot.index:
        failure_mean = float(pivot.loc[feature, ("mean", "failure")])
        success_mean = float(pivot.loc[feature, ("mean", "success")])
        pooled_std = float((pivot.loc[feature, ("std", "failure")] + pivot.loc[feature, ("std", "success")]) / 2)
        score = abs(failure_mean - success_mean) / max(pooled_std, 1e-9)
        rows.append((feature, score, failure_mean, success_mean))
    rows.sort(key=lambda item: item[1], reverse=True)
    selected = rows[:6]

    selected_features = [feature for feature, *_ in selected]
    labels = [format_feature_name(feature) for feature in selected_features]
    failure_values = []
    success_values = []
    for _, _, failure_mean, success_mean in selected:
        scale = max(abs(failure_mean), abs(success_mean), 1e-9)
        failure_values.append(failure_mean / scale)
        success_values.append(success_mean / scale)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])
    failure_values = np.array(failure_values, dtype=float)
    success_values = np.array(success_values, dtype=float)
    closed_failure = np.concatenate([failure_values, [failure_values[0]]])
    closed_success = np.concatenate([success_values, [success_values[0]]])

    fig = create_figure_mm(100.0, 95.0)
    ax = fig.add_subplot(111, projection="polar")
    failure_color = "#E69F00"
    success_color = "#56B4E9"
    ax.plot(closed_angles, closed_failure, color=failure_color, lw=1.9, marker="o", ms=3.2)
    ax.plot(closed_angles, closed_success, color=success_color, lw=1.9, marker="o", ms=3.2)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", pad=0)
    label_radius = 1.23
    label_radius_overrides = {
        "Nearest-set distance": 1.16,
        "Space group number": 1.15,
    }
    for angle, label in zip(angles, labels):
        display_angle = (np.pi / 2) - angle
        x_coord = np.cos(display_angle)
        y_coord = np.sin(display_angle)
        if abs(x_coord) < 0.20:
            ha = "center"
        elif x_coord > 0:
            ha = "left"
        else:
            ha = "right"
        if y_coord > 0.55:
            va = "bottom"
        elif y_coord < -0.55:
            va = "top"
        else:
            va = "center"
        ax.text(
            angle,
            label_radius_overrides.get(label, label_radius),
            label,
            fontsize=7.2,
            fontfamily="Arial",
            color="#111111",
            ha=ha,
            va=va,
            clip_on=False,
        )
    ax.set_rlabel_position(24)
    ax.set_yticks([0.25, 0.50, 0.75])
    ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=6.0, color="#666666", fontfamily="Arial")
    for tick in ax.yaxis.get_ticklabels():
        tick.set_zorder(10)
    ax.set_ylim(0.0, 1.14)
    ax.grid(color="#D9D9D9", lw=0.6)
    ax.spines["polar"].set_color("#B0B0B0")
    ax.set_title("")
    ax.legend(
        [
            Line2D([0], [0], color=failure_color, lw=1.9, marker="o", ms=4),
            Line2D([0], [0], color=success_color, lw=1.9, marker="o", ms=4),
        ],
        ["Failure-enriched", "Success-enriched"],
        loc="upper right",
        bbox_to_anchor=(1.18, 1.10),
        fontsize=6.8,
    )

    highlight_features = [
        "min_l1_distance_same_element_set_filled",
        "n_sites",
    ]
    for feature in highlight_features:
        if feature not in selected_features:
            continue
        idx = selected_features.index(feature)
        point_radius = max(failure_values[idx], success_values[idx]) + 0.04
        ax.scatter([angles[idx]], [point_radius], marker="*", s=46, color="#555555", zorder=5, clip_on=False)

    fig.subplots_adjust(left=0.01, right=0.92, top=0.90, bottom=0.08)
    save_panel(fig, destinations, "panel_a_feature_radar")
    compose_preview(
        package_dir,
        destinations,
        "figure_s03_preview",
        [("panel_a_feature_radar.png", [0.00, 0.00, 1.0, 1.0])],
        (100.0, 95.0),
    )
    write_notes(
        destinations,
        "Figure S3 Assembly Notes",
        [
            "Single-panel supplementary figure.",
            "Features were selected by standardized failure-versus-success separation using the exported summary table.",
            "Radar values are normalized within feature so cross-feature directions remain readable.",
            "Grey stars indicate the features with the largest gap between the two groups.",
        ],
    )


def main() -> None:
    configure_publication_style()
    make_s01_error_clustermap()
    make_s02_paired_controls_distribution()
    make_s03_feature_radar()


if __name__ == "__main__":
    main()

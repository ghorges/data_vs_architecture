from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from dva_project.utils import ensure_dir


FULL_WIDTH_MM = 180.0
DOUBLE_COLUMN_MM = 180.0
SINGLE_COLUMN_MM = 89.0

FACTOR_COLORS = {
    "Training data": "#009E73",
    "Architecture": "#E69F00",
    "Parameters": "#999999",
}

TRAINING_REGIME_COLORS = {
    "OMat24+sAlex+MPtrj": "#0072B2",
    "MPtrj+Alex+OMat24": "#56B4E9",
    "MPtrj+sAlex": "#D55E00",
    "MPtrj only": "#009E73",
    "MatterSim": "#CC79A7",
    "OpenLAM": "#F0E442",
    "COSMOSDataset": "#E69F00",
    "MPF/MP2022": "#999999",
    "MPtrj__OMat24__sAlex": "#0072B2",
    "Alex__MPtrj__OMat24": "#56B4E9",
    "MPtrj__sAlex": "#D55E00",
    "MPtrj": "#009E73",
    "MatterSim": "#CC79A7",
    "OpenLAM": "#F0E442",
    "COSMOSDataset": "#E69F00",
    "MPF": "#999999",
    "MP 2022": "#999999",
}

ARCHITECTURE_MARKERS = {
    "equivariant_gnn": "o",
    "invariant_gnn": "s",
    "transformer": "^",
    "hybrid_ensemble": "D",
    "non_gnn": "X",
}

SNAPSHOT_STYLES = {
    "Frozen 45": {
        "label": "Frozen 45",
        "line_color": "#111111",
        "marker_color": "#111111",
    },
    "Live 53": {
        "label": "Live 53",
        "line_color": "#111111",
        "marker_color": "#111111",
    },
}


def mm_to_inches(*dimensions_mm: float) -> tuple[float, ...]:
    return tuple(value / 25.4 for value in dimensions_mm)


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.titlepad": 4.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "lines.linewidth": 1.5,
            "patch.linewidth": 0.8,
        }
    )


def create_figure_mm(width_mm: float, height_mm: float, **kwargs):
    figsize = mm_to_inches(width_mm, height_mm)
    return plt.figure(figsize=figsize, **kwargs)


def style_axes(
    ax,
    *,
    show_left: bool = True,
    show_bottom: bool = True,
    show_top: bool = False,
    show_right: bool = False,
    grid: bool = False,
    grid_axis: str = "y",
) -> None:
    ax.spines["left"].set_visible(show_left)
    ax.spines["bottom"].set_visible(show_bottom)
    ax.spines["top"].set_visible(show_top)
    ax.spines["right"].set_visible(show_right)
    if not show_left:
        ax.tick_params(axis="y", left=False)
    if not show_bottom:
        ax.tick_params(axis="x", bottom=False)
    ax.grid(False)
    if grid:
        ax.grid(
            True,
            axis=grid_axis,
            color="#D9D9D9",
            linewidth=0.5,
            alpha=0.9,
        )
    ax.set_facecolor("white")


def add_panel_label(ax, label: str, *, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111111",
    )


def add_underbar_marker(
    ax,
    x: float,
    y: float,
    *,
    width: float = 0.18,
    color: str = "#111111",
    linewidth: float = 1.4,
) -> None:
    ax.hlines(
        y,
        x - width / 2,
        x + width / 2,
        color=color,
        linewidth=linewidth,
        zorder=5,
    )


def save_figure(fig, stem: Path, *, dpi: int = 300, close: bool = True) -> None:
    ensure_dir(stem.parent)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)


def save_figure_to_many(fig, stems: list[Path], *, dpi: int = 300, close: bool = True) -> None:
    for stem in stems:
        save_figure(fig, stem, dpi=dpi, close=False)
    if close:
        plt.close(fig)

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
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogFormatterMathtext, MultipleLocator
import matplotlib.patheffects as pe

from dva_project.figure_style import (
    ARCHITECTURE_MARKERS,
    TRAINING_REGIME_COLORS,
    add_panel_label,
    configure_publication_style,
    create_figure_mm,
    save_figure_to_many,
    style_axes,
)
from dva_project.settings import PROJECT_ROOT, RESULTS_DIR
from dva_project.utils import ensure_dir


PANEL_A_SIZE_MM = (108.0, 88.0)
PANEL_B_SIZE_MM = (58.0, 88.0)
PREVIEW_SIZE_MM = (108.0, 132.0)
PANEL_A_XMAX = 10 ** 16.3
LEGEND_BOX_EDGE = "#C9C9C9"

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
ARCHITECTURE_LEGEND_ORDER = [
    ("Eq. GNN", ARCHITECTURE_MARKERS["equivariant_gnn"]),
    ("Inv. GNN", ARCHITECTURE_MARKERS["invariant_gnn"]),
    ("Transformer", ARCHITECTURE_MARKERS["transformer"]),
    ("Hybrid", ARCHITECTURE_MARKERS["hybrid_ensemble"]),
    ("Non-GNN", ARCHITECTURE_MARKERS["non_gnn"]),
]
SIMPLIFIED_GROUP_ORDER = ["MPtrj family", "OMat24-era", "Other regimes"]
SIMPLIFIED_GROUP_COLORS = {
    "MPtrj family": "#009E73",
    "OMat24-era": "#0072B2",
    "Other regimes": "#9A9A9A",
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
RECOMMENDED_LABELS = {
    "eqnorm-mptrj": "Eqnorm MPtrj",
    "mattersim-v1-5M": "MatterSim v1 5M",
    "eSEN-30m-oam": "eSEN-30M-OAM",
}
RECOMMENDED_ANNOTATIONS = {
    "eqnorm-mptrj": {
        "xytext": (8.4e11, 0.826),
        "textcoords": "data",
        "ha": "left",
        "va": "center",
    },
    "mattersim-v1-5M": {
        "xytext": (2.0e13, 0.905),
        "textcoords": "data",
        "ha": "left",
        "va": "center",
    },
    "eSEN-30m-oam": {
        "xytext": (6.5e14, 0.832),
        "textcoords": "data",
        "ha": "left",
        "va": "center",
    },
}
TIER_SHADE = {
    "low": "#D6E8D8",
    "mid": "#DEE8F8",
    "high": "#EFE3F8",
}
STRATEGY_FACE_COLORS = {
    "Parameter scaling (fixed data)": "#F8E1B5",
    "Data scaling (pooled families)": "#009E73",
    "Ensemble (new training cost)": "#B3B3B3",
}
STRATEGY_EDGE_COLORS = {
    "Parameter scaling (fixed data)": "#E69F00",
    "Data scaling (pooled families)": "#0B6E59",
    "Ensemble (new training cost)": "#666666",
}
OPTION1_MODEL_LABELS = {
    "esnet": "ESNet",
    "nequip-MP-L-0.1": "Nequip-MP",
    "nequip-OAM-L-0.1": "Nequip-OAM",
    "orb-v3": "ORB v3",
}
OPTION1_LABEL_OFFSETS = {
    "esnet": (8, 7),
    "nequip-MP-L-0.1": (10, -12),
    "nequip-OAM-L-0.1": (10, 14),
    "orb-v3": (14, -4),
}
OPTION2_FAMILY_LABELS = {
    "eSEN": "eSEN",
    "orb": "ORB",
    "sevennet": "SevenNet",
    "allegro": "Allegro",
    "nequip": "Nequip",
    "deepmd": "DPA",
    "grace": "GRACE",
}
OPTION2_FAMILY_ORDER = ["sevennet", "grace", "orb", "allegro", "nequip", "deepmd", "eSEN"]
OPTION2_START_COLOR = "#B3B3B3"
OPTION2_ARROW_COLOR = "#0B8B6B"
OPTION2_END_COLOR = "#009E73"
OPTION2_GRADIENT_START = "#BFD9D0"
OPTION2_GRADIENT_END = "#0B8B6B"


def load_panel_a() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(RESULTS_DIR / "tables" / "analysis_05" / "model_cost_summary.csv")
    frontier = pd.read_csv(RESULTS_DIR / "tables" / "analysis_05" / "pareto_frontier_proxy.csv")
    budget = pd.read_csv(RESULTS_DIR / "tables" / "analysis_05" / "budget_recommendations.csv")
    metadata["training_regime_label"] = metadata["training_combo"].map(TRAINING_LABEL_MAP).fillna("MPF/MP2022")
    frontier["training_regime_label"] = frontier["training_combo"].map(TRAINING_LABEL_MAP).fillna("MPF/MP2022")
    metadata["simplified_group"] = metadata["training_regime_label"].map(classify_training_group)
    return metadata, frontier, budget


def load_panel_b() -> pd.DataFrame:
    strategy = pd.read_csv(RESULTS_DIR / "tables" / "analysis_05" / "strategy_comparison.csv")
    return pd.DataFrame(
        [
            {
                "label": "Parameter scaling (fixed data)",
                "value": float(
                    strategy.loc[
                        strategy["strategy"] == "A_fixed_data_scale_params",
                        "f1_gain_per_log10_step",
                    ].iloc[0]
                ),
            },
            {
                "label": "Data scaling (pooled families)",
                "value": float(
                    strategy.loc[
                        strategy["strategy"] == "B_fixed_family_scale_data",
                        "f1_gain_per_log10_step",
                    ].iloc[0]
                ),
            },
            {
                "label": "Ensemble (new training cost)",
                "value": float(
                    strategy.loc[
                        (strategy["strategy"] == "C_prefix_ensemble") & (strategy["view"] == "train_budget"),
                        "f1_gain_per_log10_step",
                    ].iloc[0]
                ),
            },
        ]
    )


def load_panel_b_option1() -> pd.DataFrame:
    gpu_frontier = pd.read_csv(RESULTS_DIR / "tables" / "analysis_05" / "pareto_frontier_gpu_hours.csv").copy()
    gpu_frontier["training_regime_label"] = gpu_frontier["training_combo"].map(TRAINING_LABEL_MAP).fillna("MPF/MP2022")
    return gpu_frontier.sort_values("gpu_hours").reset_index(drop=True)


def load_panel_c_points(metadata: pd.DataFrame) -> pd.DataFrame:
    gpu_points = metadata.loc[metadata["gpu_hours"].notna()].copy()
    return gpu_points.sort_values(["gpu_hours", "f1_full_test"]).reset_index(drop=True)


def load_panel_b_option2() -> pd.DataFrame:
    points = pd.read_csv(RESULTS_DIR / "tables" / "analysis_03" / "data_scaling_points.csv")
    points = points.loc[(points["analysis"] == "data_scaling") & (points["family"].isin(OPTION2_FAMILY_ORDER))].copy()
    rows = []
    for family, group in points.groupby("family"):
        group = group.sort_values("effective_training_structures")
        start = group.iloc[0]
        end = group.iloc[-1]
        rows.append(
            {
                "family": family,
                "family_label": OPTION2_FAMILY_LABELS[family],
                "start_combo": start["training_combo"],
                "end_combo": end["training_combo"],
                "start_f1": float(start["f1_full_test"]),
                "end_f1": float(end["f1_full_test"]),
                "gain": float(end["f1_full_test"] - start["f1_full_test"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["sort_order"] = frame["family"].map({family: idx for idx, family in enumerate(OPTION2_FAMILY_ORDER)})
    return frame.sort_values(["gain", "sort_order"], ascending=[False, True]).reset_index(drop=True)


def classify_training_group(label: str) -> str:
    if label in {"MPtrj only", "MPtrj+sAlex"}:
        return "MPtrj family"
    if label in {"MPtrj+Alex+OMat24", "OMat24+sAlex+MPtrj"}:
        return "OMat24-era"
    return "Other regimes"


def compute_monotonic_frontier(metadata: pd.DataFrame) -> pd.DataFrame:
    frontier = (
        metadata[["proxy_cost", "f1_full_test"]]
        .groupby("proxy_cost", as_index=False)
        .max()
        .sort_values("proxy_cost")
        .reset_index(drop=True)
    )
    frontier["f1_frontier"] = frontier["f1_full_test"].cummax()
    frontier = frontier.loc[
        frontier["f1_frontier"] > frontier["f1_frontier"].shift(fill_value=-np.inf) + 1e-12,
        ["proxy_cost", "f1_frontier"],
    ].rename(columns={"f1_frontier": "f1_full_test"})
    return frontier


def build_frontier_path(frontier: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_vals = frontier["proxy_cost"].to_numpy(dtype=float)
    y_vals = frontier["f1_full_test"].to_numpy(dtype=float)
    if len(x_vals) == 0:
        return np.array([]), np.array([])
    path_x = [x_vals[0]]
    path_y = [y_vals[0]]
    for x_val, y_prev, y_val in zip(x_vals[1:], y_vals[:-1], y_vals[1:]):
        path_x.extend([x_val, x_val])
        path_y.extend([y_prev, y_val])
    return np.asarray(path_x), np.asarray(path_y)


def add_horizontal_gradient_line(
    ax,
    x_start: float,
    x_end: float,
    y_pos: float,
    *,
    start_color: str,
    end_color: str,
    linewidth: float = 1.8,
    n_segments: int = 96,
    zorder: int = 2,
) -> None:
    xs = np.linspace(x_start, x_end, n_segments)
    ys = np.full_like(xs, y_pos, dtype=float)
    segments = np.stack(
        [
            np.column_stack([xs[:-1], ys[:-1]]),
            np.column_stack([xs[1:], ys[1:]]),
        ],
        axis=1,
    )
    start_rgba = np.array(to_rgba(start_color))
    end_rgba = np.array(to_rgba(end_color))
    colors = [
        start_rgba + (end_rgba - start_rgba) * fraction
        for fraction in np.linspace(0.0, 1.0, n_segments - 1)
    ]
    collection = LineCollection(
        segments,
        colors=colors,
        linewidths=linewidth,
        capstyle="round",
        zorder=zorder,
    )
    ax.add_collection(collection)


def build_training_handles(metadata: pd.DataFrame) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TRAINING_REGIME_COLORS[label],
            markeredgecolor="none",
            markeredgewidth=0.0,
            markersize=5.9,
            label=label,
        )
        for label in TRAINING_LEGEND_ORDER
        if label in metadata["training_regime_label"].unique()
    ]


def draw_structured_legend_box(ax, metadata: pd.DataFrame) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            facecolor="white",
            edgecolor=LEGEND_BOX_EDGE,
            linewidth=0.7,
        )
    )
    ax.plot([0.60, 0.60], [0.08, 0.84], color="#DDDDDD", linewidth=0.55, zorder=1)
    ax.plot([0.03, 0.97], [0.71, 0.71], color="#DDDDDD", linewidth=0.55, zorder=1)

    ax.text(0.03, 0.81, "Training data", fontsize=5.0, fontweight="bold", ha="left", va="center", color="#222222")
    ax.text(0.63, 0.81, "Architecture", fontsize=5.0, fontweight="bold", ha="left", va="center", color="#222222")

    training_labels = [label for label in TRAINING_LEGEND_ORDER if label in metadata["training_regime_label"].unique()]
    training_positions = [
        (0.05, 0.60),
        (0.34, 0.60),
        (0.05, 0.44),
        (0.34, 0.44),
        (0.05, 0.28),
        (0.34, 0.28),
        (0.05, 0.12),
        (0.34, 0.12),
    ]
    for (x_pos, y_pos), label in zip(training_positions, training_labels):
        ax.plot(
            [x_pos],
            [y_pos],
            marker="o",
            markersize=6.7,
            markerfacecolor=TRAINING_REGIME_COLORS[label],
            markeredgecolor="#222222",
            markeredgewidth=0.6,
            linestyle="none",
            zorder=3,
        )
        ax.text(x_pos + 0.034, y_pos, label, fontsize=4.2, ha="left", va="center", color="#222222")

    architecture_positions = [
        (0.65, 0.60),
        (0.83, 0.60),
        (0.65, 0.41),
        (0.83, 0.41),
        (0.65, 0.16),
    ]
    for (x_pos, y_pos), (label, marker) in zip(architecture_positions, ARCHITECTURE_LEGEND_ORDER):
        ax.plot(
            [x_pos],
            [y_pos],
            marker=marker,
            markersize=6.2,
            markerfacecolor="#222222",
            markeredgecolor="#222222",
            markeredgewidth=0.6,
            linestyle="none",
            zorder=3,
        )
        ax.text(x_pos + 0.022, y_pos, label, fontsize=4.3, ha="left", va="center", color="#222222")


def build_architecture_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="#8C8C8C",
            linestyle="none",
            markerfacecolor="#8C8C8C",
            markeredgecolor="#8C8C8C",
            markersize=5.9,
            label=label,
        )
        for label, marker in ARCHITECTURE_LEGEND_ORDER
    ]


def add_between_row_xlabel(fig, ax, legend_ax, text: str, *, fontsize: float = 8.3) -> None:
    plot_pos = ax.get_position()
    legend_pos = legend_ax.get_position()
    y_pos = legend_pos.y1 + 0.03 * (plot_pos.y0 - legend_pos.y1)
    fig.text(
        (plot_pos.x0 + plot_pos.x1) / 2.0,
        y_pos,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="#111111",
    )


def draw_xlabel_band(ax, text: str, *, fontsize: float = 7.5, y: float = 0.58) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(
        0.5,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111111",
    )


def draw_panel_a(
    ax,
    metadata: pd.DataFrame,
    frontier: pd.DataFrame,
    budget: pd.DataFrame,
    *,
    show_legend: bool = True,
) -> None:
    style_axes(ax, show_top=False, show_right=False, grid=False)
    ax.set_title("")
    ax.axvspan(1e10, 1e12, ymin=0.0, ymax=0.955, color=TIER_SHADE["low"], alpha=0.82, zorder=0)
    ax.axvspan(1e12, 1e14, ymin=0.0, ymax=0.955, color=TIER_SHADE["mid"], alpha=0.82, zorder=0)
    ax.axvspan(1e14, PANEL_A_XMAX, ymin=0.0, ymax=0.955, color=TIER_SHADE["high"], alpha=0.82, zorder=0)
    tier_y = 0.983
    for x_pos, label, color in [
        (np.sqrt(1e10 * 1e12), "Low", "#4E6A55"),
        (np.sqrt(1e12 * 1e14), "Mid", "#516A8A"),
        (np.sqrt(1e14 * PANEL_A_XMAX), "High", "#6A547A"),
    ]:
        ax.text(
            x_pos,
            tier_y,
            label,
            transform=ax.get_xaxis_transform(),
            fontsize=6.6,
            fontweight="semibold",
            color=color,
            ha="center",
            va="center",
        )

    for architecture_group, group in metadata.groupby("architecture_group"):
        marker = ARCHITECTURE_MARKERS[architecture_group]
        ax.scatter(
            group["proxy_cost"],
            group["f1_full_test"],
            s=42,
            marker=marker,
            c=group["training_regime_label"].map(TRAINING_REGIME_COLORS),
            edgecolors="#222222",
            linewidths=0.4,
            alpha=0.95,
            zorder=3,
        )

    frontier = compute_monotonic_frontier(metadata)
    frontier_x, frontier_y = build_frontier_path(frontier)
    ax.plot(
        frontier_x,
        frontier_y,
        color="#444444",
        linewidth=0.9,
        zorder=2,
    )

    recommended_keys = budget["best_model"].tolist()
    for model_key in recommended_keys:
        row = metadata.loc[metadata["model_key"] == model_key].iloc[0]
        ax.scatter(
            [row["proxy_cost"]],
            [row["f1_full_test"]],
            s=205,
            facecolors="none",
            edgecolors=TRAINING_REGIME_COLORS[row["training_regime_label"]],
            linewidths=1.45,
            zorder=4,
        )
        annotation_style = RECOMMENDED_ANNOTATIONS[model_key]
        arrowprops = None if model_key == "eqnorm-mptrj" else {
            "arrowstyle": "-",
            "color": "#777777",
            "lw": 0.5,
            "shrinkA": 2,
            "shrinkB": 5,
            "connectionstyle": "angle,angleA=0,angleB=90,rad=0",
        }
        annotation = ax.annotate(
            RECOMMENDED_LABELS[model_key],
            xy=(row["proxy_cost"], row["f1_full_test"]),
            xytext=annotation_style["xytext"],
            textcoords=annotation_style["textcoords"],
            fontsize=7.1,
            fontweight="bold",
            color="#222222",
            ha=annotation_style["ha"],
            va=annotation_style["va"],
            arrowprops=arrowprops,
            annotation_clip=False,
        )
        annotation.set_path_effects([pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xlim(1e10, PANEL_A_XMAX)
    ax.set_ylim(0.40, 0.95)
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.set_xlabel("Proxy training cost", fontsize=8.0)
    ax.xaxis.set_label_coords(0.5, -0.09)
    ax.set_ylabel("F1", fontsize=8.0, labelpad=4.0)
    ax.tick_params(axis="both", which="major", direction="out", top=False, right=False, labelsize=7.0, length=3.2, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="out", top=False, right=False, length=2.0, width=0.8)
    ax.tick_params(axis="x", pad=2.0)
    if show_legend:
        arch_legend = ax.legend(
            handles=build_architecture_handles(),
            title="Architecture",
            loc="lower right",
            bbox_to_anchor=(0.965, 0.035),
            fontsize=5.0,
            title_fontsize=5.0,
            handletextpad=0.94,
            labelspacing=0.62,
            borderaxespad=0.0,
            frameon=False,
        )
        arch_legend._legend_box.align = "left"
        ax.add_artist(arch_legend)
        training_legend = ax.legend(
            handles=build_training_handles(metadata),
            title="Training data",
            loc="upper left",
            bbox_to_anchor=(0.025, 0.955),
            fontsize=5.0,
            title_fontsize=5.0,
            handletextpad=0.94,
            labelspacing=0.50,
            borderaxespad=0.0,
            frameon=False,
        )
        training_legend._legend_box.align = "left"
    add_panel_label(ax, "a")


def draw_panel_b(
    ax,
    strategy: pd.DataFrame,
    *,
    compact: bool = False,
    panel_label: str = "b",
    panel_label_x: float = 0.0,
    panel_label_y: float = 1.04,
) -> None:
    style_axes(ax, show_right=False, show_top=False, grid=False)
    ax.set_title("")
    labels = [
        "Parameter scaling\n(fixed data)",
        "Data scaling\n(pooled families)",
        "Ensemble\n(new training cost)",
    ]
    values = strategy["value"].to_numpy(dtype=float)
    y_positions = np.array([1.48, 0.84, 0.20], dtype=float)
    bar_height = 0.36

    ax.barh(
        y_positions,
        values,
        color=[
            to_rgba("#E69F00", 0.12) if label == "Parameter scaling (fixed data)" else STRATEGY_FACE_COLORS[label]
            for label in strategy["label"].tolist()
        ],
        edgecolor=[STRATEGY_EDGE_COLORS[label] for label in strategy["label"].tolist()],
        linewidth=[1.1 if label == "Parameter scaling (fixed data)" else 0.7 for label in strategy["label"].tolist()],
        height=bar_height,
        zorder=3,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=5.8)
    for tick in ax.get_yticklabels():
        tick.set_linespacing(0.82)
        tick.set_multialignment("center")
    ax.set_xlabel(r"$\Delta$F1 per log$_{10}$ step", fontsize=8.0)
    ax.xaxis.set_label_coords(0.5, -0.09)
    ax.set_xlim(0.0, 0.075)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.tick_params(axis="x", which="major", direction="out", labelsize=6.3, length=3.2, width=0.8, pad=2.0)
    ax.tick_params(axis="x", which="minor", direction="out", length=2.0, width=0.8)
    ax.tick_params(axis="y", which="major", direction="out", length=3.2, width=0.8)
    ax.tick_params(axis="y", pad=4.0)
    ax.set_ylim(-0.08, 1.80)

    for y_pos, row in zip(y_positions, strategy.itertuples(index=False)):
        text_y = y_pos
        if row.label == "Data scaling (pooled families)":
            text_y = y_pos + 0.01
        ax.text(
            row.value + 0.0012,
            text_y,
            f"{row.value:.3f}",
            fontsize=6.5,
            va="center",
            ha="left",
            color="#333333",
            clip_on=False,
        )

    data_row = strategy.loc[strategy["label"] == "Data scaling (pooled families)"].iloc[0]
    if compact:
        best_marker_x = data_row["value"] + 0.0019
        best_marker_y = y_positions[1] + 0.11
        best_text = "Preferred"
        best_text_fontsize = 4.9
        best_text_offset = 0.0024
    else:
        best_marker_x = data_row["value"] + 0.0034
        best_marker_y = y_positions[1] + 0.15
        best_text = "Preferred for new runs"
        best_text_fontsize = 5.9
        best_text_offset = 0.0016
    ax.plot(
        [data_row["value"], best_marker_x - 0.0006],
        [y_positions[1] + 0.04, best_marker_y],
        color="#6A6A6A",
        linewidth=0.6,
        zorder=4,
        clip_on=False,
    )
    ax.scatter([best_marker_x], [best_marker_y], marker="*", s=44, color="#007A5A", zorder=5, clip_on=False)
    ax.text(
        best_marker_x + best_text_offset,
        best_marker_y,
        best_text,
        fontsize=best_text_fontsize,
        linespacing=0.9,
        color="#2F2F2F",
        ha="left",
        va="center",
        clip_on=not compact,
    )
    ensemble_row = strategy.iloc[2]
    ax.text(
        ensemble_row["value"] + 0.016,
        y_positions[2] - 0.010,
        "Peak F1 = 0.911 at k = 6\nsaturates thereafter",
        fontsize=5.8,
        linespacing=0.92,
        color="#4B4B4B",
        ha="left",
        va="center",
        clip_on=False,
    )
    parameter_row = strategy.loc[strategy["label"] == "Parameter scaling (fixed data)"].iloc[0]
    delta_y = (y_positions[0] - bar_height / 2.0 + y_positions[1] + bar_height / 2.0) / 2.0
    ax.text(
        (parameter_row["value"] + data_row["value"]) / 2.0,
        delta_y + 0.015,
        r"$\Delta$ = +0.005",
        fontsize=8.0,
        color="#111111",
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
    )
    add_panel_label(ax, panel_label, x=panel_label_x, y=panel_label_y)


def draw_panel_b_option1(
    ax,
    gpu_points: pd.DataFrame,
    gpu_frontier: pd.DataFrame,
    *,
    panel_label: str = "b",
    panel_label_x: float = 0.0,
    panel_label_y: float = 1.04,
) -> None:
    style_axes(ax, show_right=False, show_top=False, grid=False)
    ax.set_title("")
    frame = gpu_frontier.sort_values("gpu_hours").copy()
    background = gpu_points.sort_values("gpu_hours").copy()
    for row in background.itertuples(index=False):
        marker = ARCHITECTURE_MARKERS[row.architecture_group]
        ax.scatter(
            row.gpu_hours,
            row.f1_full_test,
            s=34,
            marker=marker,
            facecolors="#D0D0D0",
            edgecolors="#8E8E8E",
            linewidths=0.35,
            alpha=0.95,
            zorder=2,
        )
    ax.plot(
        frame["gpu_hours"],
        frame["f1_full_test"],
        color="#555555",
        linewidth=1.0,
        zorder=3,
    )
    for row in frame.itertuples(index=False):
        marker = ARCHITECTURE_MARKERS[row.architecture_group]
        color = TRAINING_REGIME_COLORS[row.training_regime_label]
        ax.scatter(
            row.gpu_hours,
            row.f1_full_test,
            s=56,
            marker=marker,
            color=color,
            edgecolors="#222222",
            linewidths=0.45,
            zorder=4,
        )
        x_off, y_off = OPTION1_LABEL_OFFSETS[row.model_key]
        ax.annotate(
            OPTION1_MODEL_LABELS[row.model_key],
            xy=(row.gpu_hours, row.f1_full_test),
            xytext=(x_off, y_off),
            textcoords="offset points",
            fontsize=6.1,
            color="#222222",
            ha="left",
            va="center",
            zorder=5,
        )
    ax.set_xlabel("GPU-hours", fontsize=8.0)
    ax.xaxis.set_label_coords(0.5, -0.125)
    ax.set_ylabel("F1", fontsize=8.0, labelpad=4.0)
    ax.set_xlim(0, 2300)
    ax.set_xticks([0, 1000, 2000])
    ax.set_ylim(0.54, 0.91)
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.tick_params(axis="both", which="major", direction="out", labelsize=6.4, length=3.0, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="out", length=1.8, width=0.8)
    ax.tick_params(axis="x", pad=2.0)
    mini_legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#D0D0D0",
            markeredgecolor="#8E8E8E",
            markeredgewidth=0.5,
            markersize=5.0,
            label="Other models",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="#555555",
            linewidth=1.0,
            marker="o",
            markerfacecolor="#1f77b4",
            markeredgecolor="#222222",
            markeredgewidth=0.4,
            markersize=4.8,
            label="Real-cost frontier",
        ),
    ]
    mini_legend = ax.legend(
        handles=mini_legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.045),
        fontsize=5.1,
        handlelength=1.7,
        handletextpad=0.6,
        labelspacing=0.45,
        borderaxespad=0.0,
        frameon=False,
    )
    mini_legend._legend_box.align = "left"
    add_panel_label(ax, panel_label, x=panel_label_x, y=panel_label_y)


def draw_panel_b_option2(
    ax,
    migration: pd.DataFrame,
    *,
    panel_label: str = "b",
    panel_label_x: float = 0.0,
    panel_label_y: float = 1.04,
) -> None:
    style_axes(ax, show_right=False, show_top=False, grid=False)
    ax.set_title("")
    frame = migration.sort_values("gain", ascending=False).reset_index(drop=True)
    y_positions = np.arange(len(frame))[::-1]

    for y_pos, row in zip(y_positions, frame.itertuples(index=False)):
        add_horizontal_gradient_line(
            ax,
            row.start_f1,
            row.end_f1,
            y_pos,
            start_color=OPTION2_GRADIENT_START,
            end_color=OPTION2_GRADIENT_END,
            linewidth=1.9,
            zorder=2,
        )
        ax.scatter(
            row.start_f1,
            y_pos,
            s=28,
            color=OPTION2_START_COLOR,
            edgecolors="none",
            zorder=3,
        )
        ax.scatter(
            row.end_f1,
            y_pos,
            s=32,
            color=OPTION2_END_COLOR,
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
        )
        ax.scatter(
            row.end_f1 - 0.004,
            y_pos,
            s=20,
            marker=">",
            color=OPTION2_GRADIENT_END,
            edgecolors="none",
            zorder=4,
        )
        ax.text(
            row.end_f1 + 0.010,
            y_pos,
            f"+{row.gain:.3f}",
            fontsize=5.8,
            ha="left",
            va="center",
            color="#2F2F2F",
        )

    guide_y = 1.05
    ax.text(
        0.31,
        guide_y,
        "MPtrj-only",
        transform=ax.transAxes,
        fontsize=5.8,
        ha="right",
        va="center",
        color="#222222",
    )
    ax.annotate(
        "",
        xy=(0.49, guide_y),
        xytext=(0.34, guide_y),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 0.9,
            "color": "#222222",
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 8,
        },
        annotation_clip=False,
    )
    ax.text(
        0.52,
        guide_y,
        "larger-data regime",
        transform=ax.transAxes,
        fontsize=5.8,
        ha="left",
        va="center",
        color="#222222",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(frame["family_label"], fontsize=6.0)
    ax.set_xlabel("F1", fontsize=8.0)
    ax.xaxis.set_label_coords(0.5, -0.125)
    ax.set_xlim(0.66, 0.915)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.tick_params(axis="both", which="major", direction="out", labelsize=6.4, length=3.0, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="out", length=1.8, width=0.8)
    ax.tick_params(axis="x", pad=2.0)
    ax.tick_params(axis="y", pad=4.0)
    add_panel_label(ax, panel_label, x=panel_label_x, y=panel_label_y)


def create_panel_a(output_stems: list[Path], metadata: pd.DataFrame, frontier: pd.DataFrame, budget: pd.DataFrame) -> None:
    fig = create_figure_mm(*PANEL_A_SIZE_MM)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.12], hspace=0.06)
    ax = fig.add_subplot(grid[0, 0])
    xlabel_ax = fig.add_subplot(grid[1, 0])
    draw_panel_a(ax, metadata, frontier, budget, show_legend=True)
    draw_xlabel_band(xlabel_ax, "Proxy training cost", fontsize=7.5, y=0.02)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.10)
    ax.set_zorder(2)
    xlabel_ax.set_zorder(2)
    plot_pos = ax.get_position()
    xlabel_pos = xlabel_ax.get_position()
    xlabel_ax.set_position([plot_pos.x0, xlabel_pos.y0, plot_pos.x1 - plot_pos.x0, xlabel_pos.height])
    ax.set_xlabel("")
    save_figure_to_many(fig, output_stems)


def create_panel_b(output_stems: list[Path], strategy: pd.DataFrame) -> None:
    fig = create_figure_mm(*PANEL_B_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_b(ax, strategy)
    fig.subplots_adjust(left=0.40, right=0.98, top=0.93, bottom=0.19)
    save_figure_to_many(fig, output_stems)


def create_panel_b_option1(output_stems: list[Path], gpu_points: pd.DataFrame, gpu_frontier: pd.DataFrame) -> None:
    fig = create_figure_mm(*PANEL_B_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_b_option1(ax, gpu_points, gpu_frontier)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.93, bottom=0.19)
    save_figure_to_many(fig, output_stems)


def create_panel_b_option2(output_stems: list[Path], migration: pd.DataFrame) -> None:
    fig = create_figure_mm(*PANEL_B_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_b_option2(ax, migration)
    fig.subplots_adjust(left=0.28, right=0.98, top=0.93, bottom=0.19)
    save_figure_to_many(fig, output_stems)


def create_preview(
    output_stems: list[Path],
    metadata: pd.DataFrame,
    frontier: pd.DataFrame,
    budget: pd.DataFrame,
    strategy: pd.DataFrame,
    migration: pd.DataFrame,
    gpu_points: pd.DataFrame,
    gpu_frontier: pd.DataFrame,
) -> None:
    fig = create_figure_mm(*PREVIEW_SIZE_MM)
    fig.patch.set_facecolor("white")
    fig2_a_aspect = 106.68 / 182.88
    fig2_bc_aspect = 76.20 / 86.36
    pos_a_width = 0.76
    pos_a_height = pos_a_width * fig2_a_aspect
    pos_a = [0.12, 0.92 - pos_a_height, pos_a_width, pos_a_height]
    pos_bc_width = 0.30
    pos_bc_height = pos_bc_width * fig2_bc_aspect
    bottom_y = 0.09
    pos_b = [0.12, bottom_y, pos_bc_width, pos_bc_height]
    pos_c = [0.58, bottom_y, pos_bc_width, pos_bc_height]
    ax_a = fig.add_axes(pos_a)
    ax_b = fig.add_axes(pos_b)
    ax_c = fig.add_axes(pos_c)
    draw_panel_a(ax_a, metadata, frontier, budget, show_legend=True)
    ax_a.xaxis.set_label_coords(0.5, -0.09)
    draw_panel_b_option2(ax_b, migration, panel_label="", panel_label_x=-0.10, panel_label_y=1.04)
    draw_panel_b_option1(ax_c, gpu_points, gpu_frontier, panel_label="", panel_label_x=-0.10, panel_label_y=1.04)
    label_x_offset = 0.10 * pos_a[2]
    label_y_offset = 0.005
    fig.text(
        pos_b[0] - label_x_offset,
        pos_b[1] + pos_b[3] + label_y_offset,
        "b",
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111111",
    )
    fig.text(
        pos_c[0] - label_x_offset,
        pos_c[1] + pos_c[3] + label_y_offset,
        "c",
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111111",
    )
    save_figure_to_many(fig, output_stems)


def create_preview_option1(
    output_stems: list[Path],
    metadata: pd.DataFrame,
    frontier: pd.DataFrame,
    budget: pd.DataFrame,
    gpu_points: pd.DataFrame,
    gpu_frontier: pd.DataFrame,
) -> None:
    fig = create_figure_mm(*PREVIEW_SIZE_MM)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.12],
        width_ratios=[PANEL_A_SIZE_MM[0], PANEL_B_SIZE_MM[0]],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    xlabel_ax_a = fig.add_subplot(grid[1, 0])
    xlabel_ax_b = fig.add_subplot(grid[1, 1])
    draw_panel_a(ax_a, metadata, frontier, budget, show_legend=True)
    draw_panel_b_option1(ax_b, gpu_points, gpu_frontier)
    draw_xlabel_band(xlabel_ax_a, "Proxy training cost", fontsize=7.5, y=0.02)
    draw_xlabel_band(xlabel_ax_b, "GPU-hours", fontsize=7.5, y=0.02)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.12, wspace=0.38, hspace=0.06)
    ax_a.set_zorder(2)
    ax_b.set_zorder(2)
    xlabel_ax_a.set_zorder(2)
    xlabel_ax_b.set_zorder(2)
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    xlabel_pos_a = xlabel_ax_a.get_position()
    xlabel_pos_b = xlabel_ax_b.get_position()
    xlabel_ax_a.set_position([pos_a.x0, xlabel_pos_a.y0, pos_a.x1 - pos_a.x0, xlabel_pos_a.height])
    xlabel_ax_b.set_position([pos_b.x0, xlabel_pos_b.y0, pos_b.x1 - pos_b.x0, xlabel_pos_b.height])
    ax_a.set_xlabel("")
    ax_b.set_xlabel("")
    save_figure_to_many(fig, output_stems)


def create_preview_option2(
    output_stems: list[Path],
    metadata: pd.DataFrame,
    frontier: pd.DataFrame,
    budget: pd.DataFrame,
    migration: pd.DataFrame,
) -> None:
    fig = create_figure_mm(*PREVIEW_SIZE_MM)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.12],
        width_ratios=[PANEL_A_SIZE_MM[0], PANEL_B_SIZE_MM[0]],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    xlabel_ax_a = fig.add_subplot(grid[1, 0])
    xlabel_ax_b = fig.add_subplot(grid[1, 1])
    draw_panel_a(ax_a, metadata, frontier, budget, show_legend=True)
    draw_panel_b_option2(ax_b, migration)
    draw_xlabel_band(xlabel_ax_a, "Proxy training cost", fontsize=7.5, y=0.02)
    draw_xlabel_band(xlabel_ax_b, "F1", fontsize=7.5, y=0.02)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.12, wspace=0.38, hspace=0.06)
    ax_a.set_zorder(2)
    ax_b.set_zorder(2)
    xlabel_ax_a.set_zorder(2)
    xlabel_ax_b.set_zorder(2)
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    xlabel_pos_a = xlabel_ax_a.get_position()
    xlabel_pos_b = xlabel_ax_b.get_position()
    xlabel_ax_a.set_position([pos_a.x0, xlabel_pos_a.y0, pos_a.x1 - pos_a.x0, xlabel_pos_a.height])
    xlabel_ax_b.set_position([pos_b.x0, xlabel_pos_b.y0, pos_b.x1 - pos_b.x0, xlabel_pos_b.height])
    ax_a.set_xlabel("")
    ax_b.set_xlabel("")
    save_figure_to_many(fig, output_stems)


def write_assembly_notes(output_dirs: list[Path]) -> None:
    contents = "\n".join(
        [
            "# Figure 5 Assembly Notes",
            "",
            "Panels exported as individual PDF and PNG assets.",
            "",
            "Suggested layout:",
            "- left ~60% width: `panel_a_pareto_frontier.pdf`",
            "- right ~35% width: `panel_b_strategy_comparison.pdf`",
            "",
            "Preview file:",
            "- `figure_05_preview.png` / `figure_05_preview.pdf`",
            "",
            "Data sources:",
            "- `results/tables/analysis_05/model_cost_summary.csv`",
            "- `results/tables/analysis_05/pareto_frontier_proxy.csv`",
            "- `results/tables/analysis_05/budget_recommendations.csv`",
            "- `results/tables/analysis_05/strategy_comparison.csv`",
            "",
        ]
    )
    for output_dir in output_dirs:
        ensure_dir(output_dir)
        (output_dir / "assembly_notes.md").write_text(contents, encoding="utf-8")


def main() -> None:
    configure_publication_style()

    results_dir = RESULTS_DIR / "figures" / "figure_05_pareto_strategy"
    package_dir = (
        PROJECT_ROOT
        / "paper"
        / "submission_package"
        / "figures"
        / "main"
        / "figure_05_pareto_strategy"
    )
    output_dirs = [results_dir, package_dir]
    for output_dir in output_dirs:
        ensure_dir(output_dir)

    metadata, frontier, budget = load_panel_a()
    strategy = load_panel_b()
    gpu_frontier = load_panel_b_option1()
    gpu_points = load_panel_c_points(metadata)
    migration = load_panel_b_option2()

    create_panel_a([output_dir / "panel_a_pareto_frontier" for output_dir in output_dirs], metadata, frontier, budget)
    create_panel_b([output_dir / "panel_b_strategy_comparison" for output_dir in output_dirs], strategy)
    create_panel_b_option1(
        [output_dir / "panel_b_option1_real_gpu_hours" for output_dir in output_dirs],
        gpu_points,
        gpu_frontier,
    )
    create_panel_b_option2(
        [output_dir / "panel_b_option2_migration_gain" for output_dir in output_dirs],
        migration,
    )
    create_preview(
        [output_dir / "figure_05_preview" for output_dir in output_dirs],
        metadata,
        frontier,
        budget,
        strategy,
        migration,
        gpu_points,
        gpu_frontier,
    )
    create_preview_option1(
        [output_dir / "figure_05_preview_option1_real_gpu_hours" for output_dir in output_dirs],
        metadata,
        frontier,
        budget,
        gpu_points,
        gpu_frontier,
    )
    create_preview_option2(
        [output_dir / "figure_05_preview_option2_migration_gain" for output_dir in output_dirs],
        metadata,
        frontier,
        budget,
        migration,
    )
    write_assembly_notes(output_dirs)


if __name__ == "__main__":
    main()

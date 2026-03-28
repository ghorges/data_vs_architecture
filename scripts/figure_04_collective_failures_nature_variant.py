from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

import figure_04_collective_failures as base
from dva_project.figure_style import (
    add_panel_label,
    configure_publication_style,
    create_figure_mm,
    save_figure_to_many,
    style_axes,
)
from dva_project.settings import PROJECT_ROOT, RESULTS_DIR
from dva_project.utils import ensure_dir


PANEL_A_FIGSIZE_IN = (18.0, 5.5)
PANEL_B_SIZE_MM = (100.0, 60.0)
PANEL_C_SIZE_MM = (90.0, 60.0)
PREVIEW_SIZE_MM = (180.0, 150.0)

FAILURE_CMAP = plt.cm.OrRd
FAILURE_BAR = "#C97D00"
FAILURE_HOTSPOT = "#A96800"
FN_BAR = "#F3D8AC"
ANNOTATION_DARK = "#7A5A24"
NEUTRAL_TEXT = "#333333"
LIGHT_TEXT = "#666666"
CARD_EDGE = "#555555"
CONNECTOR_GRAY = "#7A7A7A"
BASE_FAILURE = "#AFAFAF"
BASE_FN = "#D9D9D9"
PANEL_A_PRESENT_BASE = "#118B86"
PANEL_A_TEXT = "#111111"
PANEL_A_ABSENT_FILL = "#FBF7EF"
PANEL_A_SECONDARY_TEXT = "#9A9A9A"
PANEL_A_Z_ORDER = {
    symbol: index + 1
    for index, symbol in enumerate(
        [
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
            "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr",
            "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "In", "Sn", "Sb", "Te", "I", "Xe",
            "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
            "Ho", "Er", "Tm", "Yb", "Lu",
            "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Tl", "Pb", "Bi", "Po", "At", "Rn",
            "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
            "Es", "Fm", "Md", "No", "Lr",
            "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
        ]
    )
}
PANEL_A_SYMBOLS = [
    ("H", 1, 1), ("He", 18, 1),
    ("Li", 1, 2), ("Be", 2, 2), ("B", 13, 2), ("C", 14, 2), ("N", 15, 2), ("O", 16, 2), ("F", 17, 2), ("Ne", 18, 2),
    ("Na", 1, 3), ("Mg", 2, 3), ("Al", 13, 3), ("Si", 14, 3), ("P", 15, 3), ("S", 16, 3), ("Cl", 17, 3), ("Ar", 18, 3),
    ("K", 1, 4), ("Ca", 2, 4), ("Sc", 3, 4), ("Ti", 4, 4), ("V", 5, 4), ("Cr", 6, 4), ("Mn", 7, 4), ("Fe", 8, 4),
    ("Co", 9, 4), ("Ni", 10, 4), ("Cu", 11, 4), ("Zn", 12, 4), ("Ga", 13, 4), ("Ge", 14, 4), ("As", 15, 4), ("Se", 16, 4),
    ("Br", 17, 4), ("Kr", 18, 4),
    ("Rb", 1, 5), ("Sr", 2, 5), ("Y", 3, 5), ("Zr", 4, 5), ("Nb", 5, 5), ("Mo", 6, 5), ("Tc", 7, 5), ("Ru", 8, 5),
    ("Rh", 9, 5), ("Pd", 10, 5), ("Ag", 11, 5), ("Cd", 12, 5), ("In", 13, 5), ("Sn", 14, 5), ("Sb", 15, 5), ("Te", 16, 5),
    ("I", 17, 5), ("Xe", 18, 5),
    ("Cs", 1, 6), ("Ba", 2, 6), ("Hf", 4, 6), ("Ta", 5, 6), ("W", 6, 6), ("Re", 7, 6), ("Os", 8, 6), ("Ir", 9, 6),
    ("Pt", 10, 6), ("Au", 11, 6), ("Hg", 12, 6), ("Tl", 13, 6), ("Pb", 14, 6), ("Bi", 15, 6), ("Po", 16, 6), ("At", 17, 6),
    ("Rn", 18, 6),
    ("Fr", 1, 7), ("Ra", 2, 7), ("Rf", 4, 7), ("Db", 5, 7), ("Sg", 6, 7), ("Bh", 7, 7), ("Hs", 8, 7),
    ("Mt", 9, 7), ("Ds", 10, 7), ("Rg", 11, 7), ("Cn", 12, 7), ("Nh", 13, 7), ("Fl", 14, 7), ("Mc", 15, 7), ("Lv", 16, 7),
    ("Ts", 17, 7), ("Og", 18, 7),
    ("La", 3, 8), ("Ce", 4, 8), ("Pr", 5, 8), ("Nd", 6, 8), ("Pm", 7, 8), ("Sm", 8, 8), ("Eu", 9, 8), ("Gd", 10, 8),
    ("Tb", 11, 8), ("Dy", 12, 8), ("Ho", 13, 8), ("Er", 14, 8), ("Tm", 15, 8), ("Yb", 16, 8), ("Lu", 17, 8),
    ("Ac", 3, 9), ("Th", 4, 9), ("Pa", 5, 9), ("U", 6, 9), ("Np", 7, 9), ("Pu", 8, 9), ("Am", 9, 9), ("Cm", 10, 9), ("Bk", 11, 9),
    ("Cf", 12, 9), ("Es", 13, 9), ("Fm", 14, 9), ("Md", 15, 9), ("No", 16, 9), ("Lr", 17, 9),
]
def load_panel_a():
    return base.load_panel_a()


def load_panel_b():
    return base.load_panel_b()


def load_panel_c():
    return base.load_panel_c()


def _mix_with_white(hex_color: str, fraction: float) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(hex_color), dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)
    return tuple(white * (1.0 - fraction) + rgb * fraction)

def _panel_a_fill(failure_rate: float, max_rate: float, has_data: bool) -> tuple[float, float, float]:
    if not has_data:
        return to_rgb(PANEL_A_ABSENT_FILL)
    norm = failure_rate / max_rate if max_rate else 0.0
    fraction = 0.08 + 0.88 * norm
    return _mix_with_white(PANEL_A_PRESENT_BASE, fraction)


def _panel_a_y(period: int) -> float:
    return float(period)


def _format_failure_rate_label(total_materials: int, failure_rate: float) -> str:
    if total_materials <= 0:
        return ""
    if failure_rate < 0.0001:
        return "<0.01%"
    return f"{failure_rate * 100:.2f}%"


def draw_panel_a(ax, element_rates, summary: dict) -> None:
    max_rate = float(element_rates["failure_rate"].max())
    lookup = element_rates.set_index("element").to_dict("index")

    for symbol, group, period in PANEL_A_SYMBOLS:
        row = lookup.get(symbol, None)
        has_data = row is not None
        total_materials = int(row["total_materials"]) if row is not None else 0
        failure_rate = float(row["failure_rate"]) if row is not None else 0.0
        x = float(group)
        y = _panel_a_y(period)
        face = _panel_a_fill(failure_rate, max_rate, has_data)
        text_color = PANEL_A_TEXT
        rect = FancyBboxPatch(
            (x + 0.03, y + 0.03),
            0.94,
            0.94,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=face,
            edgecolor="white",
            linewidth=1.3,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.08,
            y + 0.10,
            str(PANEL_A_Z_ORDER[symbol]),
            ha="left",
            va="top",
            fontsize=6.0,
            color=PANEL_A_SECONDARY_TEXT,
        )
        ax.text(
            x + 0.5,
            y + 0.46,
            symbol,
            ha="center",
            va="center",
            fontsize=10.6,
            fontweight="normal",
            color=text_color,
        )
        rate_label = _format_failure_rate_label(total_materials, failure_rate)
        if rate_label:
            ax.text(
                x + 0.5,
                y + 0.78,
                rate_label,
                ha="center",
                va="center",
                fontsize=6.0,
                color=PANEL_A_TEXT,
            )

    ax.set_xlim(1, 19)
    ax.set_ylim(10.25, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        13.0,
        1.88,
        f"{summary['collective_failure_total']:,} failures | 80% vote threshold",
        fontsize=8.0,
        ha="left",
        va="center",
        color=LIGHT_TEXT,
    )
    add_panel_label(ax, "a", x=-0.015, y=1.01)


def draw_panel_b(ax, data: dict) -> None:
    style_axes(ax, grid=False)
    groups = [
        ("All materials", data["all_materials"]),
        ("Minority support", data["minority"]),
        ("Dominant support", data["dominant"]),
        ("Singleton", data["singleton"]),
    ]
    x = np.arange(len(groups), dtype=float)
    width = 0.28

    for idx, (label, values) in enumerate(groups):
        failure_color = BASE_FAILURE if label == "All materials" else FAILURE_BAR
        fn_color = BASE_FN if label == "All materials" else FN_BAR
        ax.bar(x[idx] - width / 2, values["failure"], width=width, color=failure_color, edgecolor="none", zorder=3)
        ax.bar(x[idx] + width / 2, values["fn"], width=width, color=fn_color, edgecolor="none", zorder=3)
        ax.text(
            x[idx] - width / 2,
            values["failure"] + 0.008,
            f"{values['failure']:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=NEUTRAL_TEXT,
        )

    hotspot_x = len(groups) + 0.62
    ax.bar(hotspot_x, data["tetragonal"]["failure"], width=0.36, color=FAILURE_HOTSPOT, edgecolor="none", zorder=3)
    ax.text(
        hotspot_x,
        data["tetragonal"]["failure"] + 0.008,
        f"{data['tetragonal']['failure']:.3f}",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=NEUTRAL_TEXT,
    )

    ax.set_xticks(list(x) + [hotspot_x])
    ax.set_xticklabels(
        ["All materials", "Minority\nsupport", "Dominant\nsupport", "Singleton", "Tetragonal\nhotspot"],
        fontsize=6.5,
    )
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 0.30)
    ax.set_xlim(-0.65, hotspot_x + 1.65)
    baseline = data["all_materials"]["failure"]
    for idx, (_, values) in enumerate(groups[1:], start=1):
        ax.text(
            x[idx],
            0.283,
            f"{values['failure'] / baseline:.1f}x",
            ha="center",
            va="top",
            fontsize=6.1,
            color=LIGHT_TEXT,
        )

    handles = [
        Patch(facecolor=FAILURE_BAR, edgecolor="none", label="Collective failure"),
        Patch(facecolor=FN_BAR, edgecolor="none", label="False negative"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.92), fontsize=6.4, ncol=1, handlelength=1.8)

    note_x = hotspot_x + 0.28
    ax.text(
        note_x,
        0.248,
        "Tetragonal hotspot",
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=FAILURE_HOTSPOT,
    )
    ax.text(
        note_x,
        0.225,
        f"failure = {data['tetragonal']['failure']:.3f}\nFN = {data['tetragonal']['fn']:.3f}\nn = {data['tetragonal']['n']}",
        ha="left",
        va="top",
        fontsize=6.3,
        color=ANNOTATION_DARK,
    )
    ax.text(
        note_x,
        0.176,
        "High-risk singleton",
        ha="left",
        va="top",
        fontsize=6.9,
        fontweight="bold",
        color=FAILURE_BAR,
    )
    ax.text(
        note_x,
        0.154,
        f"failure = {data['high_risk_singleton']['failure']:.3f}\nFN = {data['high_risk_singleton']['fn']:.3f}",
        ha="left",
        va="top",
        fontsize=6.3,
        color=ANNOTATION_DARK,
    )

    add_panel_label(ax, "b", x=-0.11, y=1.015)


def _wrap_mode_title(title: str) -> str:
    return title.replace(" / ", " /\n", 1)


def draw_mode_card(ax, x: float, y: float, w: float, h: float, *, title: str, n: int, failure: float, fn: float, fp: float, style_key: str) -> None:
    style = base.MODE_STYLE[style_key]
    outer = Rectangle((x, y), w, h, linewidth=0.9, edgecolor=CARD_EDGE, facecolor="white", transform=ax.transAxes)
    strip = Rectangle((x, y), 0.012, h, transform=ax.transAxes, facecolor=style["strip"], edgecolor="none")
    ax.add_patch(outer)
    ax.add_patch(strip)
    ax.text(
        x + 0.03,
        y + h - 0.075,
        _wrap_mode_title(title),
        transform=ax.transAxes,
        fontsize=6.0,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111111",
        linespacing=1.02,
    )
    ax.text(x + 0.03, y + h - 0.16, f"n = {n}", transform=ax.transAxes, fontsize=5.8, ha="left", va="top", color=NEUTRAL_TEXT)
    ax.text(x + 0.03, y + h - 0.28, f"Failure = {failure:.2f}", transform=ax.transAxes, fontsize=5.8, ha="left", va="top", color=NEUTRAL_TEXT)
    ax.text(x + 0.03, y + h - 0.40, f"FN = {fn:.2f}", transform=ax.transAxes, fontsize=5.8, ha="left", va="top", color=NEUTRAL_TEXT)
    ax.text(x + 0.03, y + h - 0.52, f"FP = {fp:.2f}", transform=ax.transAxes, fontsize=5.8, ha="left", va="top", color=NEUTRAL_TEXT)


def draw_panel_c(ax, data: dict) -> None:
    ax.set_axis_off()
    ax.set_facecolor("white")

    top = data["top"]
    top_box = Rectangle((0.13, 0.74), 0.72, 0.17, linewidth=1.0, edgecolor=CARD_EDGE, facecolor="white", transform=ax.transAxes)
    ax.add_patch(top_box)
    ax.text(0.17, 0.84, "tI10 branch", transform=ax.transAxes, fontsize=8.2, fontweight="bold", ha="left", va="center", color="#111111")
    ax.text(
        0.17,
        0.775,
        f"n = {top['n_materials']}    failure = {top['failure']:.3f}    FN = {top['fn']:.3f}",
        transform=ax.transAxes,
        fontsize=6.3,
        ha="left",
        va="center",
        color=NEUTRAL_TEXT,
    )
    ax.text(0.17, 0.725, "(Pearson symbol tI10)", transform=ax.transAxes, fontsize=5.9, ha="left", va="center", color=LIGHT_TEXT)
    ax.text(
        0.81,
        0.81,
        "sg_107 split:\nWyckoff-only change",
        transform=ax.transAxes,
        fontsize=5.5,
        ha="right",
        va="center",
        color=LIGHT_TEXT,
        linespacing=1.0,
    )

    child_positions = {
        "sg_87_other": (0.02, 0.11, 0.22, 0.42),
        "sg_139_a_e_d": (0.27, 0.11, 0.22, 0.42),
        "sg_107_a_ab_a": (0.52, 0.11, 0.21, 0.42),
        "sg_107_a_a_ab": (0.76, 0.11, 0.21, 0.42),
    }

    for key, box in child_positions.items():
        mode = data["modes"][key]
        draw_mode_card(ax, *box, title=mode["title"], n=mode["n"], failure=mode["failure"], fn=mode["fn"], fp=mode["fp"], style_key=mode["style"])

    top_anchor = (0.49, 0.74)
    left_targets = [
        (child_positions["sg_87_other"][0] + child_positions["sg_87_other"][2] / 2, 0.55),
        (child_positions["sg_139_a_e_d"][0] + child_positions["sg_139_a_e_d"][2] / 2, 0.55),
    ]
    left_pair_center = child_positions["sg_107_a_ab_a"][0] + child_positions["sg_107_a_ab_a"][2] / 2
    right_pair_center = child_positions["sg_107_a_a_ab"][0] + child_positions["sg_107_a_a_ab"][2] / 2
    sg107_cluster = ((left_pair_center + right_pair_center) / 2, 0.64)

    for target in left_targets:
        ax.add_patch(
            FancyArrowPatch(
                top_anchor,
                target,
                transform=ax.transAxes,
                arrowstyle="-",
                mutation_scale=8,
                linewidth=0.9,
                color=CONNECTOR_GRAY,
            )
        )
    ax.add_patch(
        FancyArrowPatch(
            top_anchor,
            sg107_cluster,
            transform=ax.transAxes,
            arrowstyle="-",
            mutation_scale=8,
            linewidth=0.9,
            color=CONNECTOR_GRAY,
        )
    )
    for target in [(left_pair_center, 0.55), (right_pair_center, 0.55)]:
        ax.add_patch(
            FancyArrowPatch(
                sg107_cluster,
                target,
                transform=ax.transAxes,
                arrowstyle="-",
                mutation_scale=8,
                linewidth=0.9,
                color=CONNECTOR_GRAY,
            )
        )

    ax.text(0.79, 0.585, "FN -> FP", transform=ax.transAxes, fontsize=5.5, ha="center", va="bottom", color=LIGHT_TEXT)
    ax.add_patch(
        FancyArrowPatch(
            (left_pair_center, 0.57),
            (right_pair_center, 0.57),
            transform=ax.transAxes,
            arrowstyle="->",
            mutation_scale=10,
            linewidth=0.9,
            color=CONNECTOR_GRAY,
        )
    )

    add_panel_label(ax, "c", x=-0.03, y=0.99)


def create_panel_a(output_stems: list[Path], rates, summary: dict) -> None:
    fig = plt.figure(figsize=PANEL_A_FIGSIZE_IN)
    ax = fig.add_subplot(111)
    draw_panel_a(ax, rates, summary)
    fig.subplots_adjust(left=0.025, right=0.965, top=0.91, bottom=0.08)
    save_figure_to_many(fig, output_stems)


def create_panel_b(output_stems: list[Path], data: dict) -> None:
    fig = create_figure_mm(*PANEL_B_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_b(ax, data)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.89, bottom=0.17)
    save_figure_to_many(fig, output_stems)


def create_panel_c(output_stems: list[Path], data: dict) -> None:
    fig = create_figure_mm(*PANEL_C_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_c(ax, data)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.90, bottom=0.06)
    save_figure_to_many(fig, output_stems)


def _add_preview_panel(fig, image_path: Path, outer_box: list[float]) -> None:
    x0, y0, w_box, h_box = outer_box
    image = plt.imread(image_path)
    image_h, image_w = image.shape[:2]
    image_aspect = image_w / image_h
    box_aspect = (w_box * PREVIEW_SIZE_MM[0]) / (h_box * PREVIEW_SIZE_MM[1])

    if image_aspect >= box_aspect:
        w = w_box
        h = w_box * PREVIEW_SIZE_MM[0] / (image_aspect * PREVIEW_SIZE_MM[1])
        x = x0
        y = y0 + (h_box - h) / 2
    else:
        h = h_box
        w = h_box * PREVIEW_SIZE_MM[1] * image_aspect / PREVIEW_SIZE_MM[0]
        x = x0 + (w_box - w) / 2
        y = y0

    axis = fig.add_axes([x, y, w, h])
    axis.imshow(image, aspect="auto")
    axis.axis("off")


def create_preview(output_stems: list[Path], package_dir: Path) -> None:
    fig = create_figure_mm(*PREVIEW_SIZE_MM)
    fig.patch.set_facecolor("white")
    placements = [
        ("panel_a_periodic_table.png", [0.03, 0.40, 0.94, 0.56]),
        ("panel_b_support_stratification.png", [0.02, 0.05, 0.54, 0.27]),
        ("panel_c_ti10_branch.png", [0.58, 0.05, 0.40, 0.27]),
    ]
    for file_name, position in placements:
        _add_preview_panel(fig, package_dir / file_name, position)
    save_figure_to_many(fig, output_stems)


def write_assembly_notes(output_dirs: list[Path], panel_b_data: dict) -> None:
    contents = "\n".join(
        [
            "# Figure 4 Nature-style Variant",
            "",
            "This directory contains a style-only variant of main Figure 4.",
            "Data, values, and panel ordering match the original `figure_04_collective_failures` export.",
            "",
            "Panel changes:",
            "- panel a keeps the periodic-table heatmap but simplifies the colorbar treatment",
            "- panel b replaces boxed callouts with whitespace-aligned text annotations",
            "- panel c redraws the branch summary as a cleaner flow chart with lighter connectors",
            "",
            "Key panel b values:",
            f"- all materials failure = {panel_b_data['all_materials']['failure']:.4f}, FN = {panel_b_data['all_materials']['fn']:.4f}",
            f"- minority failure = {panel_b_data['minority']['failure']:.4f}, dominant failure = {panel_b_data['dominant']['failure']:.4f}, singleton failure = {panel_b_data['singleton']['failure']:.4f}",
            f"- high-risk singleton failure = {panel_b_data['high_risk_singleton']['failure']:.4f}, FN = {panel_b_data['high_risk_singleton']['fn']:.4f}",
            f"- tetragonal hotspot failure = {panel_b_data['tetragonal']['failure']:.4f}, FN = {panel_b_data['tetragonal']['fn']:.4f}, n = {panel_b_data['tetragonal']['n']}",
            "",
        ]
    )
    for output_dir in output_dirs:
        ensure_dir(output_dir)
        (output_dir / "assembly_notes.md").write_text(contents, encoding="utf-8")


def main() -> None:
    configure_publication_style()

    results_dir = RESULTS_DIR / "figures" / "figure_04_collective_failures_nature_variant"
    package_dir = (
        PROJECT_ROOT
        / "paper"
        / "submission_package"
        / "figures"
        / "main"
        / "figure_04_collective_failures_nature_variant"
    )
    output_dirs = [results_dir, package_dir]
    for output_dir in output_dirs:
        ensure_dir(output_dir)

    panel_a_rates, panel_a_summary = load_panel_a()
    panel_b_data = load_panel_b()
    panel_c_data = load_panel_c()

    create_panel_a([output_dir / "panel_a_periodic_table" for output_dir in output_dirs], panel_a_rates, panel_a_summary)
    create_panel_b([output_dir / "panel_b_support_stratification" for output_dir in output_dirs], panel_b_data)
    create_panel_c([output_dir / "panel_c_ti10_branch" for output_dir in output_dirs], panel_c_data)
    create_preview([output_dir / "figure_04_preview" for output_dir in output_dirs], package_dir)
    write_assembly_notes(output_dirs, panel_b_data)


if __name__ == "__main__":
    main()

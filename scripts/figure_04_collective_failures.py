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
from matplotlib import cm, colors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from dva_project.figure_style import (
    add_panel_label,
    configure_publication_style,
    create_figure_mm,
    save_figure_to_many,
    style_axes,
)
from dva_project.settings import PROJECT_ROOT, RESULTS_DIR
from dva_project.utils import ensure_dir


PANEL_A_SIZE_MM = (180.0, 52.0)
PANEL_B_SIZE_MM = (98.0, 58.0)
PANEL_C_SIZE_MM = (72.0, 58.0)
PREVIEW_SIZE_MM = (180.0, 90.0)

FAILURE_COLOR = "#B31B1B"
FN_COLOR = "#F2A7A0"
GROUP_COLORS = {
    "All materials": "#9E9E9E",
    "Minority support": "#F7C8C1",
    "Dominant support": "#EB7D6E",
    "Singleton": "#C73D32",
    "Tetragonal singleton": "#8F0D13",
}
MODE_STYLE = {
    "pure_fn": {"strip": "#D55E00", "fill": "#FFF1E8"},
    "fn_heavy": {"strip": "#B23A48", "fill": "#FCECEE"},
    "pure_fp": {"strip": "#0072B2", "fill": "#EAF4FB"},
}


def load_panel_a() -> tuple[pd.DataFrame, dict]:
    rates = pd.read_csv(RESULTS_DIR / "tables" / "analysis_04" / "element_failure_rates.csv")
    summary = json.loads((RESULTS_DIR / "tables" / "analysis_04" / "summary.json").read_text(encoding="utf-8"))
    return rates, summary


def load_panel_b() -> dict:
    density = pd.read_csv(RESULTS_DIR / "tables" / "analysis_11" / "density_tier_rates.csv")
    crystal = pd.read_csv(RESULTS_DIR / "tables" / "analysis_11" / "singleton_crystal_system_rates.csv")
    outcomes = pd.read_csv(RESULTS_DIR / "tables" / "analysis_04" / "collective_outcomes.csv")

    exact = density.loc[density["subset"] == "exact_formula_nsites_subset"].copy()
    high_risk = density.loc[density["subset"] == "high_risk_nsites_subset"].copy()
    tetragonal = crystal.loc[
        (crystal["subset"] == "high_risk_nsites_subset") & (crystal["crystal_system"] == "tetragonal")
    ].iloc[0]

    all_failure = float(outcomes["collective_failure"].mean())
    all_fn = float(outcomes["collective_false_negative"].mean())

    tier_lookup = exact.set_index("exact_formula_nsites_density_tier")
    high_risk_singleton = high_risk.set_index("exact_formula_nsites_density_tier").loc["singleton_formula_signature"]
    return {
        "all_materials": {
            "failure": all_failure,
            "fn": all_fn,
        },
        "minority": {
            "failure": float(tier_lookup.loc["minority_multi_signature", "collective_failure_rate"]),
            "fn": float(tier_lookup.loc["minority_multi_signature", "collective_false_negative_rate"]),
        },
        "dominant": {
            "failure": float(tier_lookup.loc["dominant_multi_signature", "collective_failure_rate"]),
            "fn": float(tier_lookup.loc["dominant_multi_signature", "collective_false_negative_rate"]),
        },
        "singleton": {
            "failure": float(tier_lookup.loc["singleton_formula_signature", "collective_failure_rate"]),
            "fn": float(tier_lookup.loc["singleton_formula_signature", "collective_false_negative_rate"]),
        },
        "high_risk_singleton": {
            "failure": float(high_risk_singleton["collective_failure_rate"]),
            "fn": float(high_risk_singleton["collective_false_negative_rate"]),
        },
        "tetragonal": {
            "failure": float(tetragonal["collective_failure_rate"]),
            "fn": float(tetragonal["collective_false_negative_rate"]),
            "n": int(tetragonal["n_materials"]),
        },
    }


def load_panel_c() -> dict:
    summary = json.loads((RESULTS_DIR / "tables" / "analysis_22" / "summary.json").read_text(encoding="utf-8"))
    ti10_rate = pd.read_csv(RESULTS_DIR / "tables" / "analysis_15" / "pearson_symbol_bucket_rates.csv")
    ti10_row = ti10_rate.loc[ti10_rate["frequent_pearson_symbol_bucket"] == "tI10"].iloc[0]
    branch = summary["tI10_branch"]["key_structural_modes"]
    return {
        "top": {
            "n_materials": int(ti10_row["n_materials"]),
            "failure": float(ti10_row["collective_failure_rate"]),
            "fn": float(ti10_row["collective_false_negative_rate"]),
        },
        "modes": {
            "sg_87_other": {
                "title": "sg_87 / other",
                "n": int(branch["sg_87__other"]["n_materials"]),
                "failure": float(branch["sg_87__other"]["collective_failure_rate"]),
                "fn": float(branch["sg_87__other"]["collective_false_negative_rate"]),
                "fp": float(branch["sg_87__other"]["collective_false_positive_rate"]),
                "style": "pure_fn",
            },
            "sg_139_a_e_d": {
                "title": "sg_139 / a_e_d",
                "n": int(branch["sg_139__a_e_d"]["n_materials"]),
                "failure": float(branch["sg_139__a_e_d"]["collective_failure_rate"]),
                "fn": float(branch["sg_139__a_e_d"]["collective_false_negative_rate"]),
                "fp": float(branch["sg_139__a_e_d"]["collective_false_positive_rate"]),
                "style": "pure_fn",
            },
            "sg_107_a_ab_a": {
                "title": "sg_107 / a_ab_a",
                "n": int(branch["sg_107__a_ab_a"]["n_materials"]),
                "failure": float(branch["sg_107__a_ab_a"]["collective_failure_rate"]),
                "fn": float(branch["sg_107__a_ab_a"]["collective_false_negative_rate"]),
                "fp": float(branch["sg_107__a_ab_a"]["collective_false_positive_rate"]),
                "style": "fn_heavy",
            },
            "sg_107_a_a_ab": {
                "title": "sg_107 / a_a_ab",
                "n": int(branch["sg_107__a_a_ab"]["n_materials"]),
                "failure": float(branch["sg_107__a_a_ab"]["collective_failure_rate"]),
                "fn": float(branch["sg_107__a_a_ab"]["collective_false_negative_rate"]),
                "fp": float(branch["sg_107__a_a_ab"]["collective_false_positive_rate"]),
                "style": "pure_fp",
            },
        },
    }


def draw_panel_a(ax, element_rates: pd.DataFrame, summary: dict) -> None:
    style_axes(ax, show_left=False, show_bottom=False, show_top=False, show_right=False, grid=False)
    cmap = plt.get_cmap("Reds")
    norm = colors.Normalize(vmin=0.0, vmax=float(element_rates["failure_rate"].max()))

    for row in element_rates.itertuples(index=False):
        x = float(row.group)
        y = float(row.period)
        face = cmap(norm(row.failure_rate))
        rect = Rectangle((x, y), 1.0, 1.0, facecolor=face, edgecolor="white", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.5, y + 0.38, row.element, ha="center", va="center", fontsize=8, fontweight="bold", color="#111111")
        ax.text(x + 0.5, y + 0.73, f"{row.failure_rate * 100:.2f}%", ha="center", va="center", fontsize=5.7, color="#333333")

    ax.set_xlim(1, 19)
    ax.set_ylim(10, 0)
    ax.set_xticks(range(1, 19))
    ax.set_yticks(range(1, 10))
    ax.tick_params(axis="x", labelsize=6.5, length=0)
    ax.tick_params(axis="y", labelsize=6.5, length=0)
    ax.set_xlabel("Group")
    ax.set_ylabel("Period / series")
    ax.set_title("Element-level concentration of collective failures", loc="left", pad=8)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    colorbar.set_label("Fraction of materials containing element in collective-failure set", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)

    ax.text(
        0.99,
        1.02,
        f"{summary['collective_failure_total']:,} failures | 80% vote threshold",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        color="#444444",
    )
    add_panel_label(ax, "a", x=-0.03, y=1.04)


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
        base_color = GROUP_COLORS[label]
        ax.bar(
            x[idx] - width / 2,
            values["failure"],
            width=width,
            color=base_color,
            edgecolor="#222222",
            linewidth=0.5,
            zorder=3,
        )
        ax.bar(
            x[idx] + width / 2,
            values["fn"],
            width=width,
            color=FN_COLOR if label != "All materials" else "#D0D0D0",
            edgecolor="#222222",
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            x[idx] - width / 2,
            values["failure"] + 0.008,
            f"{values['failure']:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#333333",
        )

    hotspot_x = len(groups) + 0.55
    ax.bar(
        hotspot_x,
        data["tetragonal"]["failure"],
        width=0.34,
        color=GROUP_COLORS["Tetragonal singleton"],
        edgecolor="#222222",
        linewidth=0.5,
        zorder=3,
    )
    ax.text(
        hotspot_x,
        data["tetragonal"]["failure"] + 0.008,
        f"{data['tetragonal']['failure']:.3f}",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#333333",
    )
    ax.text(
        hotspot_x,
        0.288,
        f"Tetragonal singleton\nfailure {data['tetragonal']['failure']:.3f}, n={data['tetragonal']['n']}\nFN {data['tetragonal']['fn']:.3f}",
        ha="center",
        va="top",
        fontsize=6.3,
        color="#6A0F14",
        bbox={"facecolor": "white", "edgecolor": "#D3B1B5", "boxstyle": "round,pad=0.25"},
    )

    ax.set_xticks(list(x) + [hotspot_x])
    ax.set_xticklabels(
        ["All materials", "Minority\nsupport", "Dominant\nsupport", "Singleton", "Tetragonal\nhotspot"],
        fontsize=6.7,
    )
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 0.30)
    ax.set_xlim(-0.6, hotspot_x + 0.7)
    ax.set_title("Sparse support within familiar chemistry sharply elevates failure risk", loc="left")

    baseline = data["all_materials"]["failure"]
    for idx, (_, values) in enumerate(groups[1:], start=1):
        ax.text(
            x[idx],
            0.286,
            f"{values['failure'] / baseline:.1f}x",
            ha="center",
            va="top",
            fontsize=6.2,
            color="#555555",
        )

    ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor=FAILURE_COLOR, edgecolor="#222222", linewidth=0.5, label="Collective failure"),
            Rectangle((0, 0), 1, 1, facecolor=FN_COLOR, edgecolor="#222222", linewidth=0.5, label="False negative"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=6.5,
    )

    ax.text(
        x[3] + 0.42,
        0.20,
        f"High-risk singleton\nfailure {data['high_risk_singleton']['failure']:.3f}\nFN {data['high_risk_singleton']['fn']:.3f}",
        fontsize=6.4,
        ha="left",
        va="center",
        color="#7A1416",
        bbox={"facecolor": "white", "edgecolor": "#C95F5F", "boxstyle": "round,pad=0.25", "linestyle": "--"},
    )
    add_panel_label(ax, "b")


def draw_mode_box(ax, x: float, y: float, w: float, h: float, *, title: str, n: int, failure: float, fn: float, fp: float, style_key: str) -> None:
    style = MODE_STYLE[style_key]
    outer = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#333333",
        facecolor=style["fill"],
        transform=ax.transAxes,
    )
    strip = Rectangle((x, y), 0.018, h, transform=ax.transAxes, facecolor=style["strip"], edgecolor="none")
    ax.add_patch(outer)
    ax.add_patch(strip)
    ax.text(x + 0.03, y + h - 0.06, title, transform=ax.transAxes, fontsize=7.2, fontweight="bold", ha="left", va="top", color="#111111")
    ax.text(x + 0.03, y + h - 0.12, f"n = {n}", transform=ax.transAxes, fontsize=6.3, ha="left", va="top", color="#333333")
    ax.text(x + 0.03, y + h - 0.20, f"Fail = {failure:.2f}", transform=ax.transAxes, fontsize=6.3, ha="left", va="top", color="#333333")
    ax.text(x + 0.03, y + h - 0.28, f"FN = {fn:.2f}", transform=ax.transAxes, fontsize=6.3, ha="left", va="top", color="#333333")
    ax.text(x + 0.03, y + h - 0.36, f"FP = {fp:.2f}", transform=ax.transAxes, fontsize=6.3, ha="left", va="top", color="#333333")


def draw_panel_c(ax, data: dict) -> None:
    ax.set_axis_off()
    ax.set_facecolor("white")

    top = data["top"]
    top_box = FancyBboxPatch(
        (0.16, 0.72),
        0.66,
        0.14,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.1,
        edgecolor="#333333",
        facecolor="#F7F7F7",
        transform=ax.transAxes,
    )
    ax.add_patch(top_box)
    ax.text(0.20, 0.82, "tI10 branch", transform=ax.transAxes, fontsize=8.3, fontweight="bold", ha="left", va="center")
    ax.text(
        0.20,
        0.76,
        f"n = {top['n_materials']}    failure = {top['failure']:.3f}    FN = {top['fn']:.3f}",
        transform=ax.transAxes,
        fontsize=6.5,
        ha="left",
        va="center",
        color="#333333",
    )
    ax.text(0.20, 0.71, "(Pearson symbol tI10)", transform=ax.transAxes, fontsize=6.0, ha="left", va="center", color="#666666")

    child_positions = {
        "sg_87_other": (0.02, 0.10, 0.22, 0.38),
        "sg_139_a_e_d": (0.27, 0.10, 0.22, 0.38),
        "sg_107_a_ab_a": (0.52, 0.10, 0.20, 0.38),
        "sg_107_a_a_ab": (0.76, 0.10, 0.20, 0.38),
    }
    for key, box in child_positions.items():
        mode = data["modes"][key]
        draw_mode_box(ax, *box, title=mode["title"], n=mode["n"], failure=mode["failure"], fn=mode["fn"], fp=mode["fp"], style_key=mode["style"])
        cx = box[0] + box[2] / 2
        ax.add_patch(
            FancyArrowPatch(
                (0.49, 0.72),
                (cx, 0.50),
                transform=ax.transAxes,
                arrowstyle="-",
                mutation_scale=8,
                linewidth=1.0,
                color="#777777",
            )
        )

    ax.text(
        0.74,
        0.57,
        "Same sg_107\nWyckoff signature flips\nFN to FP",
        transform=ax.transAxes,
        fontsize=6.1,
        ha="center",
        va="bottom",
        color="#444444",
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.68, 0.47),
            (0.79, 0.47),
            transform=ax.transAxes,
            arrowstyle="<->",
            mutation_scale=11,
            linewidth=1.0,
            color="#666666",
        )
    )
    ax.text(0.08, 1.00, "tI10 branch decomposition", transform=ax.transAxes, fontsize=9, ha="left", va="bottom", color="#111111")
    add_panel_label(ax, "c", x=-0.03, y=0.99)


def create_panel_a(output_stems: list[Path], rates: pd.DataFrame, summary: dict) -> None:
    fig = create_figure_mm(*PANEL_A_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_a(ax, rates, summary)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.12)
    save_figure_to_many(fig, output_stems)


def create_panel_b(output_stems: list[Path], data: dict) -> None:
    fig = create_figure_mm(*PANEL_B_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_b(ax, data)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.89, bottom=0.16)
    save_figure_to_many(fig, output_stems)


def create_panel_c(output_stems: list[Path], data: dict) -> None:
    fig = create_figure_mm(*PANEL_C_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_c(ax, data)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.06)
    save_figure_to_many(fig, output_stems)


def create_preview(output_stems: list[Path], package_dir: Path) -> None:
    fig = create_figure_mm(*PREVIEW_SIZE_MM)
    fig.patch.set_facecolor("white")
    placements = [
        ("panel_a_periodic_table.png", [0.01, 0.53, 0.98, 0.43]),
        ("panel_b_support_stratification.png", [0.01, 0.05, 0.55, 0.42]),
        ("panel_c_ti10_branch.png", [0.60, 0.05, 0.38, 0.42]),
    ]
    for file_name, position in placements:
        axis = fig.add_axes(position)
        axis.imshow(plt.imread(package_dir / file_name))
        axis.axis("off")
    save_figure_to_many(fig, output_stems)


def write_assembly_notes(output_dirs: list[Path], panel_b_data: dict) -> None:
    contents = "\n".join(
        [
            "# Figure 4 Assembly Notes",
            "",
            "Panels exported as individual PDF and PNG assets.",
            "",
            "Suggested layout:",
            "- top full-width: `panel_a_periodic_table.pdf`",
            "- bottom left: `panel_b_support_stratification.pdf`",
            "- bottom right: `panel_c_ti10_branch.pdf`",
            "",
            "Preview file:",
            "- `figure_04_preview.png` / `figure_04_preview.pdf`",
            "",
            "Panel b note:",
            "- the leftmost `All materials` bar uses the full benchmark baseline from `analysis_04/collective_outcomes.csv`",
            "- the `Minority / Dominant / Singleton` support tiers use the exact-formula-plus-n_sites subset from `analysis_11/density_tier_rates.csv`",
            "- the high-risk singleton callout and tetragonal hotspot use the `high_risk_nsites_subset` rows from analysis 11",
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

    results_dir = RESULTS_DIR / "figures" / "figure_04_collective_failures"
    package_dir = (
        PROJECT_ROOT
        / "paper"
        / "submission_package"
        / "figures"
        / "main"
        / "figure_04_collective_failures"
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

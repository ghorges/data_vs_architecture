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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from dva_project.figure_style import (
    TRAINING_REGIME_COLORS,
    add_panel_label,
    configure_publication_style,
    create_figure_mm,
    save_figure_to_many,
    style_axes,
)
from dva_project.settings import PROJECT_ROOT, RESULTS_DIR
from dva_project.utils import ensure_dir


PANEL_A_SIZE_MM = (70.0, 60.0)
PANEL_B_SIZE_MM = (120.0, 60.0)
PANEL_CDE_SIZE_MM = (57.0, 45.0)
PREVIEW_SIZE_MM = (180.0, 130.0)

LEADERBOARD_MODELS = [
    ("eSEN-30m-oam", "eSEN-30M-OAM"),
    ("equflash-29M-oam", "EquFlash-29M-OAM"),
    ("alphanet-v1-oma", "AlphaNet-v1-OMA"),
    ("orb-v3", "ORB v3"),
    ("eSEN-30m-mp", "eSEN-30M-MP"),
    ("eqnorm-mptrj", "Eqnorm MPtrj"),
]
TRAINING_DISPLAY = {
    "MPtrj__OMat24__sAlex": "OMat24+sAlex+MPtrj",
    "Alex__MPtrj__OMat24": "MPtrj+Alex+OMat24",
    "MPtrj__sAlex": "MPtrj+sAlex",
    "MPtrj": "MPtrj only",
    "MatterSim": "MatterSim",
    "OpenLAM": "OpenLAM",
    "MP 2022": "MPF/MP2022",
}


def load_leaderboard_panel() -> pd.DataFrame:
    metadata = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "model_metadata_snapshot_45.csv")
    metadata = metadata.sort_values("f1_full_test", ascending=False).reset_index(drop=True)
    metadata["rank"] = metadata.index + 1
    rows = []
    for model_key, display_name in LEADERBOARD_MODELS:
        row = metadata.loc[metadata["model_key"] == model_key].iloc[0]
        rows.append(
            {
                "rank": int(row["rank"]),
                "model_key": model_key,
                "display_name": display_name,
                "f1": float(row["f1_full_test"]),
                "training_combo": row["training_combo"],
                "architecture_group": row["architecture_group"],
            }
        )
    return pd.DataFrame(rows)


def load_framework_panel() -> dict:
    overview = json.loads((RESULTS_DIR / "tables" / "figure_01" / "overview_summary.json").read_text(encoding="utf-8"))
    return overview


def load_scaling_preview() -> tuple[pd.DataFrame, pd.Series]:
    points = pd.read_csv(RESULTS_DIR / "tables" / "analysis_03" / "data_scaling_points.csv")
    fits = pd.read_csv(RESULTS_DIR / "tables" / "analysis_03" / "data_scaling_fits.csv")
    points = points.loc[points["family"].isin(["eSEN", "nequip", "grace", "allegro"])].copy()
    fit = fits.loc[
        (fits["analysis"] == "data_scaling")
        & (fits["subset"] == "pooled_family_combo_means")
        & (fits["metric"] == "f1_full_test")
    ].iloc[0]
    return points, fit


def load_failure_preview() -> dict:
    density = pd.read_csv(RESULTS_DIR / "tables" / "analysis_11" / "density_tier_rates.csv")
    crystal = pd.read_csv(RESULTS_DIR / "tables" / "analysis_11" / "singleton_crystal_system_rates.csv")
    exact = density.loc[density["subset"] == "exact_formula_nsites_subset"].set_index("exact_formula_nsites_density_tier")
    high_risk = density.loc[density["subset"] == "high_risk_nsites_subset"].set_index("exact_formula_nsites_density_tier")
    tetragonal = crystal.loc[
        (crystal["subset"] == "high_risk_nsites_subset") & (crystal["crystal_system"] == "tetragonal")
    ].iloc[0]
    return {
        "minority": float(exact.loc["minority_multi_signature", "collective_failure_rate"]),
        "dominant": float(exact.loc["dominant_multi_signature", "collective_failure_rate"]),
        "singleton": float(exact.loc["singleton_formula_signature", "collective_failure_rate"]),
        "high_risk_singleton": float(high_risk.loc["singleton_formula_signature", "collective_failure_rate"]),
        "tetragonal": float(tetragonal["collective_failure_rate"]),
    }


def load_pareto_preview() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(RESULTS_DIR / "tables" / "analysis_05" / "model_cost_summary.csv")
    frontier = pd.read_csv(RESULTS_DIR / "tables" / "analysis_05" / "pareto_frontier_proxy.csv")
    metadata["training_display"] = metadata["training_combo"].map(TRAINING_DISPLAY).fillna("MPF/MP2022")
    return metadata, frontier


def draw_panel_a(ax, leaderboard: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.text(
        0.00,
        1.02,
        "Current leaderboard conflates architecture and data",
        transform=ax.transAxes,
        fontsize=8.8,
        ha="left",
        va="bottom",
        color="#111111",
    )

    headers = ["Rank", "Model", "F1", "Data"]
    col_x = [0.04, 0.20, 0.74, 0.88]
    y0 = 0.90
    row_h = 0.11
    for label, xpos in zip(headers, col_x):
        ax.text(
            xpos,
            y0,
            label,
            transform=ax.transAxes,
            fontsize=7.0,
            fontweight="bold",
            ha="left",
            va="center",
            color="#333333",
        )
    ax.plot([0.03, 0.95], [y0 - 0.04, y0 - 0.04], transform=ax.transAxes, color="#BBBBBB", lw=0.6)

    for idx, row in enumerate(leaderboard.itertuples(index=False)):
        ypos = y0 - 0.10 - idx * row_h
        ax.text(col_x[0], ypos, f"#{row.rank}", transform=ax.transAxes, fontsize=7.0, ha="left", va="center", color="#222222")
        ax.text(col_x[1], ypos, row.display_name, transform=ax.transAxes, fontsize=7.0, ha="left", va="center", color="#222222")
        ax.text(col_x[2], ypos, f"{row.f1:.3f}", transform=ax.transAxes, fontsize=7.0, ha="right", va="center", color="#222222")
        color = TRAINING_REGIME_COLORS[TRAINING_DISPLAY.get(row.training_combo, "MPF/MP2022")]
        ax.add_patch(Rectangle((col_x[3], ypos - 0.022), 0.04, 0.044, transform=ax.transAxes, facecolor=color, edgecolor="none"))
        ax.plot([0.03, 0.95], [ypos - 0.055, ypos - 0.055], transform=ax.transAxes, color="#E0E0E0", lw=0.5)

    top_regime_y1 = y0 - 0.10
    top_regime_y2 = y0 - 0.10 - 2 * row_h
    bracket_x = 0.96
    ax.plot([bracket_x, bracket_x], [top_regime_y2, top_regime_y1], transform=ax.transAxes, color="#999999", lw=0.8)
    ax.plot([bracket_x - 0.015, bracket_x], [top_regime_y1, top_regime_y1], transform=ax.transAxes, color="#999999", lw=0.8)
    ax.plot([bracket_x - 0.015, bracket_x], [top_regime_y2, top_regime_y2], transform=ax.transAxes, color="#999999", lw=0.8)
    delta_same_regime = leaderboard.iloc[:3]["f1"].max() - leaderboard.iloc[:3]["f1"].min()
    ax.text(
        0.97,
        (top_regime_y1 + top_regime_y2) / 2,
        f"Same data regime,\n3 architectures\nDelta F1 = {delta_same_regime:.3f}",
        transform=ax.transAxes,
        fontsize=6.2,
        ha="left",
        va="center",
        color="#555555",
    )

    top_esen = leaderboard.loc[leaderboard["model_key"] == "eSEN-30m-oam"].iloc[0]
    low_esen = leaderboard.loc[leaderboard["model_key"] == "eSEN-30m-mp"].iloc[0]
    y_top = y0 - 0.10 - leaderboard.index[leaderboard["model_key"] == "eSEN-30m-oam"][0] * row_h
    y_low = y0 - 0.10 - leaderboard.index[leaderboard["model_key"] == "eSEN-30m-mp"][0] * row_h
    ax.add_patch(
        FancyArrowPatch(
            (0.15, y_top),
            (0.15, y_low),
            transform=ax.transAxes,
            arrowstyle="<->",
            mutation_scale=9,
            linewidth=0.9,
            color="#888888",
        )
    )
    ax.text(
        0.02,
        (y_top + y_low) / 2,
        f"Same architecture\n(eSEN)\nDelta F1 = {top_esen['f1'] - low_esen['f1']:.3f}",
        transform=ax.transAxes,
        fontsize=6.1,
        ha="left",
        va="center",
        color="#555555",
    )
    add_panel_label(ax, "a", x=-0.04, y=1.02)


def round_box(ax, xy, width, height, facecolor="#FFFFFF", edgecolor="#333333", radius=0.02, linewidth=0.8):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    return box


def draw_panel_b(ax, overview: dict) -> None:
    ax.set_axis_off()
    stage_x = [0.03, 0.27, 0.53, 0.80]
    widths = [0.19, 0.20, 0.20, 0.17]
    y_base = 0.11

    ax.text(
        0.00,
        1.01,
        "From benchmark matrix to factorized conclusions",
        transform=ax.transAxes,
        fontsize=8.0,
        ha="left",
        va="bottom",
        color="#111111",
    )

    round_box(ax, (stage_x[0], y_base + 0.28), widths[0], 0.28, facecolor="#F8FBFF")
    ax.text(stage_x[0] + 0.018, y_base + 0.51, "1. Prediction matrix", transform=ax.transAxes, fontsize=6.8, fontweight="bold")
    ax.add_patch(
        Rectangle(
            (stage_x[0] + 0.03, y_base + 0.33),
            0.13,
            0.12,
            transform=ax.transAxes,
            facecolor="#EEF3FA",
            edgecolor="#AAB6C7",
            linewidth=0.7,
        )
    )
    for offset in np.linspace(0.01, 0.12, 4):
        ax.plot(
            [stage_x[0] + 0.03 + offset, stage_x[0] + 0.03 + offset],
            [y_base + 0.33, y_base + 0.45],
            transform=ax.transAxes,
            color="#CFD8E3",
            lw=0.5,
        )
        ax.plot(
            [stage_x[0] + 0.03, stage_x[0] + 0.16],
            [y_base + 0.33 + offset, y_base + 0.33 + offset],
            transform=ax.transAxes,
            color="#CFD8E3",
            lw=0.5,
        )
    ax.text(stage_x[0] + 0.095, y_base + 0.385, f"{overview['snapshot_models']} models", transform=ax.transAxes, fontsize=6.0, ha="center", color="#333333")
    ax.text(stage_x[0] + 0.095, y_base + 0.35, f"x {overview['materials']:,} materials", transform=ax.transAxes, fontsize=5.8, ha="center", color="#333333")

    round_box(ax, (stage_x[1], y_base + 0.20), widths[1], 0.40, facecolor="#FFFFFF")
    ax.text(stage_x[1] + 0.018, y_base + 0.56, "2. Factor extraction", transform=ax.transAxes, fontsize=6.8, fontweight="bold")
    factor_rows = [
        ("Training data", "#009E73", "11 exact combinations"),
        ("Architecture", "#E69F00", "5 groups"),
        ("Parameters", "#999999", "continuous"),
    ]
    for idx, (label, color, subtitle) in enumerate(factor_rows):
        ypos = y_base + 0.46 - idx * 0.105
        round_box(ax, (stage_x[1] + 0.02, ypos), 0.16, 0.075, facecolor="#FFFFFF", edgecolor="#D0D0D0", radius=0.012, linewidth=0.6)
        ax.add_patch(Rectangle((stage_x[1] + 0.02, ypos), 0.010, 0.075, transform=ax.transAxes, facecolor=color, edgecolor="none"))
        ax.text(stage_x[1] + 0.045, ypos + 0.048, label, transform=ax.transAxes, fontsize=6.2, ha="left", va="center")
        ax.text(stage_x[1] + 0.045, ypos + 0.021, subtitle, transform=ax.transAxes, fontsize=5.4, ha="left", va="center", color="#666666")

    round_box(ax, (stage_x[2], y_base + 0.16), widths[2], 0.44, facecolor="#FFFFFF")
    ax.text(stage_x[2] + 0.018, y_base + 0.56, "3. Five analyses", transform=ax.transAxes, fontsize=6.8, fontweight="bold")
    analyses = ["Variance effects", "Error clusters", "Scaling laws", "Failure modes", "Pareto tradeoffs"]
    for idx, name in enumerate(analyses):
        ypos = y_base + 0.47 - idx * 0.078
        ax.add_patch(Circle((stage_x[2] + 0.035, ypos + 0.018), 0.012, transform=ax.transAxes, facecolor="#111111", edgecolor="none"))
        ax.text(stage_x[2] + 0.035, ypos + 0.018, f"{idx + 1}", transform=ax.transAxes, fontsize=4.8, ha="center", va="center", color="white")
        ax.text(stage_x[2] + 0.058, ypos + 0.018, name, transform=ax.transAxes, fontsize=5.9, ha="left", va="center", color="#222222")

    round_box(ax, (stage_x[3], y_base + 0.26), widths[3], 0.26, facecolor="#F8FBF8")
    ax.text(stage_x[3] + 0.018, y_base + 0.48, "4. Conclusion", transform=ax.transAxes, fontsize=6.8, fontweight="bold")
    ax.text(
        stage_x[3] + 0.018,
        y_base + 0.42,
        f"Training data eta^2 = {overview['analysis_callouts']['analysis_01_f1_train_eta_sq']:.2f}",
        transform=ax.transAxes,
        fontsize=6.2,
        color="#009E73",
        fontweight="bold",
    )
    ax.text(
        stage_x[3] + 0.018,
        y_base + 0.36,
        f"Architecture eta^2 = {overview['analysis_callouts']['analysis_01_f1_arch_eta_sq']:.2f}",
        transform=ax.transAxes,
        fontsize=6.0,
        color="#E69F00",
    )
    ax.text(
        stage_x[3] + 0.018,
        y_base + 0.29,
        "consistent across metrics,\npermutations, and live validation",
        transform=ax.transAxes,
        fontsize=5.4,
        color="#666666",
    )

    for idx in range(len(stage_x) - 1):
        left = stage_x[idx]
        right = stage_x[idx + 1]
        ax.add_patch(
            FancyArrowPatch(
                (left + widths[idx] + 0.01, y_base + 0.38),
                (right - 0.01, y_base + 0.38),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="#777777",
            )
        )
    add_panel_label(ax, "b", x=-0.03, y=1.02)


def draw_panel_c(ax, points: pd.DataFrame, fit: pd.Series) -> None:
    style_axes(ax, grid=False)
    family_colors = {"eSEN": "#E69F00", "nequip": "#56B4E9", "grace": "#009E73", "allegro": "#D55E00"}
    for family, frame in points.groupby("family"):
        frame = frame.sort_values("effective_training_structures")
        ax.plot(frame["effective_training_structures"], frame["f1_full_test"], marker="o", ms=3, lw=1.1, color=family_colors[family])
    x_grid = np.geomspace(points["effective_training_structures"].min(), points["effective_training_structures"].max(), 120)
    y_grid = fit["intercept"] + fit["slope_per_log10"] * np.log10(x_grid)
    ax.plot(x_grid, y_grid, color="#111111", lw=1.2, ls=(0, (4, 2)))
    ax.set_xscale("log")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("Training size")
    ax.set_ylabel("F1")
    ax.set_title("Scaling preview", loc="left", fontsize=8.1)
    ax.text(0.98, 0.94, f"Data scaling: +{fit['slope_per_log10']:.3f}/decade", transform=ax.transAxes, fontsize=5.9, ha="right", va="top")
    ax.text(0.02, 0.06, "Param. scaling: +0.063/decade", transform=ax.transAxes, fontsize=5.9, ha="left", va="bottom")
    add_panel_label(ax, "c", x=-0.14, y=1.02)


def draw_panel_d(ax, values: dict) -> None:
    style_axes(ax, grid=False)
    labels = ["Minority\nsupport", "Dominant\nsupport", "Singleton"]
    heights = [values["minority"], values["dominant"], values["singleton"]]
    colors = ["#F6C7C0", "#E87A6A", "#B61E1E"]
    x = np.arange(3)
    ax.bar(x, heights, color=colors, edgecolor="#222222", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_ylabel("Collective\nfailure rate")
    ax.set_title("Failure preview", loc="left", fontsize=8.1)
    ax.text(
        0.98,
        0.95,
        f"High-risk singleton: {values['high_risk_singleton']:.3f}\nTetragonal hotspot: {values['tetragonal']:.3f}",
        transform=ax.transAxes,
        fontsize=5.8,
        ha="right",
        va="top",
    )
    add_panel_label(ax, "d", x=-0.14, y=1.02)


def draw_panel_e(ax, metadata: pd.DataFrame, frontier: pd.DataFrame) -> None:
    style_axes(ax, grid=False)
    marker_map = {
        "equivariant_gnn": "o",
        "invariant_gnn": "s",
        "transformer": "^",
        "hybrid_ensemble": "D",
        "non_gnn": "X",
    }
    for architecture, frame in metadata.groupby("architecture_group"):
        ax.scatter(
            frame["proxy_cost"],
            frame["f1_full_test"],
            s=18,
            marker=marker_map[architecture],
            c=frame["training_display"].map(TRAINING_REGIME_COLORS),
            edgecolors="#222222",
            linewidths=0.35,
            alpha=0.95,
        )
    frontier = frontier.sort_values("proxy_cost")
    ax.plot(frontier["proxy_cost"], frontier["f1_full_test"], color="#111111", lw=1.1)
    ax.set_xscale("log")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("Relative cost")
    ax.set_ylabel("F1")
    ax.set_title("Pareto preview", loc="left", fontsize=8.1)
    for model_key, short_label in [("eqnorm-mptrj", "Eqnorm"), ("mattersim-v1-5M", "MatterSim"), ("eSEN-30m-oam", "eSEN")]:
        row = metadata.loc[metadata["model_key"] == model_key].iloc[0]
        ax.annotate(short_label, xy=(row["proxy_cost"], row["f1_full_test"]), xytext=(4, 4), textcoords="offset points", fontsize=5.7, color="#333333")
    add_panel_label(ax, "e", x=-0.14, y=1.02)


def save_single_panel(fig, stem_paths: list[Path]) -> None:
    save_figure_to_many(fig, stem_paths)


def create_preview(output_stems: list[Path], package_dir: Path) -> None:
    fig = create_figure_mm(*PREVIEW_SIZE_MM)
    fig.patch.set_facecolor("white")
    placements = [
        ("panel_a_leaderboard_conflation.png", [0.01, 0.53, 0.37, 0.42]),
        ("panel_b_decomposition_framework.png", [0.41, 0.53, 0.58, 0.42]),
        ("panel_c_scaling_preview.png", [0.01, 0.08, 0.31, 0.28]),
        ("panel_d_failure_preview.png", [0.345, 0.08, 0.31, 0.28]),
        ("panel_e_pareto_preview.png", [0.68, 0.08, 0.31, 0.28]),
    ]
    for file_name, position in placements:
        axis = fig.add_axes(position)
        axis.imshow(plt.imread(package_dir / file_name))
        axis.axis("off")
    save_figure_to_many(fig, output_stems)


def write_assembly_notes(output_dirs: list[Path]) -> None:
    contents = "\n".join(
        [
            "# Figure 1 Assembly Notes",
            "",
            "Panels exported as individual PDF and PNG assets.",
            "",
            "Suggested layout:",
            "- top row left: `panel_a_leaderboard_conflation.pdf`",
            "- top row right: `panel_b_decomposition_framework.pdf`",
            "- bottom row: `panel_c_scaling_preview.pdf`, `panel_d_failure_preview.pdf`, `panel_e_pareto_preview.pdf`",
            "",
            "Guide alignment:",
            "- panels c/d/e use real data micro-plots",
            "- panels a/b are programmatic schematic panels intended as editable assembly assets",
            "",
        ]
    )
    for output_dir in output_dirs:
        ensure_dir(output_dir)
        (output_dir / "assembly_notes.md").write_text(contents, encoding="utf-8")


def main() -> None:
    configure_publication_style()

    results_dir = RESULTS_DIR / "figures" / "figure_01_study_overview_redesign"
    package_dir = (
        PROJECT_ROOT
        / "paper"
        / "submission_package"
        / "figures"
        / "main"
        / "figure_01_study_overview"
    )
    output_dirs = [results_dir, package_dir]
    for output_dir in output_dirs:
        ensure_dir(output_dir)

    leaderboard = load_leaderboard_panel()
    framework = load_framework_panel()
    scaling_points, scaling_fit = load_scaling_preview()
    failure_preview = load_failure_preview()
    pareto_points, pareto_frontier = load_pareto_preview()

    fig = create_figure_mm(*PANEL_A_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_a(ax, leaderboard)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.08)
    save_single_panel(fig, [output_dir / "panel_a_leaderboard_conflation" for output_dir in output_dirs])

    fig = create_figure_mm(*PANEL_B_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_b(ax, framework)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.08)
    save_single_panel(fig, [output_dir / "panel_b_decomposition_framework" for output_dir in output_dirs])

    fig = create_figure_mm(*PANEL_CDE_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_c(ax, scaling_points, scaling_fit)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.88, bottom=0.18)
    save_single_panel(fig, [output_dir / "panel_c_scaling_preview" for output_dir in output_dirs])

    fig = create_figure_mm(*PANEL_CDE_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_d(ax, failure_preview)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.88, bottom=0.18)
    save_single_panel(fig, [output_dir / "panel_d_failure_preview" for output_dir in output_dirs])

    fig = create_figure_mm(*PANEL_CDE_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_e(ax, pareto_points, pareto_frontier)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.88, bottom=0.18)
    save_single_panel(fig, [output_dir / "panel_e_pareto_preview" for output_dir in output_dirs])

    create_preview([output_dir / "figure_01_preview" for output_dir in output_dirs], package_dir)
    write_assembly_notes(output_dirs)


if __name__ == "__main__":
    main()

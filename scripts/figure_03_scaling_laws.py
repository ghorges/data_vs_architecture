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
from scipy.stats import t
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, NullFormatter

from dva_project.figure_style import (
    ARCHITECTURE_MARKERS,
    TRAINING_REGIME_COLORS,
    add_panel_label,
    configure_publication_style,
    create_figure_mm,
    style_axes,
)
from dva_project.settings import PROJECT_ROOT, RESULTS_DIR
from dva_project.utils import ensure_dir


PANEL_SIZE_MM = (58.0, 65.0)
PREVIEW_SIZE_MM = (180.0, 65.0)
PANEL_XLABEL_Y = -0.16
PARAMETER_SUBSET = "MPtrj__OMat24__sAlex"
FAMILY_COLOR_MAP = {
    "eSEN": "#E69F00",
    "nequip": "#56B4E9",
    "grace": "#009E73",
    "allegro": "#D55E00",
    "deepmd": "#CC79A7",
    "orb": "#8C564B",
    "sevennet": "#555555",
}
FAMILY_LABELS = {
    "eSEN": "eSEN",
    "nequip": "Nequip",
    "grace": "GRACE",
    "allegro": "Allegro",
    "deepmd": "DPA",
    "orb": "ORB",
    "sevennet": "SevenNet",
}
FAMILY_LEGEND_ORDER = ["eSEN", "orb", "sevennet", "allegro", "nequip", "deepmd", "grace"]


def format_p_value_text(p_value: float, *, digits: int = 1) -> str:
    p_value = float(p_value)
    if p_value == 0:
        return "p < 10$^{-16}$"
    exponent = int(np.floor(np.log10(abs(p_value))))
    if exponent >= -2:
        return f"p = {p_value:.3f}"
    coefficient = p_value / (10**exponent)
    return f"p = {coefficient:.{digits}f} × 10$^{{{exponent}}}$"


def fit_with_confidence(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> pd.DataFrame:
    log_x = np.log10(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    log_x_grid = np.log10(x_grid)

    slope, intercept = np.polyfit(log_x, y, deg=1)
    fitted = intercept + slope * log_x
    residuals = y - fitted
    n_points = len(log_x)
    if n_points <= 2:
        mean = intercept + slope * log_x_grid
        return pd.DataFrame(
            {
                "x": x_grid,
                "mean": mean,
                "mean_ci_lower": mean,
                "mean_ci_upper": mean,
            }
        )

    x_mean = log_x.mean()
    sxx = np.sum((log_x - x_mean) ** 2)
    residual_std = np.sqrt(np.sum(residuals**2) / (n_points - 2))
    t_crit = t.ppf(0.975, df=n_points - 2)

    mean = intercept + slope * log_x_grid
    mean_se = residual_std * np.sqrt((1.0 / n_points) + ((log_x_grid - x_mean) ** 2 / sxx))
    interval = t_crit * mean_se
    return pd.DataFrame(
        {
            "x": x_grid,
            "mean": mean,
            "mean_ci_lower": mean - interval,
            "mean_ci_upper": mean + interval,
        }
    )


def load_parameter_panel() -> tuple[pd.DataFrame, pd.Series]:
    points = pd.read_csv(RESULTS_DIR / "tables" / "analysis_03" / "parameter_scaling_points.csv")
    fits = pd.read_csv(RESULTS_DIR / "tables" / "analysis_03" / "parameter_scaling_fits.csv")
    panel = points.loc[(points["subset"] == PARAMETER_SUBSET)].copy()
    fit = fits.loc[
        (fits["subset"] == PARAMETER_SUBSET) & (fits["metric"] == "f1_full_test")
    ].iloc[0]
    return panel, fit


def load_data_panel() -> tuple[pd.DataFrame, pd.Series]:
    points = pd.read_csv(RESULTS_DIR / "tables" / "analysis_03" / "data_scaling_points.csv")
    fits = pd.read_csv(RESULTS_DIR / "tables" / "analysis_03" / "data_scaling_fits.csv")
    panel = points.loc[points["family"].isin(FAMILY_COLOR_MAP)].copy()
    fit = fits.loc[
        (fits["analysis"] == "data_scaling")
        & (fits["subset"] == "pooled_family_combo_means")
        & (fits["metric"] == "f1_full_test")
    ].iloc[0]
    return panel, fit


def load_ensemble_panel() -> tuple[pd.DataFrame, dict]:
    curve = pd.read_csv(RESULTS_DIR / "tables" / "analysis_03" / "ensemble_scaling_curve.csv")
    summary = pd.read_json(RESULTS_DIR / "tables" / "analysis_03" / "ensemble_summary.json", typ="series").to_dict()
    return curve, summary


def save_fixed_canvas_to_many(fig, stems: list[Path], *, dpi: int = 300, close: bool = True) -> None:
    for stem in stems:
        ensure_dir(stem.parent)
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
        fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches=None, pad_inches=0)
    if close:
        plt.close(fig)


def draw_panel_a(ax, panel: pd.DataFrame, fit_row: pd.Series) -> None:
    style_axes(ax, grid=False)
    panel = panel.sort_values("model_params").copy()
    x_grid = np.geomspace(panel["model_params"].min(), panel["model_params"].max(), 200)
    confidence = fit_with_confidence(panel["model_params"].to_numpy(), panel["f1_full_test"].to_numpy(), x_grid)

    regime_color = TRAINING_REGIME_COLORS[PARAMETER_SUBSET]
    for row in panel.itertuples(index=False):
        ax.scatter(
            row.model_params,
            row.f1_full_test,
            s=34,
            marker=ARCHITECTURE_MARKERS[row.architecture_group],
            color=regime_color,
            edgecolors="#1A1A1A",
            linewidths=0.45,
            zorder=3,
        )
    ax.fill_between(
        confidence["x"],
        confidence["mean_ci_lower"],
        confidence["mean_ci_upper"],
        color="#D9D9D9",
        alpha=0.09,
        zorder=1,
    )
    ax.plot(confidence["x"], confidence["mean"], color="#111111", linewidth=1.6, zorder=2)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xlabel("Model parameters")
    ax.xaxis.set_label_coords(0.5, PANEL_XLABEL_Y)
    ax.set_ylabel("F1")
    ax.set_ylim(0.75, 0.915)
    ax.text(
        0.03,
        0.97,
        "\n".join(
            [
                f"p = {fit_row['p_value']:.3f}",
                f"Slope = {fit_row['slope_per_log10']:.3f}",
                "Fixed data: OMat24 + sAlex + MPtrj",
            ]
        ),
        transform=ax.transAxes,
        fontsize=6.2,
        ha="left",
        va="top",
        color="#222222",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor=regime_color,
            markeredgecolor="#1A1A1A",
            markersize=5,
            label=label,
        )
        for label, marker in [
            ("Equivariant GNN", ARCHITECTURE_MARKERS["equivariant_gnn"]),
            ("Invariant GNN", ARCHITECTURE_MARKERS["invariant_gnn"]),
            ("Transformer", ARCHITECTURE_MARKERS["transformer"]),
        ]
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.94, 0.08),
        fontsize=5.7,
        handletextpad=0.45,
        labelspacing=0.32,
        borderaxespad=0.0,
        frameon=False,
    )
    add_panel_label(ax, "a", x=-0.11, y=1.03)


def draw_panel_b(ax, panel: pd.DataFrame, fit_row: pd.Series) -> None:
    style_axes(ax, grid=False)
    x_min = float(panel["effective_training_structures"].min())
    x_max = float(panel["effective_training_structures"].max())
    x_grid = np.geomspace(x_min, x_max, 300)
    confidence = fit_with_confidence(
        panel["effective_training_structures"].to_numpy(),
        panel["f1_full_test"].to_numpy(),
        x_grid,
    )

    for family, family_frame in panel.groupby("family"):
        family_frame = family_frame.sort_values("effective_training_structures")
        color = FAMILY_COLOR_MAP[family]
        ax.plot(
            family_frame["effective_training_structures"],
            family_frame["f1_full_test"],
            marker="o",
            markersize=3.6,
            linewidth=1.05,
            color=color,
            zorder=3,
        )

    ax.fill_between(
        confidence["x"],
        confidence["mean_ci_lower"],
        confidence["mean_ci_upper"],
        color="#E0E0E0",
        alpha=0.28,
        zorder=1,
    )
    ax.plot(
        confidence["x"],
        confidence["mean"],
        color="#111111",
        linewidth=1.6,
        linestyle=(0, (4, 2)),
        zorder=2,
    )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(3.0,), numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=2.2, width=0.6)
    ax.set_xlim(x_min * 0.9, x_max * 1.9)
    ax.set_xlabel("Effective training structures")
    ax.xaxis.set_label_coords(0.5, PANEL_XLABEL_Y)
    ax.set_ylabel("F1")
    ax.set_ylim(0.62, 0.915)
    ax.text(
        0.03,
        0.97,
        "\n".join(
            [
                f"R$^2$ = {fit_row['r_squared']:.3f}",
                format_p_value_text(fit_row["p_value"], digits=1),
                f"Pooled slope = {fit_row['slope_per_log10']:.3f}",
            ]
        ),
        transform=ax.transAxes,
        fontsize=6.2,
        ha="left",
        va="top",
        color="#222222",
    )
    legend_handles = [
        Line2D(
            [0, 1],
            [0, 0],
            color=FAMILY_COLOR_MAP[family],
            marker="o",
            markerfacecolor=FAMILY_COLOR_MAP[family],
            markeredgecolor=FAMILY_COLOR_MAP[family],
            markersize=3.8,
            linewidth=1.05,
            label=FAMILY_LABELS[family],
        )
        for family in FAMILY_LEGEND_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        fontsize=5.5,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.75,
        labelspacing=0.25,
        borderaxespad=0.0,
        frameon=False,
    )
    add_panel_label(ax, "b", x=-0.11, y=1.03)


def draw_panel_c(ax, curve: pd.DataFrame, summary: dict) -> None:
    style_axes(ax, grid=False)
    curve = curve.sort_values("ensemble_size").copy()
    ax.axvspan(3, 10, color="#EFEFEF", alpha=0.9, zorder=0)
    ax.plot(curve["ensemble_size"], curve["f1"], color="#0072B2", linewidth=1.8, zorder=2)
    ax.axvline(summary["best_mae_k"], color="#666666", linewidth=1.0, linestyle=(0, (2, 2)), zorder=1)
    ax.axvline(summary["best_f1_k"], color="#D55E00", linewidth=1.2, linestyle=(0, (3, 2)), zorder=1)

    best_f1_row = curve.loc[curve["ensemble_size"] == summary["best_f1_k"]].iloc[0]
    best_mae_row = curve.loc[curve["ensemble_size"] == summary["best_mae_k"]].iloc[0]
    ax.scatter(best_f1_row["ensemble_size"], best_f1_row["f1"], s=22, color="#D55E00", zorder=3)
    ax.scatter(best_mae_row["ensemble_size"], best_mae_row["f1"], s=18, color="#666666", zorder=3)
    ax.text(
        8.2,
        0.9095,
        f"Best F1 = {best_f1_row['f1']:.3f} at k = {int(best_f1_row['ensemble_size'])}",
        fontsize=6.6,
        color="#D55E00",
        ha="left",
        va="bottom",
    )
    ax.text(
        3.45,
        0.881,
        "Best MAE at k = 3",
        fontsize=6.8,
        color="#222222",
        ha="left",
        va="top",
    )
    ax.text(
        7.7,
        0.8055,
        "Saturation\nzone",
        fontsize=6.2,
        color="#555555",
        va="center",
        ha="center",
        linespacing=0.9,
    )

    ax.set_xlim(1, 45)
    ax.set_ylim(0.79, 0.915)
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_xticks(np.arange(5, 46, 5), minor=True)
    ax.tick_params(axis="x", which="minor", length=2.2, width=0.6, labelbottom=False)
    ax.set_xlabel("Ensemble size k")
    ax.xaxis.set_label_coords(0.5, PANEL_XLABEL_Y)
    ax.set_ylabel("F1")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    add_panel_label(ax, "c", x=-0.11, y=1.03)


def create_panel_a(output_stems: list[Path], panel: pd.DataFrame, fit_row: pd.Series) -> None:
    fig = create_figure_mm(*PANEL_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_a(ax, panel, fit_row)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.20)
    save_fixed_canvas_to_many(fig, output_stems)


def create_panel_b(output_stems: list[Path], panel: pd.DataFrame, fit_row: pd.Series) -> None:
    fig = create_figure_mm(*PANEL_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_b(ax, panel, fit_row)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.20)
    save_fixed_canvas_to_many(fig, output_stems)


def create_panel_c(output_stems: list[Path], curve: pd.DataFrame, summary: dict) -> None:
    fig = create_figure_mm(*PANEL_SIZE_MM)
    ax = fig.add_subplot(111)
    draw_panel_c(ax, curve, summary)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.20)
    save_fixed_canvas_to_many(fig, output_stems)


def create_preview(output_stems: list[Path], package_dir: Path) -> None:
    fig = create_figure_mm(*PREVIEW_SIZE_MM)
    fig.patch.set_facecolor("white")
    placements = [
        ("panel_a_parameter_scaling.png", [0.01, 0.07, 0.31, 0.88]),
        ("panel_b_data_scaling.png", [0.345, 0.07, 0.31, 0.88]),
        ("panel_c_ensemble_scaling.png", [0.68, 0.07, 0.31, 0.88]),
    ]
    for file_name, position in placements:
        axis = fig.add_axes(position)
        axis.imshow(plt.imread(package_dir / file_name))
        axis.axis("off")
    save_fixed_canvas_to_many(fig, output_stems)


def write_assembly_notes(output_dirs: list[Path]) -> None:
    contents = "\n".join(
        [
            "# Figure 3 Assembly Notes",
            "",
            "Panels exported as individual PDF and PNG assets.",
            "",
            "Suggested layout:",
            "- `panel_a_parameter_scaling.pdf`",
            "- `panel_b_data_scaling.pdf`",
            "- `panel_c_ensemble_scaling.pdf`",
            "",
            "Suggested assembly:",
            "- place the three panels in one horizontal row",
            "- keep equal panel widths",
            "- use ~5 mm horizontal gutters",
            "",
            "Preview file:",
            "- `figure_03_preview.png` / `figure_03_preview.pdf`",
            "",
            "Data sources:",
            "- `results/tables/analysis_03/parameter_scaling_points.csv`",
            "- `results/tables/analysis_03/parameter_scaling_fits.csv`",
            "- `results/tables/analysis_03/data_scaling_points.csv`",
            "- `results/tables/analysis_03/data_scaling_fits.csv`",
            "- `results/tables/analysis_03/ensemble_scaling_curve.csv`",
            "- `results/tables/analysis_03/ensemble_summary.json`",
            "",
        ]
    )
    for output_dir in output_dirs:
        ensure_dir(output_dir)
        (output_dir / "assembly_notes.md").write_text(contents, encoding="utf-8")


def main() -> None:
    configure_publication_style()

    results_dir = RESULTS_DIR / "figures" / "figure_03_scaling_laws"
    package_dir = (
        PROJECT_ROOT
        / "paper"
        / "submission_package"
        / "figures"
        / "main"
        / "figure_03_scaling_laws"
    )
    output_dirs = [results_dir, package_dir]
    for output_dir in output_dirs:
        ensure_dir(output_dir)

    parameter_panel, parameter_fit = load_parameter_panel()
    data_panel, data_fit = load_data_panel()
    ensemble_curve, ensemble_summary = load_ensemble_panel()

    create_panel_a(
        [output_dir / "panel_a_parameter_scaling" for output_dir in output_dirs],
        parameter_panel,
        parameter_fit,
    )
    create_panel_b(
        [output_dir / "panel_b_data_scaling" for output_dir in output_dirs],
        data_panel,
        data_fit,
    )
    create_panel_c(
        [output_dir / "panel_c_ensemble_scaling" for output_dir in output_dirs],
        ensemble_curve,
        ensemble_summary,
    )
    create_preview(
        [output_dir / "figure_03_preview" for output_dir in output_dirs],
        package_dir,
    )
    write_assembly_notes(output_dirs)


if __name__ == "__main__":
    main()

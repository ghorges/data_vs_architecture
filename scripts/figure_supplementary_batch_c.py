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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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


SUPP_PACKAGE_DIR = PROJECT_ROOT / "paper" / "submission_package" / "figures" / "supplementary"
SUPP_RESULTS_DIR = RESULTS_DIR / "figures"
METRIC_ORDER = ["F1", "MAE", "DAF", "R2"]


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


def plot_permutation_panel(ax, observed: pd.DataFrame, draws: pd.DataFrame, metric_label: str, summary: dict) -> None:
    style_axes(ax, grid=False)
    fill_colors = {"snapshot_45": "#BDBDBD", "live_53": "#56B4E9"}
    edge_colors = {"snapshot_45": "#6F6F6F", "live_53": "white"}
    line_colors = {"snapshot_45": "#1F4E79", "live_53": "#56B4E9"}
    line_styles = {"snapshot_45": "-", "live_53": (0, (2.0, 1.2))}
    metric_key = {"F1": "f1_full_test", "MAE": "mae_full_test", "DAF": "daf_full_test", "R2": "r2_full_test"}[metric_label]
    observed_lookup: dict[str, float] = {}
    for scope in ["snapshot_45", "live_53"]:
        vals = draws.loc[(draws["scope"] == scope) & (draws["metric_label"] == metric_label), "train_minus_arch"]
        obs = observed.loc[(observed["scope"] == scope) & (observed["metric_label"] == metric_label), "train_minus_arch"].iloc[0]
        observed_lookup[scope] = float(obs)
        ax.hist(
            vals,
            bins=32,
            density=True,
            color=fill_colors[scope],
            alpha=0.62 if scope == "snapshot_45" else 0.42,
            edgecolor=edge_colors[scope],
            linewidth=0.55 if scope == "snapshot_45" else 0.3,
            zorder=2 if scope == "snapshot_45" else 3,
        )
        ax.axvline(
            obs,
            color=line_colors[scope],
            lw=1.6,
            ls=line_styles[scope],
            dash_capstyle="butt" if scope == "live_53" else "round",
            zorder=4,
        )
    p_snapshot = summary["anova"]["snapshot_45"][metric_key]["train_minus_arch_p_one_sided_ge"]
    p_live = summary["anova"]["live_53"][metric_key]["train_minus_arch_p_one_sided_ge"]
    ax.set_xlabel("Training η² − Architecture η²")
    ax.set_ylabel("Density")
    live_obs = observed_lookup["live_53"]
    text_transform = ax.get_xaxis_transform()
    text_kwargs = {
        "transform": text_transform,
        "va": "top",
        "fontsize": 5.5,
        "color": "#444444",
    }
    ax.text(live_obs, 0.86, "p(Frozen) =", ha="right", **text_kwargs)
    ax.text(live_obs, 0.86, f" {p_snapshot:.3f}", ha="left", **text_kwargs)
    ax.text(live_obs, 0.79, "p(Live) =", ha="right", **text_kwargs)
    ax.text(live_obs, 0.79, f" {p_live:.3f}", ha="left", **text_kwargs)


def s06_legend_handles() -> list:
    return [
        Patch(facecolor="#BDBDBD", edgecolor="#6F6F6F", linewidth=0.55, alpha=0.62, label="Frozen null"),
        Patch(facecolor="#56B4E9", edgecolor="white", linewidth=0.3, alpha=0.42, label="Live null"),
        Line2D([0], [0], color="#1F4E79", lw=1.6, ls="-", solid_capstyle="round", label="Frozen obs"),
        Line2D([0], [0], color="#56B4E9", lw=1.6, ls=(0, (2.0, 1.2)), dash_capstyle="butt", label="Live obs"),
    ]


def make_s06_permutation_nulls() -> None:
    destinations, package_dir = output_dirs("figure_s06_permutation_nulls")
    observed = pd.read_csv(RESULTS_DIR / "tables" / "analysis_24" / "anova_observed.csv")
    draws = pd.read_csv(RESULTS_DIR / "tables" / "analysis_24" / "anova_permutation_draws.csv")
    summary = json.loads((RESULTS_DIR / "tables" / "analysis_24" / "summary.json").read_text(encoding="utf-8"))

    stem_map = {
        "F1": "panel_a_f1_permutation_null",
        "MAE": "panel_b_mae_permutation_null",
        "DAF": "panel_c_daf_permutation_null",
        "R2": "panel_d_r2_permutation_null",
    }
    for idx, metric_label in enumerate(METRIC_ORDER):
        fig = create_figure_mm(58.0, 54.0)
        ax = fig.add_subplot(111)
        plot_permutation_panel(ax, observed, draws, metric_label, summary)
        add_panel_label(ax, chr(ord("a") + idx), x=-0.18, y=1.02)
        ax.legend(
            handles=s06_legend_handles(),
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            borderaxespad=0.0,
            fontsize=4.9,
            ncol=1,
            frameon=False,
            columnspacing=0.6,
            handlelength=2.25,
            handletextpad=0.34,
            labelspacing=0.25,
        )
        fig.subplots_adjust(left=0.14, right=0.995, top=0.92, bottom=0.18)
        save_panel(fig, destinations, stem_map[metric_label])

    compose_preview(
        package_dir,
        destinations,
        "figure_s06_preview",
        [
            ("panel_a_f1_permutation_null.png", [0.00, 0.52, 0.505, 0.45]),
            ("panel_b_mae_permutation_null.png", [0.495, 0.52, 0.505, 0.45]),
            ("panel_c_daf_permutation_null.png", [0.00, 0.02, 0.505, 0.45]),
            ("panel_d_r2_permutation_null.png", [0.495, 0.02, 0.505, 0.45]),
        ],
        (180.0, 120.0),
    )
    write_notes(
        destinations,
        "Figure S6 Assembly Notes",
        [
            "Four panel supplementary figure.",
            "Each panel overlays frozen and live permutation-null distributions for train-minus-architecture effect size.",
            "Vertical lines mark the observed effect for the corresponding scope.",
        ],
    )


def make_s07_family_resampling() -> None:
    destinations, package_dir = output_dirs("figure_s07_family_resampling")
    anova_summary = pd.read_csv(RESULTS_DIR / "tables" / "analysis_07" / "one_per_family_resampling_anova_summary.csv")
    cluster_draws = pd.read_csv(RESULTS_DIR / "tables" / "analysis_07" / "one_per_family_resampling_cluster.csv")
    summary = json.loads((RESULTS_DIR / "tables" / "analysis_07" / "summary.json").read_text(encoding="utf-8"))
    observed = pd.read_csv(RESULTS_DIR / "tables" / "analysis_24" / "anova_observed.csv")
    observed = observed.loc[observed["scope"] == "snapshot_45"].copy()

    panel_size_mm = (92.0, 68.0)

    fig = create_figure_mm(*panel_size_mm)
    ax = fig.add_subplot(111)
    style_axes(ax, grid=False)
    x = np.arange(len(METRIC_ORDER))
    for factor, color, offset, mean_col, q05_col, q95_col, legend_label in [
        ("Training data", FACTOR_COLORS["Training data"], -0.12, "train_eta_sq_median", "train_eta_sq_q05", "train_eta_sq_q95", "Training data η²"),
        ("Architecture", FACTOR_COLORS["Architecture"], 0.12, "arch_eta_sq_median", "arch_eta_sq_q05", "arch_eta_sq_q95", "Architecture η²"),
    ]:
        subset = anova_summary.copy()
        subset["metric_label"] = pd.Categorical(subset["metric_label"], categories=METRIC_ORDER, ordered=True)
        subset = subset.sort_values("metric_label")
        xpos = x + offset
        yvals = subset[mean_col].to_numpy()
        lower = yvals - subset[q05_col].to_numpy()
        upper = subset[q95_col].to_numpy() - yvals
        ax.errorbar(xpos, yvals, yerr=[lower, upper], fmt="o", color=color, capsize=2.5, lw=1.0, ms=4, label=legend_label)
        factor_obs = observed.loc[(observed["metric_label"].isin(METRIC_ORDER)), ["metric_label", "train_eta_sq", "arch_eta_sq"]].copy()
        factor_obs["metric_label"] = pd.Categorical(factor_obs["metric_label"], categories=METRIC_ORDER, ordered=True)
        factor_obs = factor_obs.sort_values("metric_label")
        obs_values = factor_obs["train_eta_sq"].to_numpy() if factor == "Training data" else factor_obs["arch_eta_sq"].to_numpy()
        ax.scatter(xpos, obs_values, s=20, marker="D", color="white", edgecolor=color, linewidth=1.0, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_ORDER)
    ax.set_ylabel("Partial η²")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower left", fontsize=5.7)
    add_panel_label(ax, "a", x=-0.12, y=1.02)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.92, bottom=0.16)
    save_panel(fig, destinations, "panel_a_anova_resampling")

    fig = create_figure_mm(*panel_size_mm)
    ax = fig.add_subplot(111)
    style_axes(ax, grid=False)
    ax.hist(cluster_draws["ari_train_minus_arch"], bins=28, color="#7FB3D5", alpha=0.75, edgecolor="white", linewidth=0.4)
    observed_delta = summary["full_cluster"]["ari_train_minus_arch"]
    q05 = float(cluster_draws["ari_train_minus_arch"].quantile(0.05))
    q95 = float(cluster_draws["ari_train_minus_arch"].quantile(0.95))
    ax.axvline(observed_delta, color="#0B486B", lw=1.8)
    ax.set_xlabel("ARI training - architecture")
    ax.set_ylabel("Count")
    ax.text(
        0.97,
        0.95,
        f"Observed = {observed_delta:.3f}\n90% interval: [{q05:.3f}, {q95:.3f}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.9,
        color="#444444",
    )
    add_panel_label(ax, "b", x=-0.16, y=1.02)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.92, bottom=0.16)
    save_panel(fig, destinations, "panel_b_cluster_resampling")

    compose_preview(
        package_dir,
        destinations,
        "figure_s07_preview",
        [
            ("panel_a_anova_resampling.png", [0.00, 0.06, 0.48, 0.88]),
            ("panel_b_cluster_resampling.png", [0.52, 0.06, 0.48, 0.88]),
        ],
        (180.0, 76.0),
    )
    write_notes(
        destinations,
        "Figure S7 Assembly Notes",
        [
            "Two-panel supplementary figure.",
            "Diamonds in panel a mark the full frozen snapshot; circles and intervals summarize one-per-family resampling.",
            "Panel b reports the persistence of train-versus-architecture clustering separation under family-aware resampling.",
        ],
    )


def make_s08_bootstrap_uncertainty() -> None:
    destinations, package_dir = output_dirs("figure_s08_bootstrap_uncertainty")
    summary = pd.read_csv(RESULTS_DIR / "tables" / "analysis_08" / "scaling_uncertainty_summary.csv")
    draws = pd.read_csv(RESULTS_DIR / "tables" / "analysis_08" / "scaling_bootstrap_distribution.csv")

    panel_size_mm = (92.0, 72.0)

    fig = create_figure_mm(*panel_size_mm)
    ax = fig.add_subplot(111)
    style_axes(ax, grid=False)
    metric_positions = np.arange(len(METRIC_ORDER))
    x_limits = (-0.03, 0.55)
    specs = [
        ("data_scaling", "pooled_family_combo_means", FACTOR_COLORS["Training data"], -0.10, "Data scaling"),
        ("parameter_scaling", "MPtrj", FACTOR_COLORS["Parameters"], 0.10, "Parameter scaling"),
    ]
    daf_truncated = False
    for analysis_key, subset_key, color, offset, label in specs:
        subset = summary.loc[(summary["analysis"] == analysis_key) & (summary["subset"] == subset_key)].copy()
        subset["metric_label"] = pd.Categorical(subset["metric_label"], categories=METRIC_ORDER, ordered=True)
        subset = subset.sort_values("metric_label")
        ypos = metric_positions + offset
        center = subset["observed_slope_per_log10"].to_numpy()
        lower = subset["slope_per_log10_q025"].to_numpy()
        upper = subset["slope_per_log10_q975"].to_numpy()
        left = center - lower
        right = upper - center
        ax.errorbar(center, ypos, xerr=[left, right], fmt="o", ms=4, lw=1.1, capsize=2.5, color=color, label=label)
        for metric_name, yval, center_val, lower_val, upper_val in zip(METRIC_ORDER, ypos, center, lower, upper):
            if upper_val > x_limits[1]:
                arrow_start = max(center_val, x_limits[1] - 0.055)
                ax.annotate(
                    "",
                    xy=(x_limits[1], yval),
                    xytext=(arrow_start, yval),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, shrinkA=0, shrinkB=0),
                    zorder=4,
                )
                if metric_name == "DAF":
                    daf_truncated = True
    ax.axvline(0.0, color="#777777", lw=0.9, ls=(0, (3, 2)))
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_yticks(metric_positions)
    ax.set_yticklabels(METRIC_ORDER)
    ax.set_xlim(*x_limits)
    ax.set_xlabel(r"Slope per log$_{10}$ decade")
    if daf_truncated:
        ax.text(
            0.98,
            0.78,
            "DAF 95% CI truncated",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.3,
            color="#555555",
        )
    ax.legend(loc="lower right", fontsize=5.8)
    add_panel_label(ax, "a", x=-0.12, y=1.02)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.92, bottom=0.16)
    save_panel(fig, destinations, "panel_a_scaling_slope_intervals")

    fig = create_figure_mm(*panel_size_mm)
    ax = fig.add_subplot(111)
    style_axes(ax, grid=False)
    violin_specs = [
        ("parameter_scaling_bootstrap", "MPtrj", "Parameter: MPtrj", FACTOR_COLORS["Parameters"]),
        ("parameter_scaling_bootstrap", "MPtrj__OMat24__sAlex", "Parameter: OAM mix", "#4C78A8"),
        ("data_scaling_bootstrap", "pooled_family_combo_means", "Data scaling", FACTOR_COLORS["Training data"]),
    ]
    values = []
    colors = []
    labels = []
    observed_values = []
    for analysis_key, subset_key, label, color in violin_specs:
        subset = draws.loc[
            (draws["analysis"] == analysis_key)
            & (draws["subset"] == subset_key)
            & (draws["metric_label"] == "F1"),
            "slope_per_log10",
        ]
        values.append(subset.to_numpy())
        colors.append(color)
        labels.append(label)
        summary_analysis = "data_scaling" if analysis_key.startswith("data_scaling") else "parameter_scaling"
        observed_row = summary.loc[
            (summary["analysis"] == summary_analysis)
            & (summary["subset"] == subset_key)
            & (summary["metric_label"] == "F1")
        ].iloc[0]
        observed_values.append(float(observed_row["observed_slope_per_log10"]))
    parts = ax.violinplot(values, positions=np.arange(len(values)), widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
    for xpos, vals, color, observed_value in zip(np.arange(len(values)), values, colors, observed_values):
        q25, median, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
        ax.vlines(xpos, q25, q75, color="#222222", lw=1.0)
        ax.hlines(median, xpos - 0.13, xpos + 0.13, color="#222222", lw=1.3)
        ax.scatter(xpos, observed_value, s=26, color=color, edgecolor="#222222", linewidth=0.5, zorder=4)
    ax.axhline(0.0, color="#777777", lw=0.9, ls=(0, (3, 2)))
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, fontsize=6.9)
    ax.set_ylabel(r"ΔF1 per log$_{10}$ decade")
    add_panel_label(ax, "b", x=-0.12, y=1.02)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.92, bottom=0.24)
    save_panel(fig, destinations, "panel_b_f1_bootstrap_distributions")

    compose_preview(
        package_dir,
        destinations,
        "figure_s08_preview",
        [
            ("panel_a_scaling_slope_intervals.png", [0.00, 0.06, 0.48, 0.88]),
            ("panel_b_f1_bootstrap_distributions.png", [0.52, 0.06, 0.48, 0.88]),
        ],
        (180.0, 80.0),
    )
    write_notes(
        destinations,
        "Figure S8 Assembly Notes",
        [
            "Two-panel supplementary figure.",
            "Panel a shows observed scaling slopes with bootstrap 95% intervals.",
            "Panel b focuses on the F1 slope distributions for the main parameter-scaling and data-scaling settings.",
        ],
    )


def main() -> None:
    configure_publication_style()
    make_s06_permutation_nulls()
    make_s07_family_resampling()
    make_s08_bootstrap_uncertainty()


if __name__ == "__main__":
    main()

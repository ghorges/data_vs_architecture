from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


SNAPSHOT_LABEL = "snapshot_45"
MAIN_FIGURES = [
    "Figure 1\nOverview",
    "Figure 2\nVariance",
    "Figure 3\nClustering",
    "Figure 4\nScaling",
    "Figure 5\nFailures",
    "Figure 6\nPareto",
]
SUPPLEMENTARY_FIGURES = [
    "Figure S1\nPair controls",
    "Figure S2\nFailure radar",
    "Figure S3\nDecision tree",
    "Figure S4\nSensitivity",
]
ARCHITECTURE_COLORS = {
    "invariant_gnn": "#3465a4",
    "equivariant_gnn": "#0b6e4f",
    "non_gnn": "#c17c00",
    "hybrid_ensemble": "#6c4f8b",
    "transformer": "#b83b5e",
}
MODULE_COLORS = {
    "analysis_01": "#dff3ea",
    "analysis_02": "#e6eef9",
    "analysis_03": "#fff1dd",
    "analysis_04": "#f9e4e8",
    "analysis_05": "#efe8fb",
}


def draw_round_box(ax, xy, width, height, facecolor, edgecolor="#1f2937", linewidth=1.5, radius=0.025):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    return box


def draw_label_value(ax, x, y, label, value, color):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, color="#334155", va="center")
    pill = FancyBboxPatch(
        (x + 0.19, y - 0.018),
        0.07,
        0.036,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=0.0,
        facecolor=color,
        transform=ax.transAxes,
    )
    ax.add_patch(pill)
    ax.text(
        x + 0.225,
        y,
        str(value),
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        weight="bold",
        ha="center",
        va="center",
    )


def draw_bar_list(ax, title, x, y, width, rows, color_map):
    ax.text(x, y, title, transform=ax.transAxes, fontsize=10, weight="bold", color="#111827")
    max_value = max(value for _, value in rows)
    row_y = y - 0.04
    for label, value in rows:
        color = color_map.get(label, "#6b7280")
        bar_width = width * (value / max_value)
        ax.text(x, row_y, label.replace("_", " "), transform=ax.transAxes, fontsize=8, color="#374151", va="center")
        ax.add_patch(
            Rectangle(
                (x + 0.17, row_y - 0.012),
                width,
                0.02,
                transform=ax.transAxes,
                facecolor="#e5e7eb",
                edgecolor="none",
            )
        )
        ax.add_patch(
            Rectangle(
                (x + 0.17, row_y - 0.012),
                bar_width,
                0.02,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(
            x + 0.17 + width + 0.015,
            row_y,
            str(value),
            transform=ax.transAxes,
            fontsize=8,
            color="#111827",
            va="center",
            ha="left",
        )
        row_y -= 0.036


def module_box(ax, x, y, width, height, title, summary, fill):
    draw_round_box(ax, (x, y), width, height, facecolor=fill)
    ax.text(x + 0.02, y + height - 0.042, title, transform=ax.transAxes, fontsize=10.8, weight="bold", color="#111827")
    ax.text(
        x + 0.02,
        y + 0.028,
        fill_text(summary, 48),
        transform=ax.transAxes,
        fontsize=8.2,
        color="#1f2937",
        va="bottom",
    )


def add_arrow(ax, start, end, color="#64748b"):
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.5,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fill_text(value: str, width: int) -> str:
    return fill(value, width=width)


def main() -> None:
    output_figure_dir = RESULTS_DIR / "figures" / "figure_01_overview"
    output_table_dir = RESULTS_DIR / "tables" / "figure_01"
    ensure_dir(output_figure_dir)
    ensure_dir(output_table_dir)

    metadata_snapshot = pd.read_csv(PROCESSED_DIR / f"model_metadata_{SNAPSHOT_LABEL}.csv")
    metadata_full = pd.read_csv(PROCESSED_DIR / "model_metadata.csv")
    sensitivity_summary = load_json(RESULTS_DIR / "tables" / "sensitivity" / "summary.json")
    ensemble_summary = load_json(RESULTS_DIR / "tables" / "analysis_03" / "ensemble_summary.json")
    failure_summary = load_json(RESULTS_DIR / "tables" / "analysis_04" / "summary.json")
    pareto_summary = load_json(RESULTS_DIR / "tables" / "analysis_05" / "summary.json")

    architecture_counts = (
        metadata_snapshot["architecture_group"].value_counts().rename_axis("label").reset_index(name="count")
    )
    training_counts = (
        metadata_snapshot["training_combo"].value_counts().rename_axis("label").reset_index(name="count")
    )
    top_training = training_counts.head(3).copy()
    other_training_count = int(training_counts.iloc[3:]["count"].sum())

    overview_summary = {
        "snapshot_models": int(len(metadata_snapshot)),
        "live_public_models": int(len(metadata_full)),
        "materials": 256963,
        "prediction_matrix_shape": [256963, int(len(metadata_snapshot))],
        "architecture_counts": architecture_counts.to_dict(orient="records"),
        "top_training_combos": top_training.to_dict(orient="records"),
        "other_training_combo_models": other_training_count,
        "analysis_callouts": {
            "analysis_01_f1_train_eta_sq": sensitivity_summary["anova_exact"]["snapshot_45"]["F1"]["Training data"],
            "analysis_01_f1_arch_eta_sq": sensitivity_summary["anova_exact"]["snapshot_45"]["F1"]["Architecture"],
            "analysis_02_ari_training_exact": sensitivity_summary["error_clustering"]["snapshot_45"]["ari_training_exact"],
            "analysis_03_best_f1": ensemble_summary["best_f1"],
            "analysis_03_best_f1_k": ensemble_summary["best_f1_k"],
            "analysis_04_collective_failures": failure_summary["collective_failure_total"],
            "analysis_05_best_high_budget": pareto_summary["best_high_budget_model"],
        },
    }
    (output_table_dir / "overview_summary.json").write_text(
        json.dumps(overview_summary, indent=2),
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
        }
    )
    fig, ax = plt.subplots(figsize=(17, 10))
    fig.patch.set_facecolor("#f7f5ef")
    ax.set_facecolor("#f7f5ef")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.95, "Figure 1. Study overview", fontsize=20, weight="bold", color="#111827", transform=ax.transAxes)
    ax.text(
        0.05,
        0.915,
        "Frozen benchmark scope, shared evaluation substrate, and the five analysis modules used in the manuscript.",
        fontsize=10.5,
        color="#475569",
        transform=ax.transAxes,
    )

    left_x, left_y, left_w, left_h = 0.05, 0.23, 0.28, 0.61
    center_x, center_y, center_w, center_h = 0.37, 0.36, 0.22, 0.40
    right_x, right_y, right_w, right_h = 0.63, 0.18, 0.31, 0.68
    bottom_x, bottom_y, bottom_w, bottom_h = 0.05, 0.035, 0.89, 0.14

    draw_round_box(ax, (left_x, left_y), left_w, left_h, facecolor="#fdfdfd")
    draw_round_box(ax, (center_x, center_y), center_w, center_h, facecolor="#fdfdfd")
    draw_round_box(ax, (right_x, right_y), right_w, right_h, facecolor="#fdfdfd")
    draw_round_box(ax, (bottom_x, bottom_y), bottom_w, bottom_h, facecolor="#fffdf7")

    ax.text(left_x + 0.02, left_y + left_h - 0.05, "Model universe", transform=ax.transAxes, fontsize=13, weight="bold", color="#111827")
    draw_label_value(ax, left_x + 0.02, left_y + left_h - 0.10, "Frozen discovery models", len(metadata_snapshot), "#0b6e4f")
    draw_label_value(ax, left_x + 0.02, left_y + left_h - 0.145, "Live public source state", len(metadata_full), "#6b7280")
    ax.text(
        left_x + 0.02,
        left_y + left_h - 0.19,
        "The manuscript freezes 45 models for reproducibility,\nthen checks robustness against the live 53-model state.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#475569",
        va="top",
    )

    arch_rows = [(row.label, int(row.count)) for row in architecture_counts.itertuples(index=False)]
    draw_bar_list(ax, "Architecture groups", left_x + 0.02, left_y + 0.31, 0.10, arch_rows, ARCHITECTURE_COLORS)

    training_rows = [(row.label, int(row.count)) for row in top_training.itertuples(index=False)]
    training_rows.append(("Other combos", other_training_count))
    draw_bar_list(
        ax,
        "Training-data combos",
        left_x + 0.02,
        left_y + 0.10,
        0.10,
        training_rows,
        {
            "MPtrj": "#0b6e4f",
            "MPtrj__OMat24__sAlex": "#3465a4",
            "MP 2022": "#b83b5e",
            "OpenLAM": "#c17c00",
            "Other combos": "#6b7280",
        },
    )

    ax.text(center_x + 0.02, center_y + center_h - 0.05, "Benchmark substrate", transform=ax.transAxes, fontsize=13, weight="bold", color="#111827")
    ax.text(
        center_x + 0.02,
        center_y + center_h - 0.11,
        "Shared WBM discovery test set with MP2020-corrected\nreference energies and hull labels.",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
        va="top",
    )

    matrix_box = FancyBboxPatch(
        (center_x + 0.035, center_y + 0.10),
        0.15,
        0.17,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#475569",
        facecolor="#eef4ff",
        transform=ax.transAxes,
    )
    ax.add_patch(matrix_box)
    for offset in [0.0, 0.03, 0.06, 0.09, 0.12]:
        ax.add_line(plt.Line2D([center_x + 0.045 + offset, center_x + 0.045 + offset], [center_y + 0.11, center_y + 0.26], color="#cbd5e1", linewidth=0.8, transform=ax.transAxes))
    for offset in [0.0, 0.03, 0.06, 0.09, 0.12]:
        ax.add_line(plt.Line2D([center_x + 0.045, center_x + 0.165], [center_y + 0.12 + offset, center_y + 0.12 + offset], color="#cbd5e1", linewidth=0.8, transform=ax.transAxes))
    ax.text(center_x + 0.11, center_y + 0.29, "Prediction matrix", transform=ax.transAxes, fontsize=9, color="#334155", ha="center")
    ax.text(center_x + 0.11, center_y + 0.22, "256,963 x 45", transform=ax.transAxes, fontsize=16, weight="bold", color="#111827", ha="center")
    ax.text(center_x + 0.11, center_y + 0.16, "materials x models", transform=ax.transAxes, fontsize=8.5, color="#475569", ha="center")
    ax.text(center_x + 0.11, center_y + 0.08, "formation-energy predictions\n+ shared DFT references", transform=ax.transAxes, fontsize=8.5, color="#475569", ha="center")

    ax.text(right_x + 0.02, right_y + right_h - 0.05, "Five analysis modules", transform=ax.transAxes, fontsize=13, weight="bold", color="#111827")

    module_box(
        ax,
        right_x + 0.02,
        right_y + 0.54,
        0.27,
        0.102,
        "1. Variance decomposition",
        f"F1 exact ANOVA: training data eta^2 = {overview_summary['analysis_callouts']['analysis_01_f1_train_eta_sq']:.3f}, "
        f"architecture eta^2 = {overview_summary['analysis_callouts']['analysis_01_f1_arch_eta_sq']:.3f}.",
        MODULE_COLORS["analysis_01"],
    )
    module_box(
        ax,
        right_x + 0.02,
        right_y + 0.41,
        0.27,
        0.102,
        "2. Error correlation",
        f"Training provenance organizes the error manifold: ARI = {overview_summary['analysis_callouts']['analysis_02_ari_training_exact']:.3f}.",
        MODULE_COLORS["analysis_02"],
    )
    module_box(
        ax,
        right_x + 0.02,
        right_y + 0.28,
        0.27,
        0.102,
        "3. Scaling laws",
        f"Best prefix ensemble reaches F1 = {overview_summary['analysis_callouts']['analysis_03_best_f1']:.4f} at k = {overview_summary['analysis_callouts']['analysis_03_best_f1_k']}.",
        MODULE_COLORS["analysis_03"],
    )
    module_box(
        ax,
        right_x + 0.02,
        right_y + 0.15,
        0.27,
        0.102,
        "4. Collective failures",
        f"{overview_summary['analysis_callouts']['analysis_04_collective_failures']:,} materials are collective failures at the 80% vote threshold.",
        MODULE_COLORS["analysis_04"],
    )
    module_box(
        ax,
        right_x + 0.02,
        right_y + 0.02,
        0.27,
        0.102,
        "5. Resource allocation",
        f"High-budget recommendation: {overview_summary['analysis_callouts']['analysis_05_best_high_budget']}.",
        MODULE_COLORS["analysis_05"],
    )

    ax.text(bottom_x + 0.02, bottom_y + bottom_h - 0.04, "Paper outputs", transform=ax.transAxes, fontsize=13, weight="bold", color="#111827")
    ax.text(bottom_x + 0.02, bottom_y + bottom_h - 0.07, "Main-text figures", transform=ax.transAxes, fontsize=9.5, weight="bold", color="#334155")
    ax.text(bottom_x + 0.39, bottom_y + bottom_h - 0.07, "Supplementary figures", transform=ax.transAxes, fontsize=9.5, weight="bold", color="#334155")

    for index, label in enumerate(MAIN_FIGURES):
        x = bottom_x + 0.02 + (index % 3) * 0.17
        y = bottom_y + 0.03 - (index // 3) * 0.045
        draw_round_box(ax, (x, y), 0.145, 0.038, facecolor="#ecfdf5", edgecolor="#9ca3af", linewidth=1.0, radius=0.015)
        ax.text(x + 0.0725, y + 0.019, label, transform=ax.transAxes, fontsize=7.3, color="#14532d", va="center", ha="center")

    for index, label in enumerate(SUPPLEMENTARY_FIGURES):
        x = bottom_x + 0.39 + (index % 2) * 0.19
        y = bottom_y + 0.03 - (index // 2) * 0.045
        draw_round_box(ax, (x, y), 0.165, 0.038, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.0, radius=0.015)
        ax.text(x + 0.0825, y + 0.019, label, transform=ax.transAxes, fontsize=7.3, color="#334155", va="center", ha="center")

    draw_round_box(ax, (bottom_x + 0.78, bottom_y + 0.03), 0.13, 0.07, facecolor="#fff1dd", edgecolor="#d6b980", linewidth=1.0, radius=0.02)
    ax.text(bottom_x + 0.845, bottom_y + 0.076, "Manuscript", transform=ax.transAxes, fontsize=10, weight="bold", color="#7c4a03", ha="center")
    ax.text(bottom_x + 0.845, bottom_y + 0.048, "draft + tables + refs", transform=ax.transAxes, fontsize=8, color="#92400e", ha="center")

    add_arrow(ax, (left_x + left_w, left_y + 0.33), (center_x, center_y + 0.24))
    add_arrow(ax, (center_x + center_w, center_y + 0.21), (right_x, right_y + 0.46))
    add_arrow(ax, (right_x + 0.15, right_y), (bottom_x + 0.55, bottom_y + bottom_h))
    add_arrow(ax, (center_x + 0.11, center_y), (bottom_x + 0.18, bottom_y + bottom_h))

    fig.savefig(output_figure_dir / "study_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

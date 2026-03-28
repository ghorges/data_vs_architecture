from __future__ import annotations

import json
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from sklearn.metrics import adjusted_rand_score

from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


SNAPSHOT_LABEL = "snapshot_45"


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def make_color_map(values: pd.Series, palette_name: str) -> tuple[dict, pd.Series]:
    unique_values = sorted(values.unique())
    palette = sns.color_palette(palette_name, n_colors=len(unique_values))
    mapping = dict(zip(unique_values, palette))
    return mapping, values.map(mapping)


def build_pairwise_summary(corr: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    lookup = metadata.set_index("model_key")
    rows: list[dict] = []
    for left, right in combinations(corr.index.tolist(), 2):
        rows.append(
            {
                "model_a": left,
                "model_b": right,
                "correlation": float(corr.loc[left, right]),
                "same_training_combo": lookup.loc[left, "training_combo"] == lookup.loc[right, "training_combo"],
                "same_training_group": lookup.loc[left, "training_group"] == lookup.loc[right, "training_group"],
                "same_architecture_group": lookup.loc[left, "architecture_group"]
                == lookup.loc[right, "architecture_group"],
            }
        )
    return pd.DataFrame(rows)


def summarize_pairwise(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for label in ["training_combo", "training_group", "architecture_group"]:
        same_column = f"same_{label}"
        rows.append(
            {
                "label_type": label,
                "same_label": True,
                "mean_correlation": pairwise.loc[pairwise[same_column], "correlation"].mean(),
                "median_correlation": pairwise.loc[pairwise[same_column], "correlation"].median(),
                "n_pairs": int(pairwise[same_column].sum()),
            }
        )
        rows.append(
            {
                "label_type": label,
                "same_label": False,
                "mean_correlation": pairwise.loc[~pairwise[same_column], "correlation"].mean(),
                "median_correlation": pairwise.loc[~pairwise[same_column], "correlation"].median(),
                "n_pairs": int((~pairwise[same_column]).sum()),
            }
        )
    return pd.DataFrame(rows)


def create_clustermap(
    corr: pd.DataFrame,
    linkage_matrix: np.ndarray,
    metadata: pd.DataFrame,
    output_path,
) -> None:
    sns.set_theme(style="white")
    training_palette, training_colors = make_color_map(metadata["training_combo"], "tab20")
    arch_palette, arch_colors = make_color_map(metadata["architecture_group"], "Set2")
    row_colors = pd.DataFrame(
        {
            "Training data": training_colors,
            "Architecture": arch_colors,
        },
        index=metadata["model_key"],
    )

    grid = sns.clustermap(
        corr,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        row_colors=row_colors,
        col_colors=row_colors,
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        figsize=(16, 16),
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.12, 0.12),
        colors_ratio=(0.03, 0.03),
        cbar_kws={"label": "Pearson correlation of per-material errors"},
    )
    plt.setp(grid.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(grid.ax_heatmap.get_yticklabels(), fontsize=7)

    training_handles = [
        Patch(facecolor=color, edgecolor="none", label=value)
        for value, color in training_palette.items()
    ]
    arch_handles = [
        Patch(facecolor=color, edgecolor="none", label=value)
        for value, color in arch_palette.items()
    ]
    training_legend = grid.ax_heatmap.legend(
        handles=training_handles,
        title="Training data",
        bbox_to_anchor=(1.25, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    grid.ax_heatmap.add_artist(training_legend)
    grid.ax_heatmap.legend(
        handles=arch_handles,
        title="Architecture",
        bbox_to_anchor=(1.25, 0.55),
        loc="upper left",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    ensure_dir(output_path.parent)
    grid.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def main() -> None:
    output_table_dir = RESULTS_DIR / "tables" / "analysis_02"
    output_figure_dir = RESULTS_DIR / "figures" / "analysis_02"
    ensure_dir(output_table_dir)
    ensure_dir(output_figure_dir)

    metadata = pd.read_csv(PROCESSED_DIR / f"model_metadata_{SNAPSHOT_LABEL}.csv")
    errors = pd.read_parquet(PROCESSED_DIR / f"discovery_error_matrix_{SNAPSHOT_LABEL}.parquet")

    model_order = metadata["model_key"].tolist()
    error_matrix = errors[model_order].copy()
    corr = error_matrix.corr()

    filled = error_matrix.apply(lambda col: col.fillna(col.mean()), axis=0)
    model_vectors = zscore_rows(filled.to_numpy(dtype=float).T)
    linkage_matrix = linkage(model_vectors, method="ward")
    leaf_order = leaves_list(linkage_matrix)
    ordered_models = [model_order[idx] for idx in leaf_order]

    clusters_training_exact = fcluster(
        linkage_matrix,
        t=metadata["training_combo"].nunique(),
        criterion="maxclust",
    )
    clusters_training_coarse = fcluster(
        linkage_matrix,
        t=metadata["training_group"].nunique(),
        criterion="maxclust",
    )
    clusters_architecture = fcluster(
        linkage_matrix,
        t=metadata["architecture_group"].nunique(),
        criterion="maxclust",
    )

    cluster_assignments = metadata.copy()
    cluster_assignments["cluster_order"] = [ordered_models.index(model) for model in metadata["model_key"]]
    cluster_assignments["cluster_training_exact"] = clusters_training_exact
    cluster_assignments["cluster_training_coarse"] = clusters_training_coarse
    cluster_assignments["cluster_architecture"] = clusters_architecture
    cluster_assignments = cluster_assignments.sort_values("cluster_order").reset_index(drop=True)

    pairwise = build_pairwise_summary(corr, metadata)
    pairwise_summary = summarize_pairwise(pairwise)

    summary = {
        "n_models": int(len(metadata)),
        "n_materials": int(len(errors)),
        "ari_training_exact": adjusted_rand_score(metadata["training_combo"], clusters_training_exact),
        "ari_training_coarse": adjusted_rand_score(metadata["training_group"], clusters_training_coarse),
        "ari_architecture": adjusted_rand_score(metadata["architecture_group"], clusters_architecture),
    }

    corr.loc[ordered_models, ordered_models].to_csv(output_table_dir / "error_correlation_matrix.csv")
    cluster_assignments.to_csv(output_table_dir / "cluster_assignments.csv", index=False)
    pairwise.to_csv(output_table_dir / "pairwise_correlations.csv", index=False)
    pairwise_summary.to_csv(output_table_dir / "pairwise_correlation_summary.csv", index=False)
    (output_table_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    create_clustermap(
        corr.loc[ordered_models, ordered_models],
        linkage_matrix,
        metadata.set_index("model_key").loc[ordered_models].reset_index(),
        output_figure_dir / "error_correlation_clustermap.png",
    )


if __name__ == "__main__":
    main()

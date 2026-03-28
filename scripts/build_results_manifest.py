from __future__ import annotations

import json
from pathlib import Path

from dva_project.settings import RESULTS_DIR
from dva_project.utils import ensure_dir


def file_entry(path: str, role: str) -> dict[str, str]:
    return {"path": path, "role": role}


def main() -> None:
    manifest_dir = RESULTS_DIR / "manifest"
    ensure_dir(manifest_dir)

    sections = [
        {
            "section": "foundation",
            "items": [
                {
                    "id": "data_snapshot",
                    "title": "Frozen benchmark and model snapshot",
                    "claim": "The project uses a frozen 45-model snapshot aligned to the guide document while preserving the live 53-model public state for sensitivity checks.",
                    "scripts": [
                        "scripts/build_model_table.py",
                    ],
                    "files": [
                        file_entry("data/processed/model_metadata_snapshot_45.csv", "frozen model metadata"),
                        file_entry("data/processed/discovery_prediction_matrix_snapshot_45.parquet", "frozen prediction matrix"),
                        file_entry("data/processed/snapshot_45_summary.json", "snapshot summary"),
                        file_entry("data/processed/model_metadata.csv", "live model metadata"),
                    ],
                },
                {
                    "id": "analysis_04_feature_base",
                    "title": "Collective-failure feature base",
                    "claim": "All blind-spot analyses reuse a common material-level feature table derived from the WBM benchmark.",
                    "scripts": [
                        "scripts/analysis_04_collective_failures.py",
                    ],
                    "files": [
                        file_entry("results/tables/analysis_04/collective_failure_features.csv", "shared material-level feature table"),
                        file_entry("results/tables/analysis_04/summary.json", "base blind-spot summary"),
                    ],
                },
            ],
        },
        {
            "section": "core_claims",
            "items": [
                {
                    "id": "analysis_01",
                    "title": "Training data dominates architecture",
                    "claim": "Training-data grouping explains more benchmark variance than architecture across the main regression/classification metrics.",
                    "scripts": [
                        "scripts/analysis_01_variance_decomposition.py",
                        "scripts/analysis_01_control_experiments.py",
                    ],
                    "files": [
                        file_entry("results/tables/analysis_01/anova_primary.csv", "primary ANOVA table"),
                        file_entry("results/tables/analysis_01/summary.json", "variance summary"),
                        file_entry("results/tables/analysis_01_control/pair_delta_summary.csv", "control-pair summary"),
                    ],
                },
                {
                    "id": "analysis_02",
                    "title": "Errors cluster by training data",
                    "claim": "Model error patterns align much more strongly with training-data groupings than with architecture classes.",
                    "scripts": [
                        "scripts/analysis_02_error_correlation.py",
                    ],
                    "files": [
                        file_entry("results/tables/analysis_02/summary.json", "clustering summary"),
                        file_entry("results/tables/analysis_02/pairwise_correlation_summary.csv", "pairwise correlation summary"),
                    ],
                },
                {
                    "id": "analysis_03",
                    "title": "Data scaling is more robust than parameter scaling",
                    "claim": "Expanding data gives the most stable scaling gains, while small ensembles saturate quickly.",
                    "scripts": [
                        "scripts/analysis_03_scaling_laws.py",
                    ],
                    "files": [
                        file_entry("results/tables/analysis_03/data_scaling_fits.csv", "data-scaling fits"),
                        file_entry("results/tables/analysis_03/parameter_scaling_fits.csv", "parameter-scaling fits"),
                        file_entry("results/tables/analysis_03/ensemble_outlier_audit.csv", "ensemble audit"),
                    ],
                },
                {
                    "id": "analysis_04_to_12",
                    "title": "Blind spots arise inside familiar chemistry",
                    "claim": "Collective failures are not simple chemistry-OOD cases; they concentrate in familiar compositions and sharpen into structure/prototype-density blind spots.",
                    "scripts": [
                        "scripts/analysis_04_collective_failures.py",
                        "scripts/analysis_04_coverage_reference_sensitivity.py",
                        "scripts/analysis_09_failure_overlap_stratification.py",
                        "scripts/analysis_10_structure_overlap_proxy.py",
                        "scripts/analysis_11_structure_density_and_symmetry.py",
                        "scripts/analysis_12_prototype_spacegroup_rarity.py",
                    ],
                    "files": [
                        file_entry("results/tables/analysis_04/summary.json", "collective-failure summary"),
                        file_entry("results/tables/analysis_04_coverage_sensitivity/coverage_reference_delta_summary.csv", "coverage sensitivity"),
                        file_entry("results/tables/analysis_11/density_tier_bootstrap_differences.csv", "density-tier contrasts"),
                        file_entry("results/tables/analysis_12/rarity_bootstrap_differences.csv", "rarity contrasts"),
                    ],
                },
                {
                    "id": "analysis_05",
                    "title": "Pareto frontier and budget guidance",
                    "claim": "Resource allocation is best guided by a Pareto frontier across performance and training cost, with data expansion generally beating parameter growth.",
                    "scripts": [
                        "scripts/analysis_05_resource_allocation.py",
                    ],
                    "files": [
                        file_entry("results/tables/analysis_05/budget_recommendations.csv", "budget recommendation table"),
                        file_entry("results/tables/analysis_05/strategy_comparison.csv", "strategy comparison table"),
                    ],
                },
            ],
        },
        {
            "section": "robustness_and_dependence",
            "items": [
                {
                    "id": "analysis_06_to_08_and_24",
                    "title": "Sensitivity, family dependence, uncertainty, and permutation significance",
                    "claim": "The main data-over-architecture conclusion survives the live 53-model set, family-aware resampling, uncertainty-focused resampling, and direct permutation-based significance checks.",
                    "scripts": [
                        "scripts/analysis_06_sensitivity_checks.py",
                        "scripts/analysis_07_family_dependence_robustness.py",
                        "scripts/analysis_08_uncertainty_resampling.py",
                        "scripts/analysis_24_permutation_significance.py",
                    ],
                    "files": [
                        file_entry("results/tables/sensitivity/summary.json", "53-model sensitivity summary"),
                        file_entry("results/tables/analysis_07/summary.json", "family-robustness summary"),
                        file_entry("results/tables/analysis_08/summary.json", "uncertainty-resampling summary"),
                        file_entry("results/tables/analysis_24/anova_permutation_summary.csv", "permutation ANOVA summary"),
                        file_entry("results/tables/analysis_24/cluster_permutation_summary.csv", "permutation cluster summary"),
                    ],
                },
            ],
        },
        {
            "section": "mechanism_decomposition",
            "items": [
                {
                    "id": "analysis_14_to_20",
                    "title": "Background-branch mechanism decomposition",
                    "claim": "Prototype/spacegroup hotspot signals decompose into interpretable component and interaction structure; any leftover background residual is mostly optional ranking polish.",
                    "scripts": [
                        "scripts/analysis_14_hotspot_ablation.py",
                        "scripts/analysis_15_prototype_component_decomposition.py",
                        "scripts/analysis_16_component_ablation.py",
                        "scripts/analysis_17_residual_full_token_branches.py",
                        "scripts/analysis_18_background_interactions.py",
                        "scripts/analysis_19_background_higher_order_interactions.py",
                        "scripts/analysis_20_background_residual_token_layer.py",
                        "scripts/analysis_21_background_target_encoding_ranker.py",
                    ],
                    "files": [
                        file_entry("results/tables/analysis_14/summary.json", "hotspot ablation summary"),
                        file_entry("results/tables/analysis_15/summary.json", "prototype-component summary"),
                        file_entry("results/tables/analysis_19/summary.json", "higher-order interaction summary"),
                        file_entry("results/tables/analysis_21/summary.json", "optional background target-encoding summary"),
                    ],
                },
                {
                    "id": "analysis_22_to_23",
                    "title": "tI10 branch decomposition",
                    "claim": "The tI10 hotspot is a compact mixture of structural submodes, and its small residual false-negative tail is not worth another interpretability trade.",
                    "scripts": [
                        "scripts/build_ti10_branch_mode_proxy.py",
                        "scripts/analysis_22_ti10_failure_mode_decomposition.py",
                        "scripts/analysis_23_ti10_target_encoding_ranker.py",
                    ],
                    "files": [
                        file_entry("data/processed/singleton_high_risk_ti10_branch_mode_proxy_summary.json", "tI10 proxy summary"),
                        file_entry("results/tables/analysis_22/summary.json", "tI10 structural decomposition summary"),
                        file_entry("results/tables/analysis_23/summary.json", "optional tI10 target-encoding summary"),
                    ],
                },
            ],
        },
        {
            "section": "status",
            "items": [
                {
                    "id": "current_recommendation",
                    "title": "Recommended immediate next phase",
                    "claim": "Stop branch-level residual chasing, freeze the evidence base, and move to reproducibility and manuscript packaging.",
                    "scripts": [],
                    "files": [
                        file_entry("WORKLOG.md", "active work log"),
                        file_entry("DATA_STATUS.md", "current evidence status"),
                    ],
                },
            ],
        },
    ]

    manifest = {
        "project": "data_vs_architecture",
        "manifest_version": 1,
        "sections": sections,
    }

    json_path = manifest_dir / "results_manifest.json"
    md_path = manifest_dir / "results_manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines: list[str] = [
        "# Results Manifest",
        "",
        "This manifest maps the current main claims to the scripts and output assets that support them.",
        "",
    ]
    for section in sections:
        lines.append(f"## {section['section'].replace('_', ' ').title()}")
        lines.append("")
        for item in section["items"]:
            lines.append(f"### {item['title']}")
            lines.append("")
            lines.append(f"- Claim: {item['claim']}")
            if item["scripts"]:
                lines.append("- Scripts:")
                for script in item["scripts"]:
                    lines.append(f"  - `{script}`")
            if item["files"]:
                lines.append("- Files:")
                for file_info in item["files"]:
                    lines.append(f"  - `{file_info['path']}`: {file_info['role']}")
            lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

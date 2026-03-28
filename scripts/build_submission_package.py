from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
PACKAGE_DIR = PAPER_DIR / "submission_package"
ARCHIVE_BASE = PAPER_DIR / "submission_package_bundle"


ROOT_DOCS = [
    ROOT / "DATA_RESULTS_SUMMARY.md",
    ROOT / "WORKLOG.md",
    ROOT / "DATA_STATUS.md",
    ROOT / "REPRO_RUNBOOK.md",
    PAPER_DIR / "manuscript_submission_ready.md",
    PAPER_DIR / "caption_drafts.md",
    PAPER_DIR / "manuscript_number_audit.md",
    PAPER_DIR / "packaging_checklist.md",
    PAPER_DIR / "figure_inventory.md",
    PAPER_DIR / "template_transfer_packet.md",
    PAPER_DIR / "table_01_budget_recommendations.md",
]

MAIN_FIGURE_DIRS = [
    (ROOT / "results/figures/figure_01_study_overview_redesign", "figure_01_study_overview"),
    (ROOT / "results/figures/figure_02_variance_controls", "figure_02_variance_controls"),
    (ROOT / "results/figures/figure_03_scaling_laws", "figure_03_scaling_laws"),
    (ROOT / "results/figures/figure_04_collective_failures", "figure_04_collective_failures"),
    (ROOT / "results/figures/figure_05_pareto_strategy", "figure_05_pareto_strategy"),
]

SUPPLEMENTARY_FIGURE_DIRS = [
    (ROOT / "results/figures/figure_s01_error_clustermap", "figure_s01_error_clustermap"),
    (ROOT / "results/figures/figure_s02_paired_controls_distribution", "figure_s02_paired_controls_distribution"),
    (ROOT / "results/figures/figure_s03_feature_radar", "figure_s03_feature_radar"),
    (ROOT / "results/figures/figure_s04_decision_tree", "figure_s04_decision_tree"),
    (ROOT / "results/figures/figure_s05_sensitivity_overview", "figure_s05_sensitivity_overview"),
    (ROOT / "results/figures/figure_s06_permutation_nulls", "figure_s06_permutation_nulls"),
    (ROOT / "results/figures/figure_s07_family_resampling", "figure_s07_family_resampling"),
    (ROOT / "results/figures/figure_s08_bootstrap_uncertainty", "figure_s08_bootstrap_uncertainty"),
]

TABLE_SUPPORT = [
    ROOT / "results/tables/analysis_05/budget_recommendations.csv",
    ROOT / "results/tables/analysis_05/pareto_frontier_gpu_hours.csv",
    ROOT / "results/tables/analysis_05/strategy_comparison.csv",
    ROOT / "results/tables/analysis_24/summary.json",
]

EVIDENCE_FILES = [
    ROOT / "data/processed/snapshot_45_summary.json",
    ROOT / "data/processed/model_metadata_snapshot_45.csv",
    ROOT / "results/manifest/results_manifest.md",
    ROOT / "results/manifest/results_manifest.json",
]

EVIDENCE_DIRS = [
    ROOT / "results/figures/figure_01_overview",
    ROOT / "results/figures/analysis_01",
    ROOT / "results/figures/analysis_01_control",
    ROOT / "results/figures/analysis_02",
    ROOT / "results/figures/analysis_03",
    ROOT / "results/figures/analysis_04",
    ROOT / "results/figures/analysis_05",
    ROOT / "results/figures/sensitivity",
    ROOT / "results/tables/figure_01",
    ROOT / "results/tables/analysis_01",
    ROOT / "results/tables/analysis_01_control",
    ROOT / "results/tables/analysis_02",
    ROOT / "results/tables/analysis_03",
    ROOT / "results/tables/analysis_04",
    ROOT / "results/tables/analysis_04_coverage_sensitivity",
    ROOT / "results/tables/analysis_05",
    ROOT / "results/tables/analysis_07",
    ROOT / "results/tables/analysis_08",
    ROOT / "results/tables/analysis_11",
    ROOT / "results/tables/analysis_12",
    ROOT / "results/tables/analysis_15",
    ROOT / "results/tables/analysis_19",
    ROOT / "results/tables/analysis_22",
    ROOT / "results/tables/analysis_24",
    ROOT / "results/tables/sensitivity",
]


def reset_package_dir() -> None:
    if PACKAGE_DIR.exists():
        if PACKAGE_DIR.is_dir():
            shutil.rmtree(PACKAGE_DIR)
        else:
            PACKAGE_DIR.unlink()
    PACKAGE_DIR.mkdir(parents=True)
    (PACKAGE_DIR / "figures" / "main").mkdir(parents=True)
    (PACKAGE_DIR / "figures" / "supplementary").mkdir(parents=True)
    (PACKAGE_DIR / "tables").mkdir(parents=True)
    (PACKAGE_DIR / "evidence").mkdir(parents=True)


def copy_files(files: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / source.name)


def copy_directories(source_dirs: list[tuple[Path, str]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source_dir, target_name in source_dirs:
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        if not source_dir.is_dir():
            raise NotADirectoryError(source_dir)
        shutil.copytree(source_dir, destination / target_name, dirs_exist_ok=True)


def copy_evidence() -> None:
    evidence_root = PACKAGE_DIR / "evidence"
    for source in EVIDENCE_FILES:
        relative = source.relative_to(ROOT)
        destination = evidence_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for source_dir in EVIDENCE_DIRS:
        relative = source_dir.relative_to(ROOT)
        destination = evidence_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, destination, dirs_exist_ok=True)


def write_readmes() -> None:
    (PACKAGE_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Submission Package",
                "",
                "This directory is a template-neutral handoff package for manuscript transfer, figure placement, and numeric verification.",
                "",
                "## Contents",
                "",
                "- `manuscript_submission_ready.md`: clean default main text.",
                "- `DATA_RESULTS_SUMMARY.md`: one-file summary of all major data results.",
                "- `caption_drafts.md`: figure and table captions.",
                "- `manuscript_number_audit.md`: manuscript-facing number audit.",
                "- `packaging_checklist.md`: figure and table placement map.",
                "- `figure_inventory.md`: compact figure plan.",
                "- `template_transfer_packet.md`: transfer order and scope guardrails.",
                "- `table_01_budget_recommendations.md`: paper-side Table 1 rendering support.",
                "- `WORKLOG.md`, `DATA_STATUS.md`, `REPRO_RUNBOOK.md`: project status and rerun context.",
                "- `figures/main/`: grouped main-text figure folders with panel-level PDF and PNG assets.",
                "- `figures/supplementary/`: grouped supplementary figure folders with panel-level PDF and PNG assets.",
                "- `tables/`: compact transfer-side table files.",
                "- `evidence/`: mirrored authoritative support files. If a support document references `results/...` or `data/...`, use the same relative path under `evidence/`.",
                "",
                "## Intended Use",
                "",
                "1. Start from `template_transfer_packet.md`.",
                "2. Move `manuscript_submission_ready.md` into the target journal template.",
                "3. Insert figures from the grouped folders under `figures/main/` and `figures/supplementary/`.",
                "4. Pull captions from `caption_drafts.md`.",
                "5. Validate quoted numbers with `manuscript_number_audit.md` against the mirrored files under `evidence/`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (PACKAGE_DIR / "figures" / "main" / "README.md").write_text(
        "Main-text figure folders with panel-level PDF/PNG assets and preview images.\n",
        encoding="utf-8",
    )
    (PACKAGE_DIR / "figures" / "supplementary" / "README.md").write_text(
        "Supplementary figure folders with panel-level PDF/PNG assets and preview images.\n",
        encoding="utf-8",
    )
    (PACKAGE_DIR / "figures" / "FIGURE_REDESIGN_PLAN.md").write_text(
        "\n".join(
            [
                "# Figure Redesign Plan",
                "",
                "This submission package uses grouped figure folders so each panel can be assembled manually for the manuscript.",
                "",
                "## Main Figures",
                "",
                "- `main/figure_01_study_overview/`",
                "- `main/figure_02_variance_controls/`",
                "- `main/figure_03_scaling_laws/`",
                "- `main/figure_04_collective_failures/`",
                "- `main/figure_05_pareto_strategy/`",
                "",
                "## Supplementary Figures",
                "",
                "- `supplementary/figure_s01_error_clustermap/`",
                "- `supplementary/figure_s02_paired_controls_distribution/`",
                "- `supplementary/figure_s03_feature_radar/`",
                "- `supplementary/figure_s04_decision_tree/`",
                "- `supplementary/figure_s05_sensitivity_overview/`",
                "- `supplementary/figure_s06_permutation_nulls/`",
                "- `supplementary/figure_s07_family_resampling/`",
                "- `supplementary/figure_s08_bootstrap_uncertainty/`",
                "",
                "Each figure folder contains panel-level `.pdf` and `.png` assets, a preview image, and `assembly_notes.md`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (PACKAGE_DIR / "tables" / "README.md").write_text(
        "Compact table-side support files for transfer. Authoritative source files are mirrored under `../evidence/results/tables/`.\n",
        encoding="utf-8",
    )
    (PACKAGE_DIR / "evidence" / "README.md").write_text(
        "\n".join(
            [
                "# Evidence Mirror",
                "",
                "This directory mirrors the project-relative support paths used in the manuscript-facing audit and packaging documents.",
                "",
                "- If a support file references `results/...`, use `evidence/results/...`.",
                "- If a support file references `data/...`, use `evidence/data/...`.",
                "- The mirrored files here are the authoritative bundle-side sources for manuscript validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_archive() -> Path:
    archive_path = ARCHIVE_BASE.with_suffix(".zip")
    if archive_path.exists():
        archive_path.unlink()
    return Path(
        shutil.make_archive(
            str(ARCHIVE_BASE),
            "zip",
            root_dir=PACKAGE_DIR.parent,
            base_dir=PACKAGE_DIR.name,
        )
    )


def main() -> None:
    reset_package_dir()
    copy_files(ROOT_DOCS, PACKAGE_DIR)
    copy_directories(MAIN_FIGURE_DIRS, PACKAGE_DIR / "figures" / "main")
    copy_directories(SUPPLEMENTARY_FIGURE_DIRS, PACKAGE_DIR / "figures" / "supplementary")
    copy_files(TABLE_SUPPORT, PACKAGE_DIR / "tables")
    copy_evidence()
    write_readmes()
    archive_path = build_archive()
    print(f"Built submission package at: {PACKAGE_DIR}")
    print(f"Built archive at: {archive_path}")


if __name__ == "__main__":
    main()

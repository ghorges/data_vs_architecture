from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@dataclass(frozen=True)
class Step:
    step_id: str
    stage: str
    description: str
    command: list[str]
    optional: bool = False


STAGE_ORDER = [
    "acquisition",
    "core",
    "coverage",
    "mechanism",
    "handoff",
]


PIPELINE_STEPS: list[Step] = [
    Step("download_inputs", "acquisition", "Download benchmark metadata and model predictions.", ["download_inputs.py"]),
    Step("build_model_table", "acquisition", "Build live model metadata table.", ["build_model_table.py"]),
    Step(
        "prepare_prediction_matrix",
        "acquisition",
        "Assemble prediction and error matrices for the live public model set.",
        ["prepare_prediction_matrix.py"],
    ),
    Step("freeze_snapshot_45", "acquisition", "Freeze the guide-aligned 45-model snapshot.", ["freeze_snapshot_45.py"]),
    Step("analysis_01_variance_decomposition", "core", "Primary data-vs-architecture variance decomposition.", ["analysis_01_variance_decomposition.py"]),
    Step("analysis_01_control_experiments", "core", "Same-data vs same-family control comparison.", ["analysis_01_control_experiments.py"]),
    Step("analysis_02_error_correlation", "core", "Error-correlation clustering analysis.", ["analysis_02_error_correlation.py"]),
    Step("analysis_03_scaling_laws", "core", "Scaling-law and ensemble saturation analysis.", ["analysis_03_scaling_laws.py"]),
    Step("analysis_04_collective_failures", "core", "Base collective-failure feature table and first blind-spot pass.", ["analysis_04_collective_failures.py"]),
    Step("analysis_05_resource_allocation", "core", "Pareto frontier and budget recommendation analysis.", ["analysis_05_resource_allocation.py"]),
    Step("analysis_06_sensitivity_checks", "core", "Live-53-model sensitivity checks.", ["analysis_06_sensitivity_checks.py"]),
    Step("analysis_07_family_dependence_robustness", "core", "Family-aware dependence robustness checks.", ["analysis_07_family_dependence_robustness.py"]),
    Step("analysis_08_uncertainty_resampling", "core", "Uncertainty-focused resampling for the main claims.", ["analysis_08_uncertainty_resampling.py"]),
    Step("analysis_24_permutation_significance", "core", "Permutation-based significance checks for the main data-vs-architecture claims.", ["analysis_24_permutation_significance.py"]),
    Step(
        "build_mptrj_formula_index",
        "coverage",
        "Build the MPtrj reduced-formula reference index.",
        ["build_mptrj_formula_index.py", "--download-json", "--timeout", "600"],
    ),
    Step(
        "build_mp2022_formula_index",
        "coverage",
        "Build the MP 2022 reduced-formula reference index.",
        ["build_mp2022_formula_index.py", "--download-json", "--timeout", "600"],
    ),
    Step("build_formula_union_index", "coverage", "Combine the MPtrj and MP 2022 formula indices.", ["build_formula_union_index.py"]),
    Step(
        "build_wbm_mptrj_coverage_proxy",
        "coverage",
        "Build WBM coverage proxy against MPtrj formulas.",
        ["build_wbm_mptrj_coverage_proxy.py", "--reference-index", "data/processed/mptrj_formula_index.parquet", "--output-prefix", "wbm_mptrj"],
    ),
    Step(
        "build_wbm_mp2022_coverage_proxy",
        "coverage",
        "Build WBM coverage proxy against MP 2022 formulas.",
        ["build_wbm_mptrj_coverage_proxy.py", "--reference-index", "data/processed/mp2022_formula_index.parquet", "--output-prefix", "wbm_mp2022"],
    ),
    Step(
        "build_wbm_union_coverage_proxy",
        "coverage",
        "Build WBM coverage proxy against the MPtrj+MP2022 union.",
        [
            "build_wbm_mptrj_coverage_proxy.py",
            "--reference-index",
            "data/processed/mptrj_mp2022_union_formula_index.parquet",
            "--output-prefix",
            "wbm_mptrj_mp2022_union",
        ],
    ),
    Step("analysis_04_coverage_reference_sensitivity", "coverage", "Coverage-reference sensitivity checks for the blind-spot analysis.", ["analysis_04_coverage_reference_sensitivity.py"]),
    Step("analysis_09_failure_overlap_stratification", "coverage", "Failure overlap stratification in familiar chemistry.", ["analysis_09_failure_overlap_stratification.py"]),
    Step("build_structure_signature_indexes", "coverage", "Build structure-signature reference indices for MPtrj, MP 2022, and their union.", ["build_structure_signature_indexes.py"]),
    Step("build_wbm_structure_overlap_proxy", "coverage", "Build light structure-overlap proxy for WBM materials.", ["build_wbm_structure_overlap_proxy.py"]),
    Step("analysis_10_structure_overlap_proxy", "coverage", "Test whether light structure overlap explains collective failures.", ["analysis_10_structure_overlap_proxy.py"]),
    Step("build_wbm_structure_density_proxy", "coverage", "Build structure-density support proxies inside familiar chemistry.", ["build_wbm_structure_density_proxy.py"]),
    Step("analysis_11_structure_density_and_symmetry", "coverage", "Analyze density tiers and symmetry hotspots.", ["analysis_11_structure_density_and_symmetry.py"]),
    Step("build_wbm_prototype_spacegroup_rarity_proxy", "coverage", "Build prototype and spacegroup rarity proxies.", ["build_wbm_prototype_spacegroup_rarity_proxy.py"]),
    Step("analysis_12_prototype_spacegroup_rarity", "coverage", "Prototype and spacegroup rarity analysis.", ["analysis_12_prototype_spacegroup_rarity.py"]),
    Step("build_wbm_motif_hotspot_proxy", "mechanism", "Build robust motif hotspot buckets.", ["build_wbm_motif_hotspot_proxy.py"]),
    Step("analysis_13_motif_hotspot_modeling", "mechanism", "Intermediate motif-hotspot modeling pass.", ["analysis_13_motif_hotspot_modeling.py"], optional=True),
    Step("analysis_14_hotspot_ablation", "mechanism", "Ablate hotspot buckets and test robustness.", ["analysis_14_hotspot_ablation.py"]),
    Step("build_wbm_prototype_component_proxy", "mechanism", "Build interpretable prototype-component buckets.", ["build_wbm_prototype_component_proxy.py"]),
    Step("analysis_15_prototype_component_decomposition", "mechanism", "Decompose hotspot identity into interpretable components.", ["analysis_15_prototype_component_decomposition.py"]),
    Step("analysis_16_component_ablation", "mechanism", "Branch-level component ablation analysis.", ["analysis_16_component_ablation.py"]),
    Step("analysis_17_residual_full_token_branches", "mechanism", "Residual full-token analysis across major branches.", ["analysis_17_residual_full_token_branches.py"]),
    Step("build_background_component_interaction_proxy", "mechanism", "Build pairwise interaction buckets for the background branch.", ["build_background_component_interaction_proxy.py"]),
    Step("analysis_18_background_interactions", "mechanism", "Background branch pairwise interaction analysis.", ["analysis_18_background_interactions.py"]),
    Step("build_background_higher_order_interaction_proxy", "mechanism", "Build triple-interaction buckets for the background branch.", ["build_background_higher_order_interaction_proxy.py"]),
    Step("analysis_19_background_higher_order_interactions", "mechanism", "Background higher-order interaction analysis.", ["analysis_19_background_higher_order_interactions.py"]),
    Step("build_background_residual_token_proxy", "mechanism", "Build compact residual-token buckets for the background branch.", ["build_background_residual_token_proxy.py"]),
    Step("analysis_20_background_residual_token_layer", "mechanism", "Test compact residual-token layer in the background branch.", ["analysis_20_background_residual_token_layer.py"]),
    Step("analysis_21_background_target_encoding_ranker", "mechanism", "Optional OOF target-encoding pass for the background branch.", ["analysis_21_background_target_encoding_ranker.py"], optional=True),
    Step("build_ti10_branch_mode_proxy", "mechanism", "Build structural-mode buckets for the tI10 branch.", ["build_ti10_branch_mode_proxy.py"]),
    Step("analysis_22_ti10_failure_mode_decomposition", "mechanism", "Decompose the tI10 branch into structural submodes.", ["analysis_22_ti10_failure_mode_decomposition.py"]),
    Step("analysis_23_ti10_target_encoding_ranker", "mechanism", "Optional OOF target-encoding pass for the tI10 branch.", ["analysis_23_ti10_target_encoding_ranker.py"], optional=True),
    Step("build_results_manifest", "handoff", "Build the claim-to-asset results manifest.", ["build_results_manifest.py"]),
    Step("figure_01_study_overview", "handoff", "Regenerate the study-overview figure asset.", ["figure_01_study_overview.py"], optional=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the evidence-generation pipeline in ordered stages.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGE_ORDER,
        help="Restrict execution to one or more stages. Can be passed multiple times.",
    )
    parser.add_argument(
        "--step",
        action="append",
        choices=[step.step_id for step in PIPELINE_STEPS],
        help="Run only the named step(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--from-step",
        choices=[step.step_id for step in PIPELINE_STEPS],
        help="Start execution from this step id.",
    )
    parser.add_argument(
        "--to-step",
        choices=[step.step_id for step in PIPELINE_STEPS],
        help="Stop execution after this step id.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional exploratory or packaging steps.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selected steps without running them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands in execution order without running them.",
    )
    return parser.parse_args()


def select_steps(args: argparse.Namespace) -> list[Step]:
    steps = PIPELINE_STEPS
    if not args.include_optional:
        steps = [step for step in steps if not step.optional]

    if args.stage:
        requested = set(args.stage)
        steps = [step for step in steps if step.stage in requested]

    if args.step:
        requested_steps = set(args.step)
        steps = [step for step in steps if step.step_id in requested_steps]

    if args.from_step or args.to_step:
        all_ids = [step.step_id for step in steps]
        if args.from_step and args.from_step not in all_ids:
            raise ValueError(f"--from-step {args.from_step!r} is not present in the current selection")
        if args.to_step and args.to_step not in all_ids:
            raise ValueError(f"--to-step {args.to_step!r} is not present in the current selection")
        start = all_ids.index(args.from_step) if args.from_step else 0
        end = all_ids.index(args.to_step) if args.to_step else len(steps) - 1
        if start > end:
            raise ValueError("--from-step appears after --to-step in the current selection")
        steps = steps[start : end + 1]

    return steps


def format_command(step: Step) -> str:
    python_exe = Path(sys.executable).name
    pieces = [python_exe, *step.command]
    return subprocess.list2cmdline(pieces)


def print_steps(steps: list[Step]) -> None:
    current_stage = None
    for index, step in enumerate(steps, start=1):
        if step.stage != current_stage:
            current_stage = step.stage
            print(f"\n[{current_stage}]")
        optional_label = " (optional)" if step.optional else ""
        print(f"{index:02d}. {step.step_id}{optional_label}")
        print(f"    {step.description}")
        print(f"    {format_command(step)}")


def run_step(step: Step) -> None:
    command = [sys.executable, *step.command]
    if command[1].endswith(".py"):
        command[1] = str(SCRIPTS_DIR / command[1])
    print(f"\n==> {step.step_id}")
    print(f"    {step.description}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    steps = select_steps(args)
    if not steps:
        raise SystemExit("No steps selected.")

    if args.list or args.dry_run:
        print_steps(steps)
        return

    for step in steps:
        run_step(step)


if __name__ == "__main__":
    main()

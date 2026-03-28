from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis_15_prototype_component_decomposition import load_analysis_frame
from analysis_16_component_ablation import assign_component_branch
from dva_project.settings import PROCESSED_DIR, RESULTS_DIR
from dva_project.utils import ensure_dir


PAIRWISE_COLUMNS = [
    "frequent_formula_x_pearson_bucket",
    "frequent_formula_x_spacegroup_bucket",
    "frequent_pearson_x_spacegroup_bucket",
    "frequent_pearson_x_wyckoff_bucket",
    "frequent_formula_x_wyckoff_bucket",
    "frequent_spacegroup_x_wyckoff_bucket",
]
TRIPLE_COLUMNS = [
    "frequent_formula_x_pearson_x_spacegroup_bucket",
    "frequent_formula_x_pearson_x_wyckoff_bucket",
    "frequent_formula_x_spacegroup_x_wyckoff_bucket",
    "frequent_pearson_x_spacegroup_x_wyckoff_bucket",
]
OUTCOMES = [
    "collective_failure",
    "collective_false_negative",
]
CV_N_SPLITS = 5
CV_N_REPEATS = 10
TARGET_SMOOTHING = 10.0


@dataclass(frozen=True)
class MixedFeatureSet:
    name: str
    categorical: list[str]
    numeric: list[str]


BASE_CATEGORICAL_COLUMNS = [
    "crystal_system",
    "frequent_spacegroup_bucket",
    "frequent_formula_family_bucket",
    "frequent_pearson_symbol_bucket",
    "frequent_wyckoff_signature_bucket",
    *PAIRWISE_COLUMNS,
    *TRIPLE_COLUMNS,
]
FEATURE_SETS = [
    MixedFeatureSet(
        name="higher_order_bundle",
        categorical=BASE_CATEGORICAL_COLUMNS,
        numeric=[],
    ),
    MixedFeatureSet(
        name="higher_order_plus_token_te",
        categorical=BASE_CATEGORICAL_COLUMNS,
        numeric=["token_te_mean", "token_te_log_count"],
    ),
    MixedFeatureSet(
        name="higher_order_plus_full_token",
        categorical=[*BASE_CATEGORICAL_COLUMNS, "frequent_prototype_token_bucket"],
        numeric=[],
    ),
]


def load_background_frame() -> pd.DataFrame:
    frame = assign_component_branch(load_analysis_frame())
    frame = frame.loc[frame["component_branch"] == "background"].copy()
    pairwise = pd.read_parquet(
        PROCESSED_DIR / "singleton_high_risk_background_component_interaction_proxy.parquet",
    )
    higher_order = pd.read_parquet(
        PROCESSED_DIR / "singleton_high_risk_background_higher_order_interaction_proxy.parquet",
    )
    merged = (
        frame.merge(pairwise, on="material_id", how="left", validate="one_to_one", suffixes=("", "_pair"))
        .merge(higher_order, on="material_id", how="left", validate="one_to_one", suffixes=("", "_triple"))
    )
    fill_columns = BASE_CATEGORICAL_COLUMNS + ["frequent_prototype_token_bucket"]
    for column in fill_columns:
        merged[column] = merged[column].fillna("other")
    return merged


def smoothed_target_encoding(train_tokens: pd.Series, y_train: np.ndarray, test_tokens: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    prior = float(y_train.mean())
    train_frame = pd.DataFrame({"token": train_tokens.to_numpy(), "y": y_train})
    stats = train_frame.groupby("token", as_index=False).agg(token_mean=("y", "mean"), token_count=("y", "size"))
    stats["token_te_mean"] = (
        stats["token_mean"] * stats["token_count"] + prior * TARGET_SMOOTHING
    ) / (stats["token_count"] + TARGET_SMOOTHING)
    stats["token_te_log_count"] = np.log1p(stats["token_count"]).astype(float)
    lookup = stats.set_index("token")[["token_te_mean", "token_te_log_count"]]

    train_features = train_tokens.map(lookup["token_te_mean"]).to_numpy(dtype=float)
    train_counts = train_tokens.map(lookup["token_te_log_count"]).to_numpy(dtype=float)

    test_mean = test_tokens.map(lookup["token_te_mean"]).fillna(prior).to_numpy(dtype=float)
    test_count = test_tokens.map(lookup["token_te_log_count"]).fillna(0.0).to_numpy(dtype=float)

    train_matrix = np.column_stack([train_features, train_counts])
    test_matrix = np.column_stack([test_mean, test_count])
    return train_matrix, test_matrix


def build_pipeline(categorical_columns: list[str], numeric_columns: list[str]) -> Pipeline:
    transformers = []
    if categorical_columns:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns))
    if numeric_columns:
        transformers.append(("num", StandardScaler(), numeric_columns))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def evaluate_background_target_encoding(frame: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame[outcome].astype(int).to_numpy()
    class_counts = np.bincount(y, minlength=2)
    n_splits = int(min(CV_N_SPLITS, class_counts[0], class_counts[1]))
    if n_splits < 2:
        raise ValueError(f"not enough class support for {outcome}")

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=CV_N_REPEATS,
        random_state=17,
    )

    fold_rows: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(frame, y), start=1):
        train_frame = frame.iloc[train_idx].copy()
        test_frame = frame.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        te_train, te_test = smoothed_target_encoding(
            train_tokens=train_frame["frequent_prototype_token_bucket"],
            y_train=y_train,
            test_tokens=test_frame["frequent_prototype_token_bucket"],
        )
        train_frame["token_te_mean"] = te_train[:, 0]
        train_frame["token_te_log_count"] = te_train[:, 1]
        test_frame["token_te_mean"] = te_test[:, 0]
        test_frame["token_te_log_count"] = te_test[:, 1]

        for feature_set in FEATURE_SETS:
            pipeline = build_pipeline(feature_set.categorical, feature_set.numeric)
            X_train = train_frame[feature_set.categorical + feature_set.numeric]
            X_test = test_frame[feature_set.categorical + feature_set.numeric]
            pipeline.fit(X_train, y_train)
            prob = pipeline.predict_proba(X_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            fold_rows.append(
                {
                    "feature_set": feature_set.name,
                    "outcome": outcome,
                    "fold": fold_idx,
                    "n_materials": int(len(frame)),
                    "positive_rate": float(y.mean()),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                    "roc_auc": float(roc_auc_score(y_test, prob)),
                    "average_precision": float(average_precision_score(y_test, prob)),
                    "log_loss": float(log_loss(y_test, prob, labels=[0, 1])),
                }
            )

    fold_table = pd.DataFrame(fold_rows)
    summary = (
        fold_table.groupby(["feature_set", "outcome"], as_index=False)
        .agg(
            n_materials=("n_materials", "first"),
            positive_rate=("positive_rate", "first"),
            n_folds=("fold", "size"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
        )
    )
    return fold_table, summary


def compute_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "balanced_accuracy_mean",
        "average_precision_mean",
        "roc_auc_mean",
        "log_loss_mean",
        "n_materials",
        "positive_rate",
    ]
    pivot = (
        summary.set_index(["outcome", "feature_set"])[value_columns]
        .unstack("feature_set")
        .sort_index()
    )
    out = pd.DataFrame(index=pivot.index).reset_index()
    out["n_materials"] = pivot[("n_materials", "higher_order_bundle")].to_numpy()
    out["positive_rate"] = pivot[("positive_rate", "higher_order_bundle")].to_numpy()

    base_name = "higher_order_bundle"
    te_name = "higher_order_plus_token_te"
    full_name = "higher_order_plus_full_token"

    out["token_te_balanced_accuracy_gain"] = (
        pivot[("balanced_accuracy_mean", te_name)] - pivot[("balanced_accuracy_mean", base_name)]
    ).to_numpy()
    out["full_token_balanced_accuracy_gain"] = (
        pivot[("balanced_accuracy_mean", full_name)] - pivot[("balanced_accuracy_mean", base_name)]
    ).to_numpy()
    out["token_te_vs_full_token_balanced_accuracy_delta"] = (
        pivot[("balanced_accuracy_mean", te_name)] - pivot[("balanced_accuracy_mean", full_name)]
    ).to_numpy()
    out["token_te_average_precision_gain"] = (
        pivot[("average_precision_mean", te_name)] - pivot[("average_precision_mean", base_name)]
    ).to_numpy()
    out["full_token_average_precision_gain"] = (
        pivot[("average_precision_mean", full_name)] - pivot[("average_precision_mean", base_name)]
    ).to_numpy()
    out["token_te_vs_full_token_average_precision_delta"] = (
        pivot[("average_precision_mean", te_name)] - pivot[("average_precision_mean", full_name)]
    ).to_numpy()
    out["token_te_log_loss_improvement"] = (
        pivot[("log_loss_mean", base_name)] - pivot[("log_loss_mean", te_name)]
    ).to_numpy()
    out["full_token_log_loss_improvement"] = (
        pivot[("log_loss_mean", base_name)] - pivot[("log_loss_mean", full_name)]
    ).to_numpy()
    out["token_te_vs_full_token_log_loss_delta"] = (
        pivot[("log_loss_mean", full_name)] - pivot[("log_loss_mean", te_name)]
    ).to_numpy()
    return out


def main() -> None:
    output_dir = RESULTS_DIR / "tables" / "analysis_21"
    ensure_dir(output_dir)

    frame = load_background_frame()
    fold_tables: list[pd.DataFrame] = []
    summary_tables: list[pd.DataFrame] = []
    for outcome in OUTCOMES:
        folds, summary = evaluate_background_target_encoding(frame=frame, outcome=outcome)
        fold_tables.append(folds)
        summary_tables.append(summary)
    cv_folds = pd.concat(fold_tables, ignore_index=True)
    cv_summary = pd.concat(summary_tables, ignore_index=True)
    delta_table = compute_deltas(cv_summary)

    summary_lookup = cv_summary.set_index(["feature_set", "outcome"])
    delta_lookup = delta_table.set_index("outcome")
    summary = {
        "background_branch": {
            "n_materials": int(len(frame)),
            "feature_sets": {
                outcome: {
                    feature_set.name: summary_lookup.loc[(feature_set.name, outcome), [
                        "balanced_accuracy_mean",
                        "average_precision_mean",
                        "log_loss_mean",
                    ]].to_dict()
                    for feature_set in FEATURE_SETS
                }
                for outcome in OUTCOMES
            },
            "key_deltas": {
                outcome: {
                    "token_te_balanced_accuracy_gain": float(
                        delta_lookup.loc[outcome, "token_te_balanced_accuracy_gain"]
                    ),
                    "full_token_balanced_accuracy_gain": float(
                        delta_lookup.loc[outcome, "full_token_balanced_accuracy_gain"]
                    ),
                    "token_te_vs_full_token_balanced_accuracy_delta": float(
                        delta_lookup.loc[outcome, "token_te_vs_full_token_balanced_accuracy_delta"]
                    ),
                    "token_te_average_precision_gain": float(
                        delta_lookup.loc[outcome, "token_te_average_precision_gain"]
                    ),
                    "full_token_average_precision_gain": float(
                        delta_lookup.loc[outcome, "full_token_average_precision_gain"]
                    ),
                    "token_te_vs_full_token_average_precision_delta": float(
                        delta_lookup.loc[outcome, "token_te_vs_full_token_average_precision_delta"]
                    ),
                }
                for outcome in OUTCOMES
            },
        }
    }

    cv_folds.to_csv(output_dir / "feature_set_cv_fold_metrics.csv", index=False)
    cv_summary.to_csv(output_dir / "feature_set_cv_summary.csv", index=False)
    delta_table.to_csv(output_dir / "feature_set_delta_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

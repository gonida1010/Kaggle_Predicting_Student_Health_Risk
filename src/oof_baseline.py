from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight


CLASSES = ["fit", "at-risk", "unhealthy"]
TARGET_MAP = {label: index for index, label in enumerate(CLASSES)}
TARGET_COLUMN = "health_condition"
ID_COLUMN = "id"

NUMERIC_COLUMNS = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]

CATEGORICAL_COLUMNS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]


@dataclass
class CVConfig:
    run_mode: str = "smoke"
    models: tuple[str, ...] = ("catboost", "lightgbm", "xgboost")
    seed: int = 20260704
    n_splits: int = 5
    smoke_rows: int = 60000
    smoke_splits: int = 2
    class_weight_mode: str = "balanced"
    output_dir: str = "artifacts/oof_cv_baseline"
    log_period: int = 100
    early_stopping_rounds: int = 250
    catboost_iterations: int = 4000
    lightgbm_iterations: int = 4500
    xgboost_iterations: int = 4500

    @classmethod
    def for_mode(cls, run_mode: str) -> "CVConfig":
        if run_mode == "full":
            return cls(run_mode="full")
        if run_mode != "smoke":
            raise ValueError("run_mode must be 'smoke' or 'full'")
        return cls(
            run_mode="smoke",
            n_splits=2,
            smoke_rows=60000,
            smoke_splits=2,
            early_stopping_rounds=80,
            catboost_iterations=500,
            lightgbm_iterations=700,
            xgboost_iterations=700,
            log_period=50,
            output_dir="artifacts/oof_cv_baseline_smoke",
        )


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    row_sum = values.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(values)) or np.any(row_sum <= 0):
        raise ValueError("Probability matrix contains invalid rows")
    return (values / row_sum).astype(np.float32)


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls: dict[str, float] = {}
    for class_index, label in enumerate(CLASSES):
        mask = y_true == class_index
        recalls[label] = float((y_pred[mask] == class_index).mean())
    return recalls


def make_sample_weights(y: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"Unknown class_weight_mode: {mode}")
    class_values = np.arange(len(CLASSES))
    weights = compute_class_weight("balanced", classes=class_values, y=y)
    return weights[y].astype(np.float32)


def make_class_weights(y: np.ndarray, mode: str) -> list[float] | None:
    sample_weights = make_sample_weights(y, mode)
    if sample_weights is None:
        return None
    return [float(sample_weights[y == class_index][0]) for class_index in range(len(CLASSES))]


def load_competition_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = root / "data"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    expected_train_columns = {ID_COLUMN, TARGET_COLUMN, *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS}
    expected_test_columns = expected_train_columns - {TARGET_COLUMN}
    if set(train.columns) != expected_train_columns:
        raise ValueError(f"Unexpected train columns: {train.columns.tolist()}")
    if set(test.columns) != expected_test_columns:
        raise ValueError(f"Unexpected test columns: {test.columns.tolist()}")
    if sample.columns.tolist() != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError(f"Unexpected sample columns: {sample.columns.tolist()}")
    if not test[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("test IDs do not match sample submission IDs")
    if train[ID_COLUMN].duplicated().any() or test[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate IDs found")
    return train, test, sample


def data_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    target_counts = train[TARGET_COLUMN].value_counts().reindex(CLASSES, fill_value=0)
    return {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "target_counts": {key: int(value) for key, value in target_counts.items()},
        "target_share": {
            key: float(value) for key, value in (target_counts / len(train)).items()
        },
        "train_missing": {
            key: int(value)
            for key, value in train.drop(columns=[TARGET_COLUMN]).isna().sum().items()
        },
        "test_missing": {key: int(value) for key, value in test.isna().sum().items()},
    }


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore").copy()
    source_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

    for column in source_columns:
        result[f"{column}__missing"] = frame[column].isna().astype(np.int8)
    result["missing_count"] = frame[source_columns].isna().sum(axis=1).astype(np.int8)

    result["sleep_deviation_7_5"] = (frame["sleep_duration"] - 7.5).abs()
    result["heart_rate_deviation_70"] = (frame["heart_rate"] - 70.0).abs()
    result["bmi_deviation_22_5"] = (frame["bmi"] - 22.5).abs()
    result["log_step_count"] = np.log1p(frame["step_count"].clip(lower=0))
    result["log_calorie_expenditure"] = np.log1p(
        frame["calorie_expenditure"].clip(lower=0)
    )
    result["steps_per_exercise_minute"] = frame["step_count"] / (
        frame["exercise_duration"].abs() + 1.0
    )
    result["calories_per_1000_steps"] = frame["calorie_expenditure"] / (
        frame["step_count"].abs() / 1000.0 + 1.0
    )
    result["water_per_1000_calories"] = frame["water_intake"] / (
        frame["calorie_expenditure"].abs() / 1000.0 + 1.0
    )
    result["activity_load"] = (
        np.log1p(frame["step_count"].clip(lower=0))
        + np.log1p(frame["exercise_duration"].clip(lower=0))
    )
    result["sleep_activity_interaction"] = (
        frame["sleep_duration"] * np.log1p(frame["step_count"].clip(lower=0))
    )
    return result


def prepare_feature_views(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_features = add_features(train)
    test_features = add_features(test)
    categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in train_features]

    train_cat = train_features.copy()
    test_cat = test_features.copy()
    for column in categorical_columns:
        train_cat[column] = train_cat[column].fillna("__MISSING__").astype(str)
        test_cat[column] = test_cat[column].fillna("__MISSING__").astype(str)

    train_encoded = train_features.copy()
    test_encoded = test_features.copy()
    for column in categorical_columns:
        train_values = train_features[column].fillna("__MISSING__").astype(str)
        test_values = test_features[column].fillna("__MISSING__").astype(str)
        categories = sorted(train_values.unique().tolist())
        mapping = {value: index for index, value in enumerate(categories)}
        train_encoded[column] = train_values.map(mapping).astype(np.int16)
        test_encoded[column] = test_values.map(mapping).fillna(-1).astype(np.int16)

    return train_cat, test_cat, train_encoded, test_encoded, categorical_columns


def make_fold_assignments(y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    folds = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, valid_index) in enumerate(splitter.split(np.zeros(len(y)), y)):
        folds[valid_index] = fold
    if np.any(folds < 0):
        raise RuntimeError("Some rows were not assigned to a fold")
    return folds


def smoke_indices(y: np.ndarray, rows: int, seed: int) -> np.ndarray:
    if rows <= 0 or rows >= len(y):
        return np.arange(len(y))
    indices, _ = train_test_split(
        np.arange(len(y)),
        train_size=rows,
        stratify=y,
        random_state=seed,
    )
    return np.sort(indices)


class CatBoostBalancedAccuracy:
    def get_final_error(self, error: float, weight: float) -> float:
        return error / weight

    def is_max_optimal(self) -> bool:
        return True

    def evaluate(
        self,
        approxes: list[np.ndarray],
        target: np.ndarray,
        weight: np.ndarray | None,
    ) -> tuple[float, float]:
        prediction = np.vstack(approxes).T.argmax(axis=1)
        score = balanced_accuracy_score(np.asarray(target, dtype=np.int64), prediction)
        return float(score), 1.0


def _history_frame(
    model_name: str,
    fold: int,
    train_bac: list[float],
    valid_bac: list[float],
    train_loss: list[float],
    valid_loss: list[float],
) -> pd.DataFrame:
    length = min(len(train_bac), len(valid_bac), len(train_loss), len(valid_loss))
    return pd.DataFrame(
        {
            "model": model_name,
            "fold": fold,
            "iteration": np.arange(1, length + 1),
            "train_balanced_accuracy": train_bac[:length],
            "valid_balanced_accuracy": valid_bac[:length],
            "train_logloss": train_loss[:length],
            "valid_logloss": valid_loss[:length],
        }
    )


def train_catboost_fold(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    categorical_columns: list[str],
    y: np.ndarray,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    config: CVConfig,
    fold: int,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    from catboost import CatBoostClassifier, Pool

    class_weights = make_class_weights(y[train_index], config.class_weight_mode)
    train_pool = Pool(
        train_features.iloc[train_index],
        label=y[train_index],
        cat_features=categorical_columns,
    )
    valid_pool = Pool(
        train_features.iloc[valid_index],
        label=y[valid_index],
        cat_features=categorical_columns,
    )
    test_pool = Pool(test_features, cat_features=categorical_columns)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric=CatBoostBalancedAccuracy(),
        custom_metric=["MultiClass"],
        iterations=config.catboost_iterations,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=8.0,
        random_strength=0.5,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        class_weights=class_weights,
        random_seed=config.seed + fold,
        thread_count=-1,
        allow_writing_files=False,
        verbose=config.log_period,
    )
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=config.early_stopping_rounds,
    )

    valid_probability = normalize_probabilities(model.predict_proba(valid_pool))
    test_probability = normalize_probabilities(model.predict_proba(test_pool))
    model.save_model(model_dir / f"fold_{fold}.cbm")

    history = model.get_evals_result()
    learn_metrics = history.get("learn", {})
    valid_metrics = history.get("validation", history.get("validation_0", {}))
    bac_key = next(key for key in valid_metrics if "BalancedAccuracy" in key)
    loss_key = next(key for key in valid_metrics if key == "MultiClass")
    history_frame = _history_frame(
        "catboost",
        fold,
        list(learn_metrics[bac_key]),
        list(valid_metrics[bac_key]),
        list(learn_metrics[loss_key]),
        list(valid_metrics[loss_key]),
    )
    best_iteration = int(model.get_best_iteration()) + 1
    score = balanced_accuracy_score(y[valid_index], valid_probability.argmax(axis=1))
    fold_report = {
        "model": "catboost",
        "fold": fold,
        "balanced_accuracy": float(score),
        "best_iteration": best_iteration,
        **{f"recall_{key}": value for key, value in class_recalls(
            y[valid_index], valid_probability.argmax(axis=1)
        ).items()},
    }
    return valid_probability, test_probability, fold_report, history_frame


def train_lightgbm_fold(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y: np.ndarray,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    config: CVConfig,
    fold: int,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    import lightgbm as lgb

    sample_weight = make_sample_weights(y[train_index], config.class_weight_mode)
    evaluation_result: dict[str, dict[str, list[float]]] = {}

    def evaluation_metric(
        y_true: np.ndarray, y_probability: np.ndarray
    ) -> list[tuple[str, float, bool]]:
        probability = np.asarray(y_probability)
        if probability.ndim == 1:
            probability = probability.reshape(len(y_true), len(CLASSES))
        return [
            (
                "balanced_accuracy",
                float(balanced_accuracy_score(y_true, probability.argmax(axis=1))),
                True,
            ),
            (
                "multi_logloss",
                float(log_loss(y_true, probability, labels=np.arange(len(CLASSES)))),
                False,
            ),
        ]

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(CLASSES),
        metric="None",
        n_estimators=config.lightgbm_iterations,
        learning_rate=0.035,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=150,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=config.seed + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        train_features.iloc[train_index],
        y[train_index],
        sample_weight=sample_weight,
        eval_set=[
            (train_features.iloc[train_index], y[train_index]),
            (train_features.iloc[valid_index], y[valid_index]),
        ],
        eval_names=["train", "valid"],
        eval_metric=evaluation_metric,
        callbacks=[
            lgb.early_stopping(
                config.early_stopping_rounds,
                first_metric_only=True,
                verbose=True,
            ),
            lgb.log_evaluation(config.log_period),
            lgb.record_evaluation(evaluation_result),
        ],
    )

    best_iteration = int(model.best_iteration_)
    valid_probability = normalize_probabilities(
        model.predict_proba(train_features.iloc[valid_index], num_iteration=best_iteration)
    )
    test_probability = normalize_probabilities(
        model.predict_proba(test_features, num_iteration=best_iteration)
    )
    model.booster_.save_model(model_dir / f"fold_{fold}.txt", num_iteration=best_iteration)

    history_frame = _history_frame(
        "lightgbm",
        fold,
        evaluation_result["train"]["balanced_accuracy"],
        evaluation_result["valid"]["balanced_accuracy"],
        evaluation_result["train"]["multi_logloss"],
        evaluation_result["valid"]["multi_logloss"],
    )
    score = balanced_accuracy_score(y[valid_index], valid_probability.argmax(axis=1))
    fold_report = {
        "model": "lightgbm",
        "fold": fold,
        "balanced_accuracy": float(score),
        "best_iteration": best_iteration,
        **{f"recall_{key}": value for key, value in class_recalls(
            y[valid_index], valid_probability.argmax(axis=1)
        ).items()},
    }
    return valid_probability, test_probability, fold_report, history_frame


def train_xgboost_fold(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y: np.ndarray,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    config: CVConfig,
    fold: int,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    import xgboost as xgb

    train_weight = make_sample_weights(y[train_index], config.class_weight_mode)
    train_matrix = xgb.DMatrix(
        train_features.iloc[train_index],
        label=y[train_index],
        weight=train_weight,
        feature_names=train_features.columns.tolist(),
    )
    valid_matrix = xgb.DMatrix(
        train_features.iloc[valid_index],
        label=y[valid_index],
        feature_names=train_features.columns.tolist(),
    )
    test_matrix = xgb.DMatrix(test_features, feature_names=train_features.columns.tolist())

    def balanced_accuracy_metric(
        prediction: np.ndarray, matrix: xgb.DMatrix
    ) -> tuple[str, float]:
        probability = np.asarray(prediction)
        if probability.ndim == 1:
            probability = probability.reshape(-1, len(CLASSES))
        target = matrix.get_label().astype(np.int64)
        return "balanced_accuracy", float(
            balanced_accuracy_score(target, probability.argmax(axis=1))
        )

    evaluation_result: dict[str, dict[str, list[float]]] = {}
    params = {
        "objective": "multi:softprob",
        "num_class": len(CLASSES),
        "eval_metric": "mlogloss",
        "eta": 0.035,
        "max_depth": 7,
        "min_child_weight": 10.0,
        "subsample": 0.88,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "max_bin": 256,
        "tree_method": "hist",
        "seed": config.seed + fold,
        "nthread": -1,
    }
    model = xgb.train(
        params,
        train_matrix,
        num_boost_round=config.xgboost_iterations,
        evals=[(train_matrix, "train"), (valid_matrix, "valid")],
        custom_metric=balanced_accuracy_metric,
        maximize=True,
        early_stopping_rounds=config.early_stopping_rounds,
        evals_result=evaluation_result,
        verbose_eval=config.log_period,
    )
    best_iteration = int(model.best_iteration) + 1
    valid_probability = normalize_probabilities(
        model.predict(valid_matrix, iteration_range=(0, best_iteration))
    )
    test_probability = normalize_probabilities(
        model.predict(test_matrix, iteration_range=(0, best_iteration))
    )
    model.save_model(model_dir / f"fold_{fold}.json")

    history_frame = _history_frame(
        "xgboost",
        fold,
        evaluation_result["train"]["balanced_accuracy"],
        evaluation_result["valid"]["balanced_accuracy"],
        evaluation_result["train"]["mlogloss"],
        evaluation_result["valid"]["mlogloss"],
    )
    score = balanced_accuracy_score(y[valid_index], valid_probability.argmax(axis=1))
    fold_report = {
        "model": "xgboost",
        "fold": fold,
        "balanced_accuracy": float(score),
        "best_iteration": best_iteration,
        **{f"recall_{key}": value for key, value in class_recalls(
            y[valid_index], valid_probability.argmax(axis=1)
        ).items()},
    }
    return valid_probability, test_probability, fold_report, history_frame


def save_diagnostic_graphs(
    y: np.ndarray,
    oof_probability: np.ndarray,
    fold_scores: pd.DataFrame,
    history: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = oof_probability.argmax(axis=1)

    matrix = confusion_matrix(y, prediction)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)
    for axis, values, title, value_format in [
        (axes[0], matrix, "OOF confusion matrix", "count"),
        (axes[1], normalized, "OOF confusion matrix by true class", "ratio"),
    ]:
        image = axis.imshow(values, cmap="Blues", vmin=0)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                text = (
                    f"{int(values[row, column]):,}"
                    if value_format == "count"
                    else f"{values[row, column]:.3f}"
                )
                axis.text(column, row, text, ha="center", va="center", fontsize=9)
        axis.set_xticks(range(len(CLASSES)), CLASSES)
        axis.set_yticks(range(len(CLASSES)), CLASSES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_confusion_matrix.png")
    plt.close(fig)

    recall_columns = [f"recall_{label}" for label in CLASSES]
    fig, axis = plt.subplots(figsize=(10, 5.5), dpi=160)
    x = np.arange(len(fold_scores))
    width = 0.24
    colors = ["#2563eb", "#16a34a", "#dc2626"]
    for class_index, (column, color) in enumerate(zip(recall_columns, colors)):
        axis.bar(
            x + (class_index - 1) * width,
            fold_scores[column],
            width=width,
            label=column.removeprefix("recall_"),
            color=color,
        )
    axis.set_xticks(x, [f"fold {value}" for value in fold_scores["fold"]])
    axis.set_ylim(max(0.0, fold_scores[recall_columns].min().min() - 0.03), 1.0)
    axis.set_ylabel("Recall")
    axis.set_title(f"{model_name} class recall by fold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_class_recall.png")
    plt.close(fig)

    if history.empty:
        return
    grouped = history.groupby("iteration", as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)
    axes[0].plot(
        grouped["iteration"],
        grouped["train_balanced_accuracy"],
        label="train",
        color="#dc2626",
    )
    axes[0].plot(
        grouped["iteration"],
        grouped["valid_balanced_accuracy"],
        label="valid",
        color="#2563eb",
    )
    best_row = grouped.loc[grouped["valid_balanced_accuracy"].idxmax()]
    axes[0].axvline(best_row["iteration"], color="#111827", linestyle="--")
    axes[0].set_title("Balanced accuracy")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        grouped["iteration"],
        grouped["train_logloss"],
        label="train",
        color="#dc2626",
    )
    axes[1].plot(
        grouped["iteration"],
        grouped["valid_logloss"],
        label="valid",
        color="#2563eb",
    )
    axes[1].set_title("Multi-class logloss")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Logloss")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    fig.suptitle(f"{model_name} training diagnostics")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_training_curves.png")
    plt.close(fig)


def run_model_cv(
    model_name: str,
    train_cat: pd.DataFrame,
    test_cat: pd.DataFrame,
    train_encoded: pd.DataFrame,
    test_encoded: pd.DataFrame,
    categorical_columns: list[str],
    y: np.ndarray,
    folds: np.ndarray,
    sample: pd.DataFrame,
    config: CVConfig,
    output_root: Path,
) -> dict[str, Any]:
    model_dir = output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    oof_probability = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
    test_probabilities = []
    fold_reports = []
    history_frames = []

    for fold in sorted(np.unique(folds)):
        train_index = np.flatnonzero(folds != fold)
        valid_index = np.flatnonzero(folds == fold)
        progress(
            f"{model_name} fold {fold + 1}/{len(np.unique(folds))}: "
            f"train={len(train_index):,}, valid={len(valid_index):,}"
        )
        if model_name == "catboost":
            result = train_catboost_fold(
                train_cat,
                test_cat,
                categorical_columns,
                y,
                train_index,
                valid_index,
                config,
                int(fold),
                model_dir,
            )
        elif model_name == "lightgbm":
            result = train_lightgbm_fold(
                train_encoded,
                test_encoded,
                y,
                train_index,
                valid_index,
                config,
                int(fold),
                model_dir,
            )
        elif model_name == "xgboost":
            result = train_xgboost_fold(
                train_encoded,
                test_encoded,
                y,
                train_index,
                valid_index,
                config,
                int(fold),
                model_dir,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        valid_probability, test_probability, fold_report, history_frame = result
        oof_probability[valid_index] = valid_probability
        test_probabilities.append(test_probability)
        fold_reports.append(fold_report)
        history_frames.append(history_frame)
        progress(
            f"{model_name} fold {fold + 1}: "
            f"BAC={fold_report['balanced_accuracy']:.6f}, "
            f"best_iteration={fold_report['best_iteration']}"
        )

    test_probability = normalize_probabilities(np.mean(test_probabilities, axis=0))
    if not np.allclose(oof_probability.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError(f"{model_name} OOF probability rows are invalid")
    if not np.allclose(test_probability.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError(f"{model_name} test probability rows are invalid")

    oof_prediction = oof_probability.argmax(axis=1)
    oof_score = float(balanced_accuracy_score(y, oof_prediction))
    fold_scores = pd.DataFrame(fold_reports)
    history = pd.concat(history_frames, ignore_index=True)

    np.save(model_dir / "oof_proba.npy", oof_probability)
    np.save(model_dir / "test_proba.npy", test_probability)
    fold_scores.to_csv(model_dir / "fold_scores.csv", index=False)
    history.to_csv(model_dir / "training_history.csv", index=False)

    submission = sample.copy()
    submission[TARGET_COLUMN] = np.asarray(CLASSES)[test_probability.argmax(axis=1)]
    submission.to_csv(model_dir / "submission.csv", index=False)

    report = {
        "model": model_name,
        "run_mode": config.run_mode,
        "oof_balanced_accuracy": oof_score,
        "class_recalls": class_recalls(y, oof_prediction),
        "fold_mean": float(fold_scores["balanced_accuracy"].mean()),
        "fold_std": float(fold_scores["balanced_accuracy"].std(ddof=0)),
        "class_order": CLASSES,
        "config": asdict(config),
    }
    (model_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_diagnostic_graphs(
        y,
        oof_probability,
        fold_scores,
        history,
        model_dir,
        model_name,
    )
    progress(f"{model_name} full OOF balanced_accuracy={oof_score:.6f}")
    return {
        "model": model_name,
        "oof_probability": oof_probability,
        "test_probability": test_probability,
        "report": report,
    }


def run_baseline(root: Path, config: CVConfig) -> pd.DataFrame:
    output_root = root / config.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / ".mplconfig"))

    progress("Loading competition data")
    train, test, sample = load_competition_data(root)
    audit = data_audit(train, test)
    (output_root / "data_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    progress(
        f"train={train.shape}, test={test.shape}, target_share={audit['target_share']}"
    )

    y_full = train[TARGET_COLUMN].map(TARGET_MAP).to_numpy(dtype=np.int8)
    if np.any(pd.isna(y_full)):
        raise ValueError("Unknown target label")

    if config.run_mode == "smoke":
        selected = smoke_indices(y_full, config.smoke_rows, config.seed)
        train_work = train.iloc[selected].reset_index(drop=True)
        y = y_full[selected]
        progress(f"Smoke mode selected {len(selected):,} stratified train rows")
    else:
        train_work = train.reset_index(drop=True)
        y = y_full

    folds = make_fold_assignments(y, config.n_splits, config.seed)
    fold_frame = pd.DataFrame(
        {
            ID_COLUMN: train_work[ID_COLUMN],
            TARGET_COLUMN: np.asarray(CLASSES)[y],
            "target_index": y,
            "fold": folds,
        }
    )
    fold_frame.to_csv(output_root / "fold_assignments.csv", index=False)
    progress("Building shared feature views")
    train_cat, test_cat, train_encoded, test_encoded, categorical_columns = (
        prepare_feature_views(train_work, test)
    )
    feature_manifest = {
        "catboost_columns": train_cat.columns.tolist(),
        "encoded_columns": train_encoded.columns.tolist(),
        "categorical_columns": categorical_columns,
    }
    (output_root / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2), encoding="utf-8"
    )

    results = []
    for model_name in config.models:
        results.append(
            run_model_cv(
                model_name,
                train_cat,
                test_cat,
                train_encoded,
                test_encoded,
                categorical_columns,
                y,
                folds,
                sample,
                config,
                output_root,
            )
        )

    summary_rows = [
        {
            "model": result["model"],
            "oof_balanced_accuracy": result["report"]["oof_balanced_accuracy"],
            "fold_mean": result["report"]["fold_mean"],
            "fold_std": result["report"]["fold_std"],
            **{
                f"recall_{key}": value
                for key, value in result["report"]["class_recalls"].items()
            },
        }
        for result in results
    ]

    if len(results) >= 2:
        ensemble_oof = normalize_probabilities(
            np.mean([result["oof_probability"] for result in results], axis=0)
        )
        ensemble_test = normalize_probabilities(
            np.mean([result["test_probability"] for result in results], axis=0)
        )
        ensemble_prediction = ensemble_oof.argmax(axis=1)
        ensemble_score = float(balanced_accuracy_score(y, ensemble_prediction))
        ensemble_dir = output_root / "equal_ensemble"
        ensemble_dir.mkdir(exist_ok=True)
        np.save(ensemble_dir / "oof_proba.npy", ensemble_oof)
        np.save(ensemble_dir / "test_proba.npy", ensemble_test)
        submission = sample.copy()
        submission[TARGET_COLUMN] = np.asarray(CLASSES)[ensemble_test.argmax(axis=1)]
        submission.to_csv(ensemble_dir / "submission.csv", index=False)
        ensemble_report = {
            "model": "equal_ensemble",
            "sources": [result["model"] for result in results],
            "oof_balanced_accuracy": ensemble_score,
            "class_recalls": class_recalls(y, ensemble_prediction),
            "class_order": CLASSES,
        }
        (ensemble_dir / "report.json").write_text(
            json.dumps(ensemble_report, indent=2), encoding="utf-8"
        )
        summary_rows.append(
            {
                "model": "equal_ensemble",
                "oof_balanced_accuracy": ensemble_score,
                "fold_mean": np.nan,
                "fold_std": np.nan,
                **{
                    f"recall_{key}": value
                    for key, value in ensemble_report["class_recalls"].items()
                },
            }
        )
        progress(f"equal ensemble OOF balanced_accuracy={ensemble_score:.6f}")

    summary = pd.DataFrame(summary_rows).sort_values(
        "oof_balanced_accuracy", ascending=False
    )
    summary.to_csv(output_root / "model_summary.csv", index=False)
    (output_root / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    progress(f"Completed. Artifacts: {output_root}")
    print(summary.to_string(index=False), flush=True)
    return summary

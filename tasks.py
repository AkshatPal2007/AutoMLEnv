"""
tasks.py — Task definitions, bug injection, and ground-truth configs.

Each task is a dataclass carrying:
  - A synthetic dataset (generated deterministically from seed)
  - A pipeline state dict with injected bugs
  - Ground truth for grader validation
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    task_id: int
    name: str
    description: str
    difficulty: str

    # Dataset
    X_train: np.ndarray = field(repr=False)
    X_test:  np.ndarray = field(repr=False)
    y_train: np.ndarray = field(repr=False)
    y_test:  np.ndarray = field(repr=False)
    feature_names: List[str] = field(repr=False)
    categorical_cols: List[str] = field(repr=False)
    null_cols: List[str] = field(repr=False)
    task_type: str = "classification"   # "classification" | "regression"

    # Buggy pipeline state (mutable during episode)
    pipeline_state: Dict[str, Any] = field(default_factory=dict)

    # Ground truth (used by grader — never exposed to agent)
    ground_truth: Dict[str, Any] = field(default_factory=dict)

    # Class balance info for metric_warning
    class_imbalance: bool = False
    majority_ratio: float = 0.5


# ---------------------------------------------------------------------------
# Task 1 — Easy: Broken Preprocessor
# ---------------------------------------------------------------------------

def make_task1(seed: int = 42) -> TaskConfig:
    """
    Bugs injected:
      1. No imputer  (NaN crash)
      2. Categorical column not encoded (dtype error)
      3. Scaler fitted on full dataset (data leakage — runs but scores low)
    """
    rng = _rng(seed)

    n_samples = 800
    n_features = 8

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=1,
        weights=[0.82, 0.18],   # imbalanced — triggers metric_warning
        random_state=seed,
    )

    feature_names = [f"feat_{i}" for i in range(n_features)]

    # Introduce NaNs deterministically
    null_indices = rng.choice(n_samples, size=int(n_samples * 0.08), replace=False)
    X[null_indices, 2] = np.nan
    null_cols = ["feat_2"]

    # Add a categorical column (as int codes, un-encoded)
    cat_col = rng.integers(0, 4, size=(n_samples, 1)).astype(float)
    X = np.hstack([X, cat_col])
    feature_names.append("cat_region")
    categorical_cols = ["cat_region"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    pipeline_state = {
        "imputer":          {"present": False,  "strategy": None,   "columns": []},
        "scaler":           {"type": "standard", "fit_on": "full"},   # BUG: should be train
        "encoder":          {"present": False,  "columns": []},       # BUG: not encoded
        "feature_selector": {"feature_list": feature_names},           # all features
        "model":            {"type": "random_forest_classifier", "params": {"n_estimators": 100, "max_depth": 6}},
        "evaluator":        {"strategy": "holdout", "eval_on": "test", "metric": "f1"},
    }

    ground_truth = {
        "required_imputer":   True,
        "imputer_columns":    null_cols,
        "scaler_fit_on":      "train",
        "encoded_columns":    categorical_cols,
        "correct_feature_set": feature_names,
        "target_f1":          0.65,
        "task_type":          "classification",
    }

    return TaskConfig(
        task_id=1,
        name="Fix the Broken Preprocessor",
        description=(
            "A pipeline fails during preprocessing. "
            "Diagnose and fix missing imputation, encoding errors, and data leakage."
        ),
        difficulty="easy",
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        categorical_cols=categorical_cols,
        null_cols=null_cols,
        task_type="classification",
        pipeline_state=pipeline_state,
        ground_truth=ground_truth,
        class_imbalance=True,
        majority_ratio=0.82,
    )


# ---------------------------------------------------------------------------
# Task 2 — Medium: Fix + Optimize a Failing Pipeline
# ---------------------------------------------------------------------------

def make_task2(seed: int = 42) -> TaskConfig:
    """
    Bugs injected:
      1. Wrong feature columns (irrelevant cols included, key col excluded)
      2. Degenerate hyperparams (n_estimators=1, max_depth=1)
      3. Inverted target (0/1 flipped)
      4. Eval computed on train set  ← hidden, requires inspect_step("evaluator")
    """
    rng = _rng(seed)

    n_samples = 1000
    n_features = 12

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        weights=[0.55, 0.45],
        random_state=seed,
    )

    feature_names = [f"feat_{i}" for i in range(n_features)]

    # True informative features: feat_0..feat_5. Buggy selection omits feat_0 and adds noise.
    buggy_features     = [f"feat_{i}" for i in range(1, 10)]   # missing feat_0, adds 6-9 (noise)
    correct_features   = [f"feat_{i}" for i in range(8)]       # informative set

    # Flip target
    y_flipped = 1 - y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_flipped, test_size=0.2, random_state=seed, stratify=y_flipped
    )

    pipeline_state = {
        "imputer":          {"present": True, "strategy": "mean", "columns": []},  # fine
        "scaler":           {"type": "standard", "fit_on": "train"},               # fine
        "encoder":          {"present": True,  "columns": []},                     # fine
        "feature_selector": {"feature_list": buggy_features},                      # BUG
        "model":            {"type": "random_forest_classifier",
                             "params": {"n_estimators": 1, "max_depth": 1}},       # BUG
        "evaluator":        {"strategy": "holdout", "eval_on": "train",            # BUG (hidden)
                             "metric": "accuracy"},
        "target_encoding":  {"flipped": True},                                     # BUG
    }

    ground_truth = {
        "correct_feature_set":    correct_features,
        "correct_n_estimators_min": 10,
        "correct_max_depth_min":   3,
        "target_flipped":          True,
        "eval_on":                 "test",
        "target_f1":               0.74,
        "task_type":               "classification",
    }

    return TaskConfig(
        task_id=2,
        name="Fix and Optimize a Failing Pipeline",
        description=(
            "The pipeline runs but produces terrible metrics. "
            "Find all hidden bugs including a deceptive evaluation trap."
        ),
        difficulty="medium",
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        categorical_cols=[],
        null_cols=[],
        task_type="classification",
        pipeline_state=pipeline_state,
        ground_truth=ground_truth,
        class_imbalance=False,
        majority_ratio=0.55,
    )


# ---------------------------------------------------------------------------
# Task 3 — Hard: End-to-End Pipeline Reconstruction
# ---------------------------------------------------------------------------

def make_task3(seed: int = 42) -> TaskConfig:
    """
    Cascading bugs (must fix in order):
      1. Wrong model class (classifier on regression target)
      2. Custom transformer off-by-one index
      3. Scaler fitted on full dataset (leakage)
      4. Eval on train set (inflated score illusion)
      5. Irrelevant features included

    Short-term traps:
      - Fixing eval split before fixing leakage makes leakage appear harmful
      - apply_minmax_scaler(fit_on="full") gives temporary R² boost
    """
    rng = _rng(seed)

    n_samples = 1200
    n_features = 15

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        noise=12.0,
        random_state=seed,
    )

    feature_names = [f"feat_{i}" for i in range(n_features)]
    correct_features = [f"feat_{i}" for i in range(10)]    # 0-9 informative
    noisy_features   = feature_names                        # all 15 (buggy)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    pipeline_state = {
        "imputer":    {"present": True, "strategy": "mean", "columns": []},
        "scaler":     {"type": "standard", "fit_on": "full"},          # BUG 3
        "encoder":    {"present": True, "columns": []},
        "feature_selector": {"feature_list": noisy_features},           # BUG 5
        "model":      {"type": "random_forest_classifier",              # BUG 1
                       "params": {"n_estimators": 100, "max_depth": 8}},
        "custom_transformer": {
            "name": "PolynomialExpander",
            "off_by_one": True,                                         # BUG 2
            "fixed": False,
        },
        "evaluator":  {"strategy": "holdout", "eval_on": "train",       # BUG 4
                       "metric": "r2"},
    }

    ground_truth = {
        "correct_model_class":    "random_forest_regressor",
        "scaler_fit_on":          "train",
        "eval_on":                "test",
        "correct_feature_set":    correct_features,
        "transformer_fixed":      True,
        "target_r2":              0.78,
        "task_type":              "regression",
        "fix_order":              ["model", "custom_transformer", "scaler", "evaluator", "feature_selector"],
    }

    return TaskConfig(
        task_id=3,
        name="End-to-End Pipeline Reconstruction",
        description=(
            "A heavily corrupted regression pipeline with cascading interacting bugs. "
            "Fixing one bug reveals the next. Order matters."
        ),
        difficulty="hard",
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        categorical_cols=[],
        null_cols=[],
        task_type="regression",
        pipeline_state=pipeline_state,
        ground_truth=ground_truth,
        class_imbalance=False,
        majority_ratio=0.5,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

TASK_FACTORIES = {
    1: make_task1,
    2: make_task2,
    3: make_task3,
}


def get_task(task_id: int, seed: int = 42) -> TaskConfig:
    if task_id not in TASK_FACTORIES:
        raise ValueError(f"Unknown task_id: {task_id}. Valid: {list(TASK_FACTORIES)}")
    return TASK_FACTORIES[task_id](seed=seed)

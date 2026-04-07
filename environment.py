"""
environment.py — AutoMLEnv core.

Implements step() / reset() / state() with:
  - Partial observability  (hidden step configs)
  - Dense reward shaping   (F1-anchored, step cost, penalty)
  - Action history         (trajectory reasoning support)
  - Deceptive signal layer (metric_warning, eval-on-train trap)
"""

from __future__ import annotations

import copy
import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

from models import (
    Action, ActionRecord, ActionType,
    Observation, StepResult,
)
from tasks import TaskConfig, get_task


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEP_COST       = -0.01
CLEAN_RUN_BONUS = +0.05
ERROR_FIX_BONUS = +0.02     # per error resolved
F1_IMPROVE_BONUS = +0.04    # per unit F1 delta (capped at +0.12)
REPEAT_PENALTY  = -0.10
DESTRUCTIVE_PENALTY = -0.15
KNOWN_ERROR_PENALTY = -0.03

HIDDEN_STEPS = {
    "imputer", "feature_selector", "scaler",
    "model", "evaluator", "dataset", "custom_transformer",
}


# ---------------------------------------------------------------------------
# Pipeline executor
# ---------------------------------------------------------------------------

def _execute_pipeline(task: TaskConfig) -> Tuple[Dict[str, float], List[str]]:
    """
    Run the current pipeline_state against the task dataset.
    Returns (metrics_dict, error_list).
    Metrics depend on task_type; errors are human-readable strings.
    """
    ps = task.pipeline_state
    errors: List[str] = []
    metrics: Dict[str, float] = {}

    try:
        X_tr = task.X_train.copy()
        X_te = task.X_test.copy()
        y_tr = task.y_train.copy()
        y_te = task.y_test.copy()

        # ---- Imputer ----
        imputer_cfg = ps.get("imputer", {})
        if np.isnan(X_tr).any():
            if not imputer_cfg.get("present", False):
                errors.append("ValueError: NaN values in training data — imputer not configured.")
                return metrics, errors
            strategy = imputer_cfg.get("strategy", "mean")
            imp = SimpleImputer(strategy=strategy)
            X_tr = imp.fit_transform(X_tr)
            X_te = imp.transform(X_te)

        # ---- Encoder ----
        encoder_cfg = ps.get("encoder", {})
        if task.categorical_cols:
            if not encoder_cfg.get("present", False):
                errors.append(
                    "ValueError: Categorical column 'cat_region' contains non-numeric data. "
                    "Fit a LabelEncoder or OneHotEncoder first."
                )
                return metrics, errors
            # In our synthetic data cat col is already numeric; encoding is structural fix
            pass

        # ---- Target flip (Task 2 bug) ----
        target_cfg = ps.get("target_encoding", {})
        if target_cfg.get("flipped", False):
            y_tr = 1 - y_tr
            y_te = 1 - y_te

        # ---- Feature selection ----
        feat_cfg = ps.get("feature_selector", {})
        feat_list = feat_cfg.get("feature_list", task.feature_names)
        all_names = task.feature_names
        indices = [all_names.index(f) for f in feat_list if f in all_names]
        if not indices:
            errors.append("ValueError: feature_list is empty — no features selected.")
            return metrics, errors
        X_tr = X_tr[:, indices]
        X_te = X_te[:, indices]

        # ---- Scaler ----
        scaler_cfg = ps.get("scaler", {})
        scaler_type = scaler_cfg.get("type", "standard")
        fit_on = scaler_cfg.get("fit_on", "train")

        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            fr = scaler_cfg.get("feature_range", [0, 1])
            scaler = MinMaxScaler(feature_range=tuple(fr))
        else:
            scaler = StandardScaler()

        if fit_on == "full":
            # DATA LEAKAGE: fit on train+test
            X_all = np.vstack([X_tr, X_te])
            scaler.fit(X_all)
        else:
            scaler.fit(X_tr)

        X_tr = scaler.transform(X_tr)
        X_te = scaler.transform(X_te)

        # ---- Custom transformer (Task 3) ----
        ct_cfg = ps.get("custom_transformer", {})
        if ct_cfg and not ct_cfg.get("fixed", True):
            if ct_cfg.get("off_by_one", False):
                errors.append(
                    "IndexError: PolynomialExpander — index out of bounds (off-by-one in "
                    "column selection). Call patch_custom_transformer to fix."
                )
                return metrics, errors

        # ---- Model ----
        model_cfg  = ps.get("model", {})
        model_type = model_cfg.get("type", "random_forest_classifier")
        params     = model_cfg.get("params", {})

        MODEL_MAP = {
            "random_forest_classifier": RandomForestClassifier,
            "random_forest_regressor":  RandomForestRegressor,
            "logistic_regression":      LogisticRegression,
            "ridge_regression":         Ridge,
        }

        if model_type not in MODEL_MAP:
            errors.append(f"ValueError: Unknown model type '{model_type}'.")
            return metrics, errors

        model_cls = MODEL_MAP[model_type]

        # Wrong model for task type check
        is_classifier = model_type in ("random_forest_classifier", "logistic_regression")
        if task.task_type == "regression" and is_classifier:
            errors.append(
                f"TypeError: Task requires a regressor but got classifier '{model_type}'. "
                "Use set_model_class with a regressor."
            )
            return metrics, errors
        if task.task_type == "classification" and not is_classifier:
            errors.append(
                f"TypeError: Task requires a classifier but got regressor '{model_type}'."
            )
            return metrics, errors

        model = model_cls(**params)
        model.fit(X_tr, y_tr)

        # ---- Evaluator ----
        eval_cfg = ps.get("evaluator", {})
        eval_on  = eval_cfg.get("eval_on", "test")

        X_eval = X_tr if eval_on == "train" else X_te
        y_eval = y_tr if eval_on == "train" else y_te

        if task.task_type == "classification":
            y_pred = model.predict(X_eval)
            metrics["accuracy"] = round(float(accuracy_score(y_eval, y_pred)), 4)
            metrics["f1"]       = round(float(f1_score(y_eval, y_pred, zero_division=0)), 4)
        else:
            y_pred = model.predict(X_eval)
            metrics["r2"]  = round(float(r2_score(y_eval, y_pred)), 4)
            metrics["mse"] = round(float(np.mean((y_eval - y_pred) ** 2)), 4)

    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")

    return metrics, errors


def _execute_cv(task: TaskConfig, n_folds: int) -> Tuple[Dict[str, float], List[str]]:
    """Simple cross-validation on training set."""
    from sklearn.model_selection import cross_val_score
    errors: List[str] = []
    metrics: Dict[str, float] = {}

    try:
        ps = task.pipeline_state
        model_cfg  = ps.get("model", {})
        model_type = model_cfg.get("type", "random_forest_classifier")
        params     = model_cfg.get("params", {})

        MODEL_MAP = {
            "random_forest_classifier": RandomForestClassifier,
            "random_forest_regressor":  RandomForestRegressor,
            "logistic_regression":      LogisticRegression,
            "ridge_regression":         Ridge,
        }
        if model_type not in MODEL_MAP:
            errors.append(f"ValueError: Unknown model type '{model_type}'.")
            return metrics, errors

        model = MODEL_MAP[model_type](**params)
        scoring = "f1" if task.task_type == "classification" else "r2"
        scores = cross_val_score(model, task.X_train, task.y_train,
                                 cv=n_folds, scoring=scoring)
        key = "cv_f1" if task.task_type == "classification" else "cv_r2"
        metrics[key] = round(float(scores.mean()), 4)
        metrics[f"{key}_std"] = round(float(scores.std()), 4)

    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")

    return metrics, errors


# ---------------------------------------------------------------------------
# AutoMLEnv
# ---------------------------------------------------------------------------

class AutoMLEnv:
    def __init__(self) -> None:
        self._initialized: bool = False
        self._task: Optional[TaskConfig] = None
        self._seed: int = 42
        self._steps_taken: int = 0
        self._max_steps: int = 20
        self._action_history: List[ActionRecord] = []
        self._last_metrics: Dict[str, float] = {}
        self._last_errors: List[str] = []
        self._pipeline_status: str = "idle"
        self._inspected: List[str] = []
        self._inspected_configs: Dict[str, Any] = {}
        self._metric_history: List[float] = []   # for stability bonus
        self._consecutive_same: int = 0
        self._last_action_type: Optional[str] = None
        self._episode_done: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: int, seed: int = 42) -> Observation:
        self._initialized = True
        self._task   = get_task(task_id, seed=seed)
        self._seed   = seed
        self._steps_taken    = 0
        self._action_history = []
        self._last_metrics   = {}
        self._last_errors    = []
        self._pipeline_status = "idle"
        self._inspected      = []
        self._inspected_configs = {}
        self._metric_history = []
        self._consecutive_same = 0
        self._last_action_type = None
        self._episode_done   = False
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        if not self._initialized or self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._episode_done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"message": "Episode already done. Call reset()."},
            )

        self._steps_taken += 1
        metrics_before = dict(self._last_metrics)
        errors_before  = list(self._last_errors)

        # Step cost
        reward = STEP_COST

        # Repeat-action penalty
        if action.action_type.value == self._last_action_type:
            self._consecutive_same += 1
            if self._consecutive_same >= 4:
                reward += REPEAT_PENALTY
        else:
            self._consecutive_same = 0
        self._last_action_type = action.action_type.value

        # Dispatch
        reward_delta, info = self._dispatch(action)
        reward += reward_delta

        metrics_after = dict(self._last_metrics)

        # Record history
        self._action_history.append(ActionRecord(
            step=self._steps_taken,
            action=action.action_type.value,
            args=action.args,
            reward_delta=round(reward, 4),
            metric_before=metrics_before,
            metric_after=metrics_after,
        ))

        # Check done
        done = (
            self._steps_taken >= self._max_steps
            or info.get("done", False)
        )
        self._episode_done = done

        return StepResult(
            observation=self._build_observation(),
            reward=round(reward, 4),
            done=done,
            info=info,
        )

    def state(self) -> Observation:
        if not self._initialized or self._task is None:
            raise RuntimeError("Call reset() before state().")
        return self._build_observation()

    # ------------------------------------------------------------------
    # Action dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, action: Action) -> Tuple[float, Dict[str, Any]]:
        ps   = self._task.pipeline_state
        args = action.args
        reward = 0.0
        info: Dict[str, Any] = {}

        at = action.action_type

        # -- Inspection --
        if at == ActionType.INSPECT_STEP:
            step_name = args.get("step_name", "")
            if step_name not in HIDDEN_STEPS:
                info["message"] = f"Step '{step_name}' is not a hidden step."
            else:
                config = ps.get(step_name, {})
                self._inspected_configs[step_name] = copy.deepcopy(config)
                if step_name not in self._inspected:
                    self._inspected.append(step_name)
                info["revealed"] = {step_name: config}
                info["message"] = f"Config for '{step_name}' revealed."

        # -- Preprocessing --
        elif at == ActionType.APPLY_STANDARD_SCALER:
            fit_on = args.get("fit_on", "train")
            if fit_on not in ("train", "full"):
                info["message"] = "fit_on must be 'train' or 'full'."
            else:
                ps["scaler"] = {"type": "standard", "fit_on": fit_on}
                info["message"] = f"StandardScaler configured (fit_on={fit_on})."

        elif at == ActionType.APPLY_MINMAX_SCALER:
            fr = args.get("feature_range", [0, 1])
            fit_on = args.get("fit_on", "train")
            ps["scaler"] = {"type": "minmax", "feature_range": fr, "fit_on": fit_on}
            info["message"] = f"MinMaxScaler configured (range={fr}, fit_on={fit_on})."

        elif at == ActionType.APPLY_MEAN_IMPUTER:
            cols = args.get("columns", self._task.null_cols)
            ps["imputer"] = {"present": True, "strategy": "mean", "columns": cols}
            info["message"] = f"Mean imputer configured for columns: {cols}."

        elif at == ActionType.APPLY_MEDIAN_IMPUTER:
            cols = args.get("columns", self._task.null_cols)
            ps["imputer"] = {"present": True, "strategy": "median", "columns": cols}
            info["message"] = f"Median imputer configured for columns: {cols}."

        elif at == ActionType.FIX_LABEL_ENCODING:
            cols = args.get("columns", self._task.categorical_cols)
            ps["encoder"] = {"present": True, "type": "label", "columns": cols}
            info["message"] = f"Label encoding applied to: {cols}."

        elif at == ActionType.FIX_ONEHOT_ENCODING:
            cols = args.get("columns", self._task.categorical_cols)
            ps["encoder"] = {"present": True, "type": "onehot", "columns": cols}
            info["message"] = f"OneHot encoding applied to: {cols}."

        elif at == ActionType.SET_FEATURE_COLUMNS:
            feat_list = args.get("feature_list", [])
            if not feat_list:
                reward += DESTRUCTIVE_PENALTY
                info["message"] = "Penalty: empty feature_list is destructive."
            else:
                valid = [f for f in feat_list if f in self._task.feature_names]
                ps["feature_selector"]["feature_list"] = valid
                info["message"] = f"Feature set updated: {valid}."

        # -- Model --
        elif at == ActionType.SET_MODEL_CLASS:
            model_type = args.get("model_type", "")
            params     = args.get("params", {})
            valid_types = {
                "random_forest_classifier", "random_forest_regressor",
                "logistic_regression", "ridge_regression",
            }
            if model_type not in valid_types:
                info["message"] = f"Unknown model_type '{model_type}'. Valid: {valid_types}"
            else:
                ps["model"] = {"type": model_type, "params": params}
                info["message"] = f"Model set to {model_type} with params={params}."

        elif at == ActionType.SET_MODEL_HYPERPARAMS:
            param_dict = args.get("param_dict", {})
            ps["model"]["params"].update(param_dict)
            info["message"] = f"Hyperparams updated: {param_dict}."

        elif at == ActionType.SET_EVAL_SPLIT:
            strategy  = args.get("strategy", "holdout")
            test_size = args.get("test_size", 0.2)
            eval_on   = "test"   # fixing the split always means eval on test
            ps["evaluator"].update({
                "strategy": strategy,
                "eval_on": eval_on,
                "test_size": test_size,
            })
            info["message"] = f"Eval split set: strategy={strategy}, eval_on=test."

        elif at == ActionType.SET_EVAL_METRIC:
            metric = args.get("metric_name", "f1")
            ps["evaluator"]["metric"] = metric
            info["message"] = f"Eval metric set to '{metric}'."

        # -- Execution --
        elif at == ActionType.RUN_PIPELINE:
            metrics, errors = _execute_pipeline(self._task)
            self._last_errors = errors
            errors_fixed = max(0, len(self._last_errors) - len(errors))

            if errors:
                self._pipeline_status = "error"
                # Penalty for running with known errors that weren't fixed
                if self._last_errors and len(errors) >= len(self._last_errors):
                    reward += KNOWN_ERROR_PENALTY
            else:
                self._pipeline_status = "complete"
                reward += CLEAN_RUN_BONUS
                reward += ERROR_FIX_BONUS * errors_fixed

                # F1 / R2 improvement bonus (not accuracy)
                primary_key = "f1" if self._task.task_type == "classification" else "r2"
                new_score = metrics.get(primary_key, 0.0)
                old_score = self._last_metrics.get(primary_key, 0.0)
                delta = new_score - old_score
                if delta > 0:
                    reward += min(F1_IMPROVE_BONUS * delta * 10, 0.12)

                # Track for stability bonus
                self._metric_history.append(new_score)

                # Detect deceptive signal (accuracy up, F1 down)
                if self._task.task_type == "classification":
                    acc_delta = metrics.get("accuracy", 0) - self._last_metrics.get("accuracy", 0)
                    f1_delta  = metrics.get("f1", 0)       - self._last_metrics.get("f1", 0)
                    if acc_delta > 0.02 and f1_delta < -0.01:
                        info["metric_warning"] = (
                            "Accuracy improved but F1 dropped — "
                            "possible class imbalance exploitation."
                        )

            self._last_metrics = metrics
            self._last_errors  = errors
            info["metrics"]    = metrics
            info["errors"]     = errors
            info["message"]    = "Pipeline executed."

        elif at == ActionType.RUN_CROSS_VALIDATION:
            n_folds = args.get("n_folds", 5)
            metrics, errors = _execute_cv(self._task, n_folds=n_folds)
            self._last_metrics.update(metrics)
            self._last_errors = errors
            info["message"] = f"{n_folds}-fold CV complete."
            info["metrics"] = metrics

        # -- Task 3 Advanced --
        elif at == ActionType.PATCH_CUSTOM_TRANSFORMER:
            ct = ps.get("custom_transformer", {})
            if not ct:
                info["message"] = "No custom transformer in this task."
            else:
                ct["off_by_one"] = False
                ct["fixed"]      = True
                ps["custom_transformer"] = ct
                info["message"] = "PolynomialExpander patched — off-by-one fixed."

        elif at == ActionType.FLIP_TARGET_ENCODING:
            tc = ps.get("target_encoding", {})
            currently_flipped = tc.get("flipped", False)
            ps["target_encoding"] = {"flipped": not currently_flipped}
            status = "restored" if currently_flipped else "flipped"
            info["message"] = f"Target encoding {status}."

        else:
            info["message"] = f"Unrecognised action: {at}"

        # Metric warning for class imbalance (always present if applicable)
        if self._task.class_imbalance and self._pipeline_status == "complete":
            info["metric_warning"] = (
                f"Class imbalance detected: majority class ratio "
                f"{self._task.majority_ratio:.0%}. "
                "Accuracy may be misleading — check F1."
            )

        return reward, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        # metric_warning: surface from last pipeline run if class imbalance present
        warning: Optional[str] = None
        if (
            self._task
            and self._task.class_imbalance
            and self._pipeline_status == "complete"
        ):
            warning = (
                f"Class imbalance detected: majority class ratio "
                f"{self._task.majority_ratio:.0%}. "
                "F1 is a more reliable metric than accuracy here."
            )

        return Observation(
            pipeline_status=self._pipeline_status,
            last_run_metrics=dict(self._last_metrics),
            error_log=list(self._last_errors),
            visible_steps=list(self._inspected),
            steps_taken=self._steps_taken,
            max_steps=self._max_steps,
            action_history=list(self._action_history),
            metric_warning=warning,
            inspected_configs=copy.deepcopy(self._inspected_configs),
        )

    # ------------------------------------------------------------------
    # Grader (called at end of episode)
    # ------------------------------------------------------------------

    def grade(self) -> Dict[str, Any]:
        """
        Compute final score for the current episode.
        Re-evaluates the pipeline with a clean evaluator — cannot be spoofed
        by inflated train metrics.
        """
        if self._task is None:
            return {"score": 0.0, "breakdown": {}, "message": "No active task."}

        task = self._task
        ps   = task.pipeline_state
        gt   = task.ground_truth
        breakdown: Dict[str, Any] = {}
        base = 0.0

        # Force eval on test for grader — grader is immune to eval_on=train bug
        saved_eval = ps.get("evaluator", {}).get("eval_on", "test")
        ps["evaluator"]["eval_on"] = "test"
        metrics, errors = _execute_pipeline(task)
        ps["evaluator"]["eval_on"] = saved_eval   # restore

        primary_key = "f1" if task.task_type == "classification" else "r2"
        primary_score = metrics.get(primary_key, 0.0) if not errors else 0.0

        if task.task_id == 1:
            base, breakdown = self._grade_task1(ps, gt, errors, primary_score, metrics)
        elif task.task_id == 2:
            base, breakdown = self._grade_task2(ps, gt, errors, primary_score, metrics)
        else:
            base, breakdown = self._grade_task3(ps, gt, errors, primary_score, metrics)

        # Efficiency bonus
        eff_weight = {1: 0.10, 2: 0.10, 3: 0.12}[task.task_id]
        efficiency_bonus = round(
            max(0.0, eff_weight * (1 - self._steps_taken / self._max_steps)), 4
        )
        breakdown["efficiency_bonus"] = efficiency_bonus

        # Stability bonus (low variance in last N metric values)
        stab_weight = {1: 0.05, 2: 0.05, 3: 0.08}[task.task_id]
        n_hist = {1: 3, 2: 3, 3: 5}[task.task_id]
        if len(self._metric_history) >= 2:
            recent = self._metric_history[-n_hist:]
            std = statistics.stdev(recent) if len(recent) > 1 else 0.0
            stability_bonus = round(max(0.0, stab_weight * (1 - std)), 4)
        else:
            stability_bonus = 0.0
        breakdown["stability_bonus"] = stability_bonus

        total = round(min(base + efficiency_bonus + stability_bonus, 1.15), 4)
        breakdown["base_score"] = round(base, 4)
        breakdown["total_score"] = total
        breakdown["grader_metrics"] = metrics
        breakdown["grader_errors"]  = errors

        return {
            "score":            total,
            "breakdown":        breakdown,
            "efficiency_bonus": efficiency_bonus,
            "stability_bonus":  stability_bonus,
        }

    # ------------------------------------------------------------------
    # Per-task grader helpers
    # ------------------------------------------------------------------

    def _grade_task1(self, ps, gt, errors, primary_score, metrics):
        b = 0.0
        bd = {}

        bd["pipeline_runs"] = not bool(errors)
        if not errors:         b += 0.25

        bd["f1_threshold"] = primary_score >= gt["target_f1"]
        if primary_score >= gt["target_f1"]: b += 0.25

        scaler_ok = ps.get("scaler", {}).get("fit_on") == "train"
        bd["no_leakage"] = scaler_ok
        if scaler_ok: b += 0.20

        imputer_ok = ps.get("imputer", {}).get("present", False)
        bd["imputer_present"] = imputer_ok
        if imputer_ok: b += 0.15

        encoder_ok = ps.get("encoder", {}).get("present", False)
        bd["encoder_present"] = encoder_ok
        if encoder_ok: b += 0.15

        return b, bd

    def _grade_task2(self, ps, gt, errors, primary_score, metrics):
        b = 0.0
        bd = {}

        bd["pipeline_runs"] = not bool(errors)
        if not errors: b += 0.20

        bd["f1_threshold"] = primary_score >= gt["target_f1"]
        if primary_score >= gt["target_f1"]: b += 0.30

        current_feats = set(ps.get("feature_selector", {}).get("feature_list", []))
        correct_feats = set(gt["correct_feature_set"])
        overlap = len(current_feats & correct_feats) / max(len(correct_feats), 1)
        bd["feature_quality"] = round(overlap, 3)
        if overlap >= 0.85: b += 0.20

        params = ps.get("model", {}).get("params", {})
        hp_ok = (
            params.get("n_estimators", 0) >= gt["correct_n_estimators_min"]
            and params.get("max_depth", 0)  >= gt["correct_max_depth_min"]
        )
        bd["hyperparams_ok"] = hp_ok
        if hp_ok: b += 0.15

        eval_ok = ps.get("evaluator", {}).get("eval_on") == "test"
        bd["eval_on_test"] = eval_ok
        if eval_ok: b += 0.15

        return b, bd

    def _grade_task3(self, ps, gt, errors, primary_score, metrics):
        b = 0.0
        bd = {}

        bd["pipeline_runs"] = not bool(errors)
        if not errors: b += 0.15

        bd["r2_threshold"] = primary_score >= gt["target_r2"]
        if primary_score >= gt["target_r2"]: b += 0.25

        model_ok = ps.get("model", {}).get("type") == gt["correct_model_class"]
        bd["correct_model_class"] = model_ok
        if model_ok: b += 0.15

        scaler_ok = ps.get("scaler", {}).get("fit_on") == gt["scaler_fit_on"]
        bd["no_leakage"] = scaler_ok
        if scaler_ok: b += 0.15

        eval_ok = ps.get("evaluator", {}).get("eval_on") == gt["eval_on"]
        bd["eval_on_test"] = eval_ok
        if eval_ok: b += 0.10

        ct_ok = ps.get("custom_transformer", {}).get("fixed", False)
        bd["transformer_fixed"] = ct_ok
        if ct_ok: b += 0.10

        current_feats = set(ps.get("feature_selector", {}).get("feature_list", []))
        correct_feats = set(gt["correct_feature_set"])
        overlap = len(current_feats & correct_feats) / max(len(correct_feats), 1)
        bd["feature_quality"] = round(overlap, 3)
        if overlap >= 0.80: b += 0.10

        return b, bd

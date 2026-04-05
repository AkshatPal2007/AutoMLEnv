"""
main.py — AutoMLEnv FastAPI server.

Endpoints required by OpenEnv spec + competition:
  POST /reset          → Observation
  POST /step           → StepResult
  GET  /state          → Observation
  GET  /tasks          → task list + action schema
  POST /grader         → GraderResult
  POST /baseline       → baseline scores across all 3 tasks
"""

from __future__ import annotations
from typing import Optional

import subprocess
import sys
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from environment import AutoMLEnv
from models import (
    Action, GraderResult, Observation,
    ResetRequest, StepResult,
)

app = FastAPI(
    title="AutoMLEnv",
    description=(
        "OpenEnv-compliant environment where an LLM agent debugs broken "
        "ML pipelines across 3 difficulty levels."
    ),
    version="1.0.0",
)

# Single global env instance (one episode at a time)
_env = AutoMLEnv()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "env": "AutoMLEnv", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None) -> Observation:
    """
    Start a new episode for task_id (1, 2, or 3).
    Returns the initial observation.
    """
    try:
        task_id = req.task_id if req else 1
        seed = req.seed if req else 42

        obs = _env.reset(task_id=task_id, seed=seed)
        return obs
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------

@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    """
    Execute one action in the current episode.
    Returns observation, reward, done, info.
    """
    try:
        result = _env.step(action)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------

@app.get("/state", response_model=Observation)
def state() -> Observation:
    """Return the current observation without advancing the episode."""
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /tasks
# ---------------------------------------------------------------------------

@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    """
    Returns task metadata and the action schema (fields required per step).
    Required by competition spec.
    """
    return {
        "tasks": [
            {
                "task_id": 1,
                "name": "Fix the Broken Preprocessor",
                "difficulty": "easy",
                "description": (
                    "A pipeline fails during preprocessing. Diagnose and fix "
                    "missing imputation, encoding errors, and data leakage."
                ),
                "task_type": "classification",
                "expected_score_range": [0.0, 1.0],
            },
            {
                "task_id": 2,
                "name": "Fix and Optimize a Failing Pipeline",
                "difficulty": "medium",
                "description": (
                    "The pipeline runs but produces terrible metrics. "
                    "Find all hidden bugs including a deceptive evaluation trap."
                ),
                "task_type": "classification",
                "expected_score_range": [0.0, 1.0],
            },
            {
                "task_id": 3,
                "name": "End-to-End Pipeline Reconstruction",
                "difficulty": "hard",
                "description": (
                    "A heavily corrupted regression pipeline with cascading interacting bugs. "
                    "Fixing one bug reveals the next. Order matters."
                ),
                "task_type": "regression",
                "expected_score_range": [0.0, 1.0],
            },
        ],
        "action_schema": {
            "required_fields": ["action_type"],
            "optional_fields": ["args"],
            "action_types": [
                {"name": "inspect_step",           "args": {"step_name": "str"}},
                {"name": "apply_standard_scaler",  "args": {"fit_on": "train|full"}},
                {"name": "apply_minmax_scaler",    "args": {"feature_range": "[min,max]", "fit_on": "train|full"}},
                {"name": "apply_mean_imputer",     "args": {"columns": "[str]"}},
                {"name": "apply_median_imputer",   "args": {"columns": "[str]"}},
                {"name": "fix_label_encoding",     "args": {"columns": "[str]"}},
                {"name": "fix_onehot_encoding",    "args": {"columns": "[str]"}},
                {"name": "set_feature_columns",    "args": {"feature_list": "[str]"}},
                {"name": "set_model_class",        "args": {"model_type": "str", "params": "dict"}},
                {"name": "set_model_hyperparams",  "args": {"param_dict": "dict"}},
                {"name": "set_eval_split",         "args": {"strategy": "holdout|cv", "test_size": "float"}},
                {"name": "set_eval_metric",        "args": {"metric_name": "str"}},
                {"name": "run_pipeline",           "args": {}},
                {"name": "run_cross_validation",   "args": {"n_folds": "int"}},
                {"name": "patch_custom_transformer","args": {"transformer_name": "str", "fix_type": "str"}},
                {"name": "flip_target_encoding",   "args": {}},
            ],
        },
        "observation_schema": {
            "always_visible": [
                "pipeline_status", "last_run_metrics", "error_log",
                "visible_steps", "steps_taken", "max_steps",
                "action_history", "metric_warning", "inspected_configs",
            ],
            "hidden_until_inspected": [
                "imputer", "feature_selector", "scaler",
                "model", "evaluator", "dataset", "custom_transformer",
            ],
        },
    }


# ---------------------------------------------------------------------------
# POST /grader
# ---------------------------------------------------------------------------

@app.post("/grader", response_model=GraderResult)
def grader() -> GraderResult:
    """
    Score the current episode state.
    Re-evaluates on held-out test set — immune to eval_on=train bug.
    Call at any point during or after an episode.
    """
    try:
        result = _env.grade()
        return GraderResult(
            score=result["score"],
            breakdown=result["breakdown"],
            efficiency_bonus=result["efficiency_bonus"],
            stability_bonus=result["stability_bonus"],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /baseline
# ---------------------------------------------------------------------------

@app.post("/baseline")
def baseline() -> Dict[str, Any]:
    """
    Trigger the baseline inference script across all 3 tasks.
    Returns per-task scores and aggregate.
    Required by competition spec.
    """
    try:
        # Import and run directly (same process) for speed
        # inference.py must be in the same directory or on PYTHONPATH
        import importlib.util, os

        inference_path = os.path.join(os.path.dirname(__file__), "inference.py")
        spec = importlib.util.spec_from_file_location("inference", inference_path)
        inference_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference_mod)
        scores = inference_mod.main()
        return {"status": "ok", "scores": scores}

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="inference.py not found. Place it in the project root.",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

"""
models.py — Typed Pydantic models for OpenEnv compliance.
All action/observation/reward types live here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    # Inspection
    INSPECT_STEP              = "inspect_step"
    # Preprocessing
    APPLY_STANDARD_SCALER     = "apply_standard_scaler"
    APPLY_MINMAX_SCALER       = "apply_minmax_scaler"
    APPLY_MEAN_IMPUTER        = "apply_mean_imputer"
    APPLY_MEDIAN_IMPUTER      = "apply_median_imputer"
    FIX_LABEL_ENCODING        = "fix_label_encoding"
    FIX_ONEHOT_ENCODING       = "fix_onehot_encoding"
    SET_FEATURE_COLUMNS       = "set_feature_columns"
    # Model
    SET_MODEL_CLASS           = "set_model_class"
    SET_MODEL_HYPERPARAMS     = "set_model_hyperparams"
    SET_EVAL_SPLIT            = "set_eval_split"
    SET_EVAL_METRIC           = "set_eval_metric"
    # Execution
    RUN_PIPELINE              = "run_pipeline"
    RUN_CROSS_VALIDATION      = "run_cross_validation"
    # Task-3 advanced
    PATCH_CUSTOM_TRANSFORMER  = "patch_custom_transformer"
    FLIP_TARGET_ENCODING      = "flip_target_encoding"


class Action(BaseModel):
    action_type: ActionType
    args: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class ActionRecord(BaseModel):
    step: int
    action: str
    args: Dict[str, Any]
    reward_delta: float
    metric_before: Dict[str, float]
    metric_after: Dict[str, float]


class Observation(BaseModel):
    # Always visible
    pipeline_status: Literal["error", "running", "complete", "idle"]
    last_run_metrics: Dict[str, float] = Field(default_factory=dict)
    error_log: List[str] = Field(default_factory=list)
    visible_steps: List[str] = Field(default_factory=list)
    steps_taken: int = 0
    max_steps: int = 20
    action_history: List[ActionRecord] = Field(default_factory=list)
    metric_warning: Optional[str] = None

    # Revealed by inspect_step — populated after inspection
    inspected_configs: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Grader result
# ---------------------------------------------------------------------------

class GraderResult(BaseModel):
    score: float
    breakdown: Dict[str, Any] = Field(default_factory=dict)
    efficiency_bonus: float = 0.0
    stability_bonus: float = 0.0


# ---------------------------------------------------------------------------
# Reset request
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: int = Field(ge=1, le=3)
    seed: int = 42

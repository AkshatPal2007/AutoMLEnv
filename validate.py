#!/usr/bin/env python3
"""
validate.py — Pre-submission validation script for AutoMLEnv.

Run this before submitting to catch all disqualification criteria.
Mirrors what the competition's automated validator will check.

Usage:
    python validate.py                    # validates against localhost:8000
    python validate.py --url <space_url>  # validates against deployed HF Space
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Tuple

import requests
import yaml

BASE_URL = "http://localhost:8000"
PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
WARN = "\033[93m  WARN\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"{status}  {label}{suffix}")
    return condition


def post(path: str, body: Dict = None, timeout: int = 15) -> requests.Response:
    return requests.post(f"{BASE_URL}{path}", json=body or {}, timeout=timeout)


def get(path: str, timeout: int = 10) -> requests.Response:
    return requests.get(f"{BASE_URL}{path}", timeout=timeout)


# ---------------------------------------------------------------------------
# Check groups
# ---------------------------------------------------------------------------

def check_deployment() -> bool:
    """HF Space deploys — must return 200 and respond to reset()."""
    print("\n── Deployment")
    try:
        r = get("/")
        ok1 = check("GET / returns 200", r.status_code == 200)
        ok2 = check("Response has status field", "status" in r.json())
    except Exception as exc:
        check("Server reachable", False, str(exc))
        return False

    try:
        r = post("/reset", {"task_id": 1, "seed": 42})
        ok3 = check("POST /reset returns 200", r.status_code == 200)
    except Exception as exc:
        ok3 = check("POST /reset reachable", False, str(exc))

    return ok1 and ok2 and ok3


def check_openenv_spec() -> bool:
    """OpenEnv spec compliance — typed models, step/reset/state endpoints, openenv.yaml."""
    print("\n── OpenEnv spec")
    results = []

    # openenv.yaml
    try:
        with open("openenv.yaml") as f:
            spec = yaml.safe_load(f)
        results.append(check("openenv.yaml present and valid YAML", True))
        results.append(check("openenv.yaml has 'name' field", "name" in spec))
        results.append(check("openenv.yaml has 'tasks' field", "tasks" in spec))
        results.append(check("openenv.yaml has 'api' field", "api" in spec))
    except FileNotFoundError:
        results.append(check("openenv.yaml present", False, "file not found"))
    except Exception as exc:
        results.append(check("openenv.yaml valid", False, str(exc)))

    # reset() returns typed observation
    try:
        r = post("/reset", {"task_id": 1, "seed": 42})
        obs = r.json()
        required_obs_fields = [
            "pipeline_status", "last_run_metrics", "error_log",
            "visible_steps", "steps_taken", "max_steps", "action_history",
        ]
        for field in required_obs_fields:
            results.append(check(f"Observation has '{field}'", field in obs))
    except Exception as exc:
        results.append(check("reset() returns valid Observation", False, str(exc)))

    # step() returns StepResult
    try:
        r = post("/step", {"action_type": "run_pipeline", "args": {}})
        sr = r.json()
        results.append(check("StepResult has 'observation'", "observation" in sr))
        results.append(check("StepResult has 'reward'", "reward" in sr))
        results.append(check("StepResult has 'done'", "done" in sr))
        results.append(check("StepResult has 'info'", "info" in sr))
    except Exception as exc:
        results.append(check("step() returns valid StepResult", False, str(exc)))

    # state() endpoint
    try:
        r = get("/state")
        results.append(check("GET /state returns 200", r.status_code == 200))
        results.append(check("GET /state returns Observation", "pipeline_status" in r.json()))
    except Exception as exc:
        results.append(check("GET /state reachable", False, str(exc)))

    return all(results)


def check_tasks_and_graders() -> bool:
    """3+ tasks with graders — enumerate, run grader, verify scores 0.0–1.0."""
    print("\n── Tasks and graders")
    results = []

    try:
        r = get("/tasks")
        results.append(check("GET /tasks returns 200", r.status_code == 200))
        data = r.json()
        tasks = data.get("tasks", [])
        results.append(check("At least 3 tasks defined", len(tasks) >= 3, f"found {len(tasks)}"))

        has_action_schema = "action_schema" in data
        results.append(check("/tasks returns action_schema", has_action_schema))

        has_obs_schema = "observation_schema" in data
        results.append(check("/tasks returns observation_schema", has_obs_schema))

        difficulties = [t.get("difficulty") for t in tasks]
        results.append(check(
            "Tasks span easy → medium → hard",
            "easy" in difficulties and "medium" in difficulties and "hard" in difficulties,
            str(difficulties),
        ))
    except Exception as exc:
        results.append(check("GET /tasks reachable", False, str(exc)))
        return False

    # Run grader for each task — score must be in [0.0, 1.0]
    for task_id in [1, 2, 3]:
        try:
            post("/reset", {"task_id": task_id, "seed": 42})
            post("/step", {"action_type": "run_pipeline", "args": {}})
            r = post("/grader")
            results.append(check(f"POST /grader returns 200 (task {task_id})", r.status_code == 200))
            score = r.json().get("score", -1)
            in_range = 0.0 <= score <= 1.0
            results.append(check(
                f"Grader score in [0.0, 1.0] (task {task_id})",
                in_range,
                f"got {score}",
            ))
            # Graders must NOT always return the same score
        except Exception as exc:
            results.append(check(f"Grader reachable (task {task_id})", False, str(exc)))

    # Verify graders don't always return same score (disqualification criterion)
    try:
        post("/reset", {"task_id": 1, "seed": 42})
        post("/step", {"action_type": "run_pipeline", "args": {}})
        r_broken = post("/grader")
        score_broken = r_broken.json().get("score", 0)

        post("/reset", {"task_id": 1, "seed": 42})
        post("/step", {"action_type": "apply_mean_imputer", "args": {"columns": ["feat_2"]}})
        post("/step", {"action_type": "fix_label_encoding", "args": {"columns": ["cat_region"]}})
        post("/step", {"action_type": "apply_standard_scaler", "args": {"fit_on": "train"}})
        post("/step", {"action_type": "run_pipeline", "args": {}})
        r_fixed = post("/grader")
        score_fixed = r_fixed.json().get("score", 0)

        results.append(check(
            "Grader varies with pipeline state (not constant)",
            abs(score_fixed - score_broken) > 0.01,
            f"broken={score_broken}, fixed={score_fixed}",
        ))
    except Exception as exc:
        results.append(check("Grader variance check", False, str(exc)))

    return all(results)


def check_additional_endpoints() -> bool:
    """Additional endpoints: /baseline, /grader, /tasks."""
    print("\n── Additional endpoints")
    results = []

    # /grader
    try:
        post("/reset", {"task_id": 1, "seed": 42})
        r = post("/grader")
        results.append(check("POST /grader endpoint present", r.status_code == 200))
        data = r.json()
        results.append(check("/grader returns 'score'",     "score"     in data))
        results.append(check("/grader returns 'breakdown'", "breakdown" in data))
    except Exception as exc:
        results.append(check("POST /grader", False, str(exc)))

    # /baseline — may be slow (runs inference), just check it responds
    try:
        r = post("/baseline", timeout=60)
        # Accept 200 or 500 (if no API key configured) — presence of endpoint matters
        results.append(check(
            "POST /baseline endpoint present",
            r.status_code in (200, 500),
            f"status={r.status_code}",
        ))
        if r.status_code == 200:
            data = r.json()
            results.append(check("/baseline returns 'scores'", "scores" in data))
    except requests.Timeout:
        results.append(check("POST /baseline responds (may be slow)", False, "timeout >60s"))
    except Exception as exc:
        results.append(check("POST /baseline reachable", False, str(exc)))

    return all(results)


def check_baseline_script() -> bool:
    """Baseline inference script — must exist and be importable."""
    print("\n── Baseline inference script")
    results = []

    import os
    results.append(check("inference.py exists in project root", os.path.exists("inference.py")))

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("inference", "inference.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        results.append(check("inference.py importable without error", True))
        results.append(check("inference.py has main() function", hasattr(mod, "main")))
        results.append(check("inference.py has TASK_IDS", hasattr(mod, "TASK_IDS")))
        results.append(check(
            "TASK_IDS covers all 3 tasks",
            hasattr(mod, "TASK_IDS") and set(mod.TASK_IDS) == {1, 2, 3},
        ))
    except Exception as exc:
        results.append(check("inference.py loads cleanly", False, str(exc)))

    return all(results)


def check_reproducibility() -> bool:
    """Same seed → same score on every run."""
    print("\n── Reproducibility")
    results = []

    try:
        scores: List[float] = []
        for _ in range(2):
            post("/reset", {"task_id": 1, "seed": 42})
            post("/step", {"action_type": "apply_mean_imputer", "args": {"columns": ["feat_2"]}})
            post("/step", {"action_type": "run_pipeline", "args": {}})
            r = post("/grader")
            scores.append(r.json().get("score", -1))

        results.append(check(
            "Identical seed → identical score",
            scores[0] == scores[1],
            f"run1={scores[0]}, run2={scores[1]}",
        ))

        # Different seed → potentially different score (soft check)
        post("/reset", {"task_id": 1, "seed": 99})
        post("/step", {"action_type": "run_pipeline", "args": {}})
        r2 = post("/grader")
        score_seed99 = r2.json().get("score", -1)
        results.append(check(
            "Different seeds accepted without error",
            score_seed99 >= 0.0,
            f"seed=99 score={score_seed99}",
        ))

    except Exception as exc:
        results.append(check("Reproducibility test", False, str(exc)))

    return all(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AutoMLEnv pre-submission validator")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the running environment")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url.rstrip("/")

    print(f"\n{'='*60}")
    print(f"  AutoMLEnv Pre-Submission Validator")
    print(f"  Target: {BASE_URL}")
    print(f"{'='*60}")

    checks: List[Tuple[str, bool]] = [
        ("Deployment",           check_deployment()),
        ("OpenEnv spec",         check_openenv_spec()),
        ("Tasks & graders",      check_tasks_and_graders()),
        ("Additional endpoints", check_additional_endpoints()),
        ("Baseline script",      check_baseline_script()),
        ("Reproducibility",      check_reproducibility()),
    ]

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    all_passed = True
    for name, passed in checks:
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print(f"{'='*60}")
    if all_passed:
        print("  \033[92mALL CHECKS PASSED — ready to submit.\033[0m")
    else:
        print("  \033[91mSOME CHECKS FAILED — fix before submitting.\033[0m")
    print(f"{'='*60}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

"""
inference.py — AutoMLEnv Baseline Inference Script
===================================================
MANDATORY environment variables (injected by the evaluation platform):
    API_BASE_URL   The LiteLLM proxy endpoint
    API_KEY        The API key for the proxy
    MODEL_NAME     Model identifier (default: meta-llama/Llama-3.1-8B-Instruct)

Runs the baseline agent against all 3 AutoMLEnv tasks and prints per-task
scores plus an aggregate.  Also called by the /baseline FastAPI endpoint.

STDOUT FORMAT (strict — parsed by the evaluator):
    [START] task=<name> env=openenv model=<model>
    [STEP]  step=<n> action=<type> reward=<.2f> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<.3f> rewards=<comma_sep>
"""

import os
import json
import textwrap

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8000")

MAX_STEPS:   int   = 20
TEMPERATURE: float = 0.0
MAX_TOKENS:  int   = 512
TASK_IDS:    list  = [1, 2, 3]
SEED:        int   = 42

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert ML engineer debugging broken machine learning pipelines.

    You interact with an AutoMLEnv environment through a structured action API.
    On each step you receive a JSON observation and must respond with exactly one
    JSON action object — nothing else.

    === OBSERVATION FIELDS ===
    - pipeline_status     : "error" | "running" | "complete"
    - last_run_metrics    : dict of metric_name -> float (e.g. {"accuracy": 0.71, "f1": 0.44})
    - error_log           : list of error strings from the last run
    - visible_steps       : list of step names whose configs have been inspected
    - steps_taken         : int
    - max_steps           : int
    - action_history      : list of {step, action, args, reward_delta, metric_before, metric_after}
    - metric_warning      : optional string warning (e.g. class imbalance notice)

    === ACTION FORMAT ===
    Respond with a JSON object:
    {
        "action_type": "<action name>",
        "args": { <key>: <value>, ... }
    }

    === AVAILABLE ACTIONS (cost -0.01 each) ===
    Inspection:
        inspect_step            args: {"step_name": str}

    Preprocessing:
        apply_standard_scaler   args: {"fit_on": "train"|"full"}
        apply_minmax_scaler     args: {"feature_range": [min, max]}
        apply_mean_imputer      args: {"columns": [str, ...]}
        apply_median_imputer    args: {"columns": [str, ...]}
        fix_label_encoding      args: {"columns": [str, ...]}
        fix_onehot_encoding     args: {"columns": [str, ...]}
        set_feature_columns     args: {"feature_list": [str, ...]}

    Model:
        set_model_class         args: {"model_type": str, "params": dict}
        set_model_hyperparams   args: {"param_dict": dict}
        set_eval_split          args: {"strategy": "holdout"|"cv", "test_size": float}

    Execution:
        run_pipeline            args: {}
        run_cross_validation    args: {"n_folds": int}

    Task-3 only:
        patch_custom_transformer  args: {"transformer_name": str, "fix_type": str}
        flip_target_encoding      args: {}
        set_eval_metric           args: {"metric_name": str}

    === STRATEGY GUIDELINES ===
    1. Always inspect hidden steps before trying to fix them.
       Hidden steps: imputer, feature_selector, scaler, model, evaluator, dataset.
    2. Prefer F1 over accuracy — accuracy on imbalanced data is misleading.
    3. Check metric_warning — it signals deceptive conditions.
    4. After every fix, run_pipeline to observe the effect.
    5. Inspect evaluator early — eval on train set is a common hidden bug.
    6. Be efficient: fewer steps = higher bonus score.
    7. If F1 improves but then drops after a later fix, you likely introduced
       or uncovered another bug — keep debugging.
    8. In task 3, fix order matters: model class → transformer → scaler → eval → features.

    Return ONLY valid JSON. No explanation, no markdown, no extra text.
""").strip()


# ---------------------------------------------------------------------------
# Strict log functions (ONLY these print to stdout)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Environment client helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: int, seed: int = SEED) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action_type: str, args: dict) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action_type": action_type, "args": args},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_grader() -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/grader", timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# OpenAI client — uses ONLY evaluator-injected env vars
# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    """Build OpenAI client strictly from injected env vars. No fallbacks."""
    return OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def build_user_prompt(observation: dict) -> str:
    """Serialize the observation into a clear prompt for the LLM."""
    metrics = observation.get("last_run_metrics", {})
    metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()) or "none yet"

    errors = observation.get("error_log", [])
    errors_str = "\n    ".join(errors) if errors else "none"

    warning = observation.get("metric_warning")
    warning_str = f"\n⚠ WARNING: {warning}" if warning else ""

    history = observation.get("action_history", [])
    if history:
        recent = history[-5:]
        history_lines = []
        for h in recent:
            delta = h.get("reward_delta", 0)
            before_f1 = h.get("metric_before", {}).get("f1", "?")
            after_f1  = h.get("metric_after",  {}).get("f1", "?")
            history_lines.append(
                f"  step {h['step']}: {h['action']}({h.get('args', {})}) "
                f"-> reward_delta={delta:+.3f}, f1 {before_f1} -> {after_f1}"
            )
        history_str = "\n".join(history_lines)
    else:
        history_str = "  none"

    visible = observation.get("visible_steps", [])
    visible_str = ", ".join(visible) if visible else "none inspected yet"

    return textwrap.dedent(f"""
        === CURRENT OBSERVATION ===
        Status        : {observation.get("pipeline_status", "unknown")}
        Metrics       : {metrics_str}{warning_str}
        Errors        :
            {errors_str}
        Visible steps : {visible_str}
        Steps taken   : {observation.get("steps_taken", 0)} / {observation.get("max_steps", MAX_STEPS)}

        Recent action history:
        {history_str}

        What is your next action? Respond with exactly one JSON object.
    """).strip()


def parse_action(response_text: str) -> tuple[str, dict]:
    """Extract action_type and args from the LLM response."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        parsed = json.loads(text)
        action_type = str(parsed.get("action_type", "")).strip()
        args = parsed.get("args", {})
        if not isinstance(args, dict):
            args = {}
        if action_type:
            return action_type, args
    except (json.JSONDecodeError, AttributeError):
        pass

    # Parse failure — return inspect_step as a safe action
    return "inspect_step", {"step_name": "dataset"}


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_id: int) -> float:
    """
    Run one full episode for the given task.
    Returns the grader score [0.0, 1.0+].
    """
    task_name = f"task_{task_id}"
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

    rewards_list: list[float] = []
    total_steps = 0
    success = False

    log_start(task=task_name, env="openenv", model=model_name)

    try:
        # Reset environment
        result = env_reset(task_id, seed=SEED)
        # reset returns Observation directly (not wrapped)
        if "observation" in result:
            observation = result["observation"]
        else:
            observation = result

        conversation: list = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        done = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            user_prompt = build_user_prompt(observation)
            conversation.append({"role": "user", "content": user_prompt})

            # LLM call — MUST go through the proxy, no fallback
            completion = client.chat.completions.create(
                model=model_name,
                messages=conversation,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""

            # Keep assistant turn in conversation for multi-turn context
            conversation.append({"role": "assistant", "content": response_text})

            action_type, args = parse_action(response_text)

            # Execute action in environment
            result = env_step(action_type, args)

            observation = result.get("observation", {})
            reward      = float(result.get("reward", 0.0))
            done        = result.get("done", False)
            total_steps = step

            rewards_list.append(reward)

            # Get last error for log
            error_log = observation.get("error_log", [])
            last_error = error_log[-1] if error_log else None

            log_step(
                step=step,
                action=action_type,
                reward=reward,
                done=done,
                error=last_error,
            )

        # Get grader score
        grader_result = env_grader()
        score = float(grader_result.get("score", 0.0))
        success = True

    except Exception:
        score = 0.0
        success = False

    finally:
        log_end(
            success=success,
            steps=total_steps,
            score=score,
            rewards=rewards_list,
        )

    return score


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> dict[str, float]:
    """
    Run the baseline agent across all 3 tasks.
    Returns a dict of scores suitable for the /baseline endpoint.
    """
    client = build_client()
    scores: dict[str, float] = {}

    for task_id in TASK_IDS:
        task_key = f"task_{task_id}"
        score = run_episode(client, task_id)
        scores[task_key] = round(score, 4)

    aggregate = round(sum(scores.values()) / max(1, len(scores)), 4)
    scores["aggregate"] = aggregate

    return scores


if __name__ == "__main__":
    main()

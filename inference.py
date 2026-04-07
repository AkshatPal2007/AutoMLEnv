"""
inference.py — AutoMLEnv Baseline Inference Script
===================================================
MANDATORY environment variables:
    API_BASE_URL   The LLM API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier
    HF_TOKEN       Hugging Face / API key

Runs the baseline agent against all 3 AutoMLEnv tasks and prints per-task
scores plus an aggregate.  Also called by the /baseline FastAPI endpoint.

Uses the OpenAI client per competition rules.
"""

import os
import json
import textwrap

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str | None = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME: str   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
LOCAL_IMAGE_NAME: str | None = os.getenv("LOCAL_IMAGE_NAME")

ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8000")

MAX_STEPS:   int   = 20       # per task episode
TEMPERATURE: float = 0.0      # deterministic for reproducibility
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
# Environment client helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: int, seed: int = SEED) -> dict:
    try:
        resp = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {
            "observation": {"pipeline_status": "error", "error_log": [f"Network Error: {str(e)}"]},
            "done": True,
            "reward": 0.0
        }

def env_step(action_type: str, args: dict) -> dict:
    try:
        resp = requests.post(
            f"{ENV_BASE_URL}/step",
            json={"action_type": action_type, "args": args},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {
            "observation": {"pipeline_status": "error", "error_log": [f"Network Error: {str(e)}"]},
            "done": True,
            "reward": 0.0
        }

def env_grader() -> dict:
    try:
        resp = requests.post(f"{ENV_BASE_URL}/grader", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"score": 0.0, "breakdown": {"error": f"Network Error: {str(e)}" }}

def build_client() -> OpenAI | None:
    if not API_KEY:
        print("  [WARN] Missing HF_TOKEN/API_KEY. Running in fallback mode.")
        return None
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"  [WARN] Failed to initialize OpenAI client: {e}")
        return None


def log_start(task_name: str) -> None:
    print(f"[START] task={task_name}", flush=True)


def log_step(step_num: int, reward: float) -> None:
    print(f"[STEP] step={step_num} reward={reward}", flush=True)


def log_end(task_name: str, final_score: float, total_steps: int) -> None:
    print(
        f"[END] task={task_name} score={final_score} steps={total_steps}",
        flush=True,
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
    """
    Extract action_type and args from the LLM response.
    Returns (action_type, args) or falls back to inspect_step("dataset").
    """
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

    # Last-resort fallback
    print(f"  [WARN] Could not parse action from response: {response_text!r:.120}")
    return "inspect_step", {"step_name": "dataset"}


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI | None, task_id: int) -> float:
    """
    Run one full episode for the given task.
    Returns the grader score [0.0, 1.0+].
    """
    task_name = f"task_{task_id}"
    log_start(task_name)

    try:
        result = env_reset(task_id, seed=SEED)
        observation: dict = result.get("observation", result)
    except Exception as exc:
        observation = {"error_log": [str(exc)]}
        result = {"done": True, "observation": observation}

    conversation: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    done = result.get("done", False)
    total_steps = 0

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        user_prompt = build_user_prompt(observation)
        conversation.append({"role": "user", "content": user_prompt})

        # LLM call
        response_text = ""
        if client is None:
            print("  [ERROR] No LLM client. Using fallback.")
            response_text = '{"action_type": "inspect_step", "args": {"step_name": "dataset"}}'
        else:
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=conversation,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"  [ERROR] LLM call failed: {exc}. Using fallback.")
                response_text = '{"action_type": "inspect_step", "args": {"step_name": "dataset"}}'

        # Keep assistant turn in conversation for multi-turn context
        conversation.append({"role": "assistant", "content": response_text})

        action_type, args = parse_action(response_text)
        
        # Execute action in environment
        try:
            result = env_step(action_type, args)
        except Exception:
            continue

        observation = result.get("observation", {})
        reward      = result.get("reward", 0.0)
        done        = result.get("done", False)
        total_steps = step

        log_step(step, reward)

    # Get grader score
    try:
        grader_result = env_grader()
        score: float = float(grader_result.get("score", 0.0))
        if total_steps == 0:
            total_steps = int(observation.get("steps_taken", 0))
        log_end(task_name, score, total_steps)
    except Exception as exc:
        if total_steps == 0:
            total_steps = int(observation.get("steps_taken", 0))
        log_end(task_name, 0.0, total_steps)
        score = 0.0

    return score


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> dict[str, float]:
    """
    Run the baseline agent across all 3 tasks.
    Returns a dict of scores suitable for the /baseline endpoint.
    """
    try:
        client = build_client()
    except Exception as exc:
        print(f"  [ERROR] Failed to initialize client: {exc}")
        client = None

    scores: dict[str, float] = {}

    for task_id in TASK_IDS:
        task_key = f"task_{task_id}"
        try:
            score = run_episode(client, task_id)
        except Exception as exc:
            print(f"  [ERROR] Episode {task_id} unexpectedly failed: {exc}")
            score = 0.0
        scores[task_key] = round(score, 4)

    aggregate = round(sum(scores.values()) / max(1, len(scores)), 4)
    scores["aggregate"] = aggregate

    return scores


if __name__ == "__main__":
    main()

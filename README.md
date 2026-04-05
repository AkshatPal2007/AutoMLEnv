---
title: AutoMLEnv
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# AutoMLEnv — ML Pipeline Debugging Environment

An OpenEnv-compliant environment where an LLM agent debugs broken machine learning pipelines. Designed for the **OpenEnv Hackathon** judged by Meta and Hugging Face.

---

## Overview

AutoMLEnv simulates a **real-world task**: a data scientist inherits a broken ML pipeline and must identify and fix all bugs to produce a working, high-quality model. The environment features partial observability, deceptive signals, dense reward shaping, and three difficulty-graded tasks.

### Why this environment?

ML pipeline debugging is something practitioners do daily. It requires:
- Systematic inspection before acting
- Distinguishing real improvement from misleading metrics
- Correct sequencing of interdependent fixes
- Efficient reasoning (fewest steps to correct solution)

---

## Environment Design

### Action / Observation / Reward

**Observation** (partial observability):

| Field | Always visible | Requires `inspect_step` |
|-------|---------------|------------------------|
| `pipeline_status` | ✅ | — |
| `last_run_metrics` | ✅ | — |
| `error_log` | ✅ | — |
| `action_history` | ✅ | — |
| `metric_warning` | ✅ (optional) | — |
| Imputer config | ❌ | `inspect_step("imputer")` |
| Feature selection | ❌ | `inspect_step("feature_selector")` |
| Scaler fit scope | ❌ | `inspect_step("scaler")` |
| Model hyperparams | ❌ | `inspect_step("model")` |
| Eval split strategy | ❌ | `inspect_step("evaluator")` |

**Action space** (concrete ML operations):

```
inspect_step            apply_standard_scaler    apply_minmax_scaler
apply_mean_imputer      apply_median_imputer     fix_label_encoding
fix_onehot_encoding     set_feature_columns      set_model_class
set_model_hyperparams   set_eval_split           set_eval_metric
run_pipeline            run_cross_validation     patch_custom_transformer
flip_target_encoding
```

**Reward function** (dense, F1-anchored):

| Signal | Value |
|--------|-------|
| Per-step cost | −0.01 |
| Pipeline runs without error | +0.05 |
| F1 / R² improves | +0.04 × delta |
| Error count decreases | +0.02 × fixed |
| Same action repeated ≥4× | −0.10 |
| Destructive action | −0.15 |
| Run with known unfixed errors | −0.03 |
| Efficiency bonus (terminal) | up to +0.12 |
| Stability bonus (terminal) | up to +0.08 |

---

## Tasks

### Task 1 — Easy: Fix the Broken Preprocessor

**Scenario:** A classification pipeline crashes on run. The dataset has class imbalance (82% majority) — accuracy is a deceptive metric.

**Injected bugs:**
1. No imputer (NaN crash)
2. Categorical column not encoded (dtype error)
3. StandardScaler fitted on full dataset (data leakage — runs but scores poorly)

**Deceptive signal:** After fixing bugs 1 & 2, accuracy jumps to 0.89 — looks great. But F1 stays at 0.67 and `metric_warning` fires. Bug 3 (leakage) still needs fixing.

**Baseline score:** ~0.10 | **Perfect score:** ~1.10

---

### Task 2 — Medium: Fix and Optimize a Failing Pipeline

**Scenario:** Pipeline runs but produces terrible metrics. One bug is hidden behind the evaluator config.

**Injected bugs:**
1. Wrong feature columns (irrelevant cols in, key col out)
2. Degenerate hyperparams (`n_estimators=1`, `max_depth=1`)
3. Target variable inverted (0/1 flipped)
4. **Hidden:** Evaluation computed on train set — requires `inspect_step("evaluator")`

**Deceptive trap:** After fixing bugs 1–3, train F1 = 0.91 — looks perfect. True test F1 = 0.76. The agent must independently suspect and fix the eval split.

**Baseline score:** ~0.50 | **Perfect score:** ~1.08

---

### Task 3 — Hard: End-to-End Pipeline Reconstruction

**Scenario:** A regression pipeline with five cascading bugs. Fixing them in the wrong order creates misleading intermediate states.

**Injected bugs (correct fix order):**
1. Wrong model class (classifier on regression target) — crash on fix reveals bug 2
2. Custom transformer off-by-one index — patching reveals inflated R²
3. Scaler fitted on full dataset (leakage) — fixing drops apparent R² briefly
4. Eval on train set — true R² drops from 0.91 to 0.48 after fix
5. Irrelevant features included — removing them requires continued refinement

**Short-term trap:** `apply_minmax_scaler(fit_on="full")` boosts R² by 0.08 immediately — but this is the leakage. Stability bonus penalises high-variance trajectories.

**Baseline score:** ~0.21 | **Perfect score:** ~1.08

---

## API

All endpoints follow the OpenEnv spec.

### `POST /reset`
```json
{ "task_id": 1, "seed": 42 }
```
Returns: `Observation`

### `POST /step`
```json
{ "action_type": "inspect_step", "args": { "step_name": "imputer" } }
```
Returns: `StepResult` — `{ observation, reward, done, info }`

### `GET /state`
Returns current `Observation` without advancing the episode.

### `GET /tasks`
Returns all task metadata and the full action/observation schema.

### `POST /grader`
Returns `GraderResult` — `{ score, breakdown, efficiency_bonus, stability_bonus }`.
Re-evaluates on held-out test set — cannot be spoofed by eval-on-train bug.

### `POST /baseline`
Triggers `inference.py` across all 3 tasks. Returns:
```json
{
  "status": "ok",
  "scores": { "task_1": 0.21, "task_2": 0.33, "task_3": 0.14, "aggregate": 0.23 }
}
```

---

## Grader Formulas

All graders are deterministic and reproducible (fixed seed).

### Task 1 (Easy)
| Criterion | Weight |
|-----------|--------|
| Pipeline runs without error | 0.25 |
| F1 > 0.65 (on true test set) | 0.25 |
| No data leakage (scaler fit on train) | 0.20 |
| Imputer present | 0.15 |
| Encoder present | 0.15 |
| Efficiency bonus | up to 0.10 |
| Stability bonus | up to 0.05 |

### Task 2 (Medium)
| Criterion | Weight |
|-----------|--------|
| Pipeline runs cleanly | 0.20 |
| F1 > 0.74 (on true test set) | 0.30 |
| Feature set overlap ≥ 85% | 0.20 |
| Hyperparams not degenerate | 0.15 |
| Eval confirmed on test set | 0.15 |
| Efficiency bonus | up to 0.10 |
| Stability bonus | up to 0.05 |

### Task 3 (Hard)
| Criterion | Weight |
|-----------|--------|
| Pipeline runs end-to-end | 0.15 |
| R² > 0.78 (on true test set) | 0.25 |
| Correct model class (regressor) | 0.15 |
| No data leakage | 0.15 |
| Eval confirmed on test set | 0.10 |
| Custom transformer fixed | 0.10 |
| Feature set overlap ≥ 80% | 0.10 |
| Efficiency bonus | up to 0.12 |
| Stability bonus | up to 0.08 |

---

## Setup and Local Development

### Requirements
- Python 3.11+
- Docker (for containerised deployment)

### Run locally

```bash
git clone <your-repo>
cd automlenv

pip install -r requirements.txt

# Start the environment server
uvicorn main:app --host 0.0.0.0 --port 8000

# In a second terminal — run the baseline agent
export API_BASE_URL=https://router.huggingface.co/v1
export HF_TOKEN=your_token_here
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export ENV_BASE_URL=http://localhost:8000
python inference.py
```

### Run with Docker

```bash
docker build -t automlenv .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  automlenv
```

### Pre-submission validation

```bash
# With server running:
python validate.py

# Against deployed HF Space:
python validate.py --url https://your-space.hf.space
```

---

## Project Structure

```
automlenv/
├── main.py           # FastAPI server — all OpenEnv endpoints
├── environment.py    # Core env: step(), reset(), state(), grade()
├── tasks.py          # Task configs + deterministic bug injection
├── models.py         # Pydantic schemas (Action, Observation, StepResult...)
├── inference.py      # Baseline agent using OpenAI client
├── validate.py       # Pre-submission validation script
├── openenv.yaml      # OpenEnv spec metadata
├── Dockerfile        # HF Spaces deployment
├── requirements.txt
└── README.md
```

---

## Baseline Scores

Scores achieved by `meta-llama/Llama-3.1-8B-Instruct` with no task-specific prompting:

| Task | Baseline Score |
|------|---------------|
| Task 1 (Easy) | ~0.21 |
| Task 2 (Medium) | ~0.33 |
| Task 3 (Hard) | ~0.14 |
| **Aggregate** | **~0.23** |

A strong agent solving all bugs optimally achieves ~1.05–1.10 across tasks.

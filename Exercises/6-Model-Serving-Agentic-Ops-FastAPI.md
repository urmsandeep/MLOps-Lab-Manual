# Exercise 6 — FastAPI Model Serving with Agentic-Ops

> **Series position:** Builds directly on Exercise 5 (MLflow Model Registry).  
> The registered model artifact produced there is the starting point for everything here.

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Background Concepts](#background-concepts)
   - [Why FastAPI for model serving?](#why-fastapi-for-model-serving)
   - [MLflow Model Aliases](#mlflow-model-aliases)
   - [What is Agentic-Ops?](#what-is-agentic-ops)
4. [Project Structure](#project-structure)
5. [Setup](#setup)
6. [API Reference](#api-reference)
   - [Ops Endpoints](#ops-endpoints)
   - [Inference Endpoints](#inference-endpoints)
   - [Agentic-Ops Endpoints](#agentic-ops-endpoints)
7. [Architecture Deep-Dive](#architecture-deep-dive)
   - [Startup: Model Loading via Lifespan](#startup-model-loading-via-lifespan)
   - [Request Flow](#request-flow)
   - [Pydantic Schemas](#pydantic-schemas)
   - [Background Tasks and Trace Storage](#background-tasks-and-trace-storage)
   - [The Agentic-Ops Layer](#the-agentic-ops-layer)
8. [Code Walkthrough](#code-walkthrough)
   - [Configuration](#configuration)
   - [State Management](#state-management)
   - [`/predict` — Single-row inference](#predict--single-row-inference)
   - [`/predict/batch` — Batch inference](#predictbatch--batch-inference)
   - [`/agent/schema` — Tool discovery](#agentschema--tool-discovery)
   - [`/agent/invoke` — LLM-callable inference](#agentinvoke--llm-callable-inference)
   - [`/agent/feedback` — HITL data collection](#agentfeedback--hitl-data-collection)
   - [`/agent/trace/{id}` — Audit log](#agenttraceid--audit-log)
9. [Running the Tests](#running-the-tests)
10. [Exercises](#exercises)
11. [Production Upgrade Path](#production-upgrade-path)
12. [Key Takeaways](#key-takeaways)

---

## Overview

This exercise wraps a model registered in the **MLflow Model Registry** in a production-grade **FastAPI** service. You will build two layers simultaneously:

**Standard REST layer** — the minimum viable serving API any ML model needs: health checks, metadata, single-row inference, and batch inference.

**Agentic-Ops layer** — four additional endpoints that make the service *agent-native*: an LLM agent can discover the tool from a schema endpoint, call the model as a function, submit human-in-the-loop labels, and retrieve prediction traces for auditing. No hand-written adapter code is required on the agent side.

The domain is **customer churn prediction** (Telco dataset), but every pattern here generalises to any classification or regression model registered in MLflow.

---

## Learning Objectives

By the end of this exercise you will be able to:

- Load a model from the MLflow Model Registry using the modern **alias URI** (`models:/name@alias`) instead of the deprecated stage-based URI.
- Structure a FastAPI app with a proper **lifespan** handler for one-time model loading.
- Use **Pydantic v2 models** for request validation and automatic OpenAPI documentation.
- Implement **BackgroundTasks** so trace storage never adds latency to the response path.
- Expose a service as an **OpenAI-compatible tool** that any function-calling LLM can discover and invoke without custom wrapper code.
- Understand the **HITL feedback loop**: how a prediction trace flows from inference → human review → label storage → future retraining.
- Write a **pytest suite** that mocks MLflow entirely so tests run offline in CI.

---

## Background Concepts

### Why FastAPI for model serving?

FastAPI gives you three things that matter for ML services:

**Automatic schema validation.** Pydantic models at the boundary mean malformed payloads are rejected before they reach inference code, with a clear 422 response. No defensive `try/except` boilerplate needed for input issues.

**Automatic OpenAPI docs.** Every endpoint, request body, and response type is visible at `/docs` (Swagger UI) and `/redoc` the moment the server starts. This is the schema your `/agent/schema` endpoint replaces and extends for LLM agents.

**Async-first with sync escape hatches.** Inference with sklearn/XGBoost is synchronous CPU work; FastAPI runs sync endpoints in a thread pool automatically, keeping the event loop free for health checks and lightweight requests.

Alternative serving frameworks (BentoML, Triton, Ray Serve, TorchServe) offer more infra-level features like batching queues and GPU memory management. FastAPI is the right choice when you want full control over request handling logic, especially when adding agentic layers.

### MLflow Model Aliases

MLflow 2.x deprecated the `Production` / `Staging` / `Archived` stages in favour of **aliases** — arbitrary string tags that point to a specific model version.

```
# Old (deprecated)
models:/churn_classifier/Production

# New (aliases)
models:/churn_classifier@champion
```

Aliases decouple deployment decisions from version numbers. Your serving code always loads `@champion`; the MLOps team moves the alias to a new version after validation. The server picks it up on restart (or via a `/model/reload` endpoint — see Exercise 1 below).

In this exercise the alias `champion` is set by `scripts/register_model.py` and read at startup via:

```python
client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
```

### What is Agentic-Ops?

*Agentic-Ops* is the emerging practice of designing ML services so that **LLM-based agents can be first-class consumers** alongside human users and traditional application code.

A conventional ML API is designed for application developers who read documentation, write code, and deploy. An agent-native API is designed so that:

1. The **tool schema** is machine-readable and self-describing — the LLM fetches it once and needs no human to translate it into code.
2. The **invoke endpoint** returns not just a prediction, but structured metadata (trace ID, explanation hints) the agent can use to reason about and narrate the result.
3. **Human feedback** flows back through a dedicated endpoint, creating a data flywheel for continuous improvement.
4. Every prediction is **traceable** — guardrail systems, safety layers, and audit pipelines can retrieve the exact inputs and outputs for any inference call.

This exercise implements all four pieces. The `/agent/*` prefix is a naming convention; the important thing is the contract each endpoint satisfies.

---

## Project Structure

```
ex06_fastapi_serving/
│
├── app/
│   ├── __init__.py
│   └── main.py                  # All FastAPI routes, schemas, and logic
│
├── scripts/
│   └── register_model.py  # Bootstraps MLflow registry (use if Ex 6 not done)
│
├── tests/
│   ├── __init__.py
│   └── test_api.py              # 19-test pytest suite (fully mocked, no server needed)
│
├── notebooks/
│   └── ex06_walkthrough.ipynb   # Step-by-step interactive walkthrough
│
├── requirements.txt
├── README.md                    # Quick-start reference
└── EXERCISE_06.md               # This file
```

requirements.txt

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
mlflow>=2.13.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
pydantic>=2.7.0
httpx>=0.27.0        # for notebook smoke-tests
pytest>=8.2.0
```

```
"""
main.py

Exercise 6 – FastAPI Model Serving
===================================
Serves a model registered in the MLflow Model Registry and exposes:

  REST Layer
  ----------
  POST /predict            – single-row inference
  POST /predict/batch      – batch inference
  GET  /model/info         – registered model metadata
  GET  /health             – liveness probe
  GET  /metrics            – Prometheus-compatible counters (agentic-ops ready)

  Agentic-Ops Layer  (new in 2024-25)
  ------------------------------------
  POST /agent/invoke       – LLM-callable tool endpoint (OpenAI tool-call schema)
  GET  /agent/schema       – JSON Schema the LLM reads to auto-discover the tool
  POST /agent/feedback     – human-in-the-loop correctness signal
  GET  /agent/trace/{id}   – retrieve a stored prediction trace for auditability

The /agent/* endpoints make this service "agent-native":
  • an LLM agent can call /agent/schema once, then invoke /agent/invoke
    with structured args without any hand-written wrapper code.
  • /agent/feedback enables RLHF-style online data collection.
  • /agent/trace supports explainability pipelines and guardrail systems.
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME          = os.getenv("REGISTERED_MODEL_NAME", "churn_classifier")
MODEL_ALIAS         = os.getenv("MODEL_ALIAS", "champion")   # replaces deprecated "Production" stage

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ── In-memory state (swap for Redis / DB in production) ──────────────────────
_state: dict[str, Any] = {
    "model": None,
    "model_version": None,
    "loaded_at": None,
    "request_count": 0,
    "error_count": 0,
    "traces": {},       # trace_id → PredictionTrace
    "feedback": [],     # list of FeedbackPayload
}

# ── Lifespan: load model once at startup ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model '%s' @ alias '%s' …", MODEL_NAME, MODEL_ALIAS)
    try:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        _state["model"] = mlflow.pyfunc.load_model(model_uri)
        # Pull version metadata
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        _state["model_version"] = mv.version
        _state["loaded_at"] = time.time()
        log.info("Model v%s loaded OK.", mv.version)
    except Exception as exc:
        log.error("Model load failed: %s – serving /predict will return 503", exc)
        # Don't crash the server; /health will report degraded state
    yield
    log.info("Shutting down – bye!")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MLflow Model Serving API",
    description=(
        "Exercise 7 – serves a registered MLflow model with an agent-native "
        "tool interface so LLM agents can discover and call it automatically."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    """Single-row feature payload. Keys must match training feature names."""
    features: dict[str, Any] = Field(
        ...,
        example={
            "tenure": 24,
            "monthly_charges": 65.5,
            "total_charges": 1572.0,
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "tech_support": "No",
        },
    )
    return_proba: bool = Field(False, description="If true, return class probabilities.")

class PredictResponse(BaseModel):
    trace_id: str
    prediction: Any
    model_version: str | None
    latency_ms: float

class BatchPredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1, max_length=512)
    return_proba: bool = False

class BatchPredictResponse(BaseModel):
    trace_ids: list[str]
    predictions: list[Any]
    model_version: str | None
    latency_ms: float

class ModelInfo(BaseModel):
    name: str
    alias: str
    version: str | None
    loaded_at: float | None
    request_count: int
    error_count: int

class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded" | "down"
    model_loaded: bool
    uptime_s: float | None

# ── Agentic-Ops schemas ───────────────────────────────────────────────────────

class AgentInvokeRequest(BaseModel):
    """
    Mirrors the OpenAI tool-call format so any function-calling LLM
    (GPT-4o, Claude, Gemini …) can call this endpoint without a wrapper.
    """
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    features: dict[str, Any]
    return_proba: bool = False

class AgentInvokeResponse(BaseModel):
    call_id: str
    trace_id: str
    result: Any                    # prediction or probabilities
    model_version: str | None
    latency_ms: float
    explanation_hint: str | None   # lightweight explainability stub

class FeedbackPayload(BaseModel):
    trace_id: str
    correct_label: Any
    annotator: str = "human"
    notes: str = ""

class PredictionTrace(BaseModel):
    trace_id: str
    features: dict[str, Any]
    prediction: Any
    model_version: str | None
    timestamp: float
    latency_ms: float

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _require_model():
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded – check server logs.")

def _run_inference(features: dict[str, Any], return_proba: bool) -> tuple[Any, float]:
    """Run model inference; return (prediction, latency_ms)."""
    _require_model()
    t0 = time.perf_counter()
    df = pd.DataFrame([features])
    if return_proba and hasattr(_state["model"], "predict_proba"):
        result = _state["model"].predict_proba(df).tolist()
    else:
        raw = _state["model"].predict(df)
        result = raw.tolist() if hasattr(raw, "tolist") else raw
        result = result[0] if isinstance(result, list) and len(result) == 1 else result
    latency_ms = (time.perf_counter() - t0) * 1000
    return result, latency_ms

def _store_trace(trace_id: str, features: dict, prediction: Any, latency_ms: float):
    _state["traces"][trace_id] = PredictionTrace(
        trace_id=trace_id,
        features=features,
        prediction=prediction,
        model_version=_state["model_version"],
        timestamp=time.time(),
        latency_ms=latency_ms,
    )

# ═════════════════════════════════════════════════════════════════════════════
# REST endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health():
    loaded = _state["model"] is not None
    uptime = (time.time() - _state["loaded_at"]) if _state["loaded_at"] else None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        uptime_s=uptime,
    )

@app.get("/metrics", tags=["Ops"])
def metrics():
    """Prometheus-style plain-text metrics (swap for prometheus_fastapi_instrumentator in prod)."""
    lines = [
        f'model_requests_total {_state["request_count"]}',
        f'model_errors_total {_state["error_count"]}',
        f'model_version_info{{version="{_state["model_version"] or "unknown"}"}} 1',
        f'traces_stored {len(_state["traces"])}',
        f'feedback_records {len(_state["feedback"])}',
    ]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines) + "\n")

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
def model_info():
    return ModelInfo(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=_state["model_version"],
        loaded_at=_state["loaded_at"],
        request_count=_state["request_count"],
        error_count=_state["error_count"],
    )

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    _state["request_count"] += 1
    try:
        prediction, latency_ms = _run_inference(req.features, req.return_proba)
    except HTTPException:
        raise
    except Exception as exc:
        _state["error_count"] += 1
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(exc))

    trace_id = str(uuid.uuid4())
    background_tasks.add_task(_store_trace, trace_id, req.features, prediction, latency_ms)
    return PredictResponse(
        trace_id=trace_id,
        prediction=prediction,
        model_version=_state["model_version"],
        latency_ms=round(latency_ms, 2),
    )

@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(req: BatchPredictRequest, background_tasks: BackgroundTasks):
    _require_model()
    _state["request_count"] += len(req.rows)
    t0 = time.perf_counter()
    try:
        df = pd.DataFrame(req.rows)
        if req.return_proba and hasattr(_state["model"], "predict_proba"):
            preds = _state["model"].predict_proba(df).tolist()
        else:
            raw = _state["model"].predict(df)
            preds = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    except Exception as exc:
        _state["error_count"] += 1
        log.exception("Batch inference error")
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    trace_ids = [str(uuid.uuid4()) for _ in preds]
    for tid, feat, pred in zip(trace_ids, req.rows, preds):
        background_tasks.add_task(_store_trace, tid, feat, pred, latency_ms / len(preds))
    return BatchPredictResponse(
        trace_ids=trace_ids,
        predictions=preds,
        model_version=_state["model_version"],
        latency_ms=latency_ms,
    )

# ═════════════════════════════════════════════════════════════════════════════
# Agentic-Ops endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/agent/schema", tags=["Agentic-Ops"])
def agent_schema():
    """
    Returns the OpenAI-compatible tool schema for this endpoint.
    An LLM agent fetches this once and can then call /agent/invoke
    autonomously – no hand-written tool wrapper needed.

    Compatible with:  OpenAI function-calling, Anthropic tool-use,
                      LangChain Tool, LlamaIndex FunctionTool
    """
    return {
        "type": "function",
        "function": {
            "name": "predict_churn",
            "description": (
                "Predict customer churn probability using the registered ML model. "
                "Call this whenever you need a churn score for a single customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tenure":           {"type": "integer",  "description": "Months as customer"},
                    "monthly_charges":  {"type": "number",   "description": "USD per month"},
                    "total_charges":    {"type": "number",   "description": "Lifetime USD spend"},
                    "contract_type":    {"type": "string",   "enum": ["Month-to-month","One year","Two year"]},
                    "internet_service": {"type": "string",   "enum": ["DSL","Fiber optic","No"]},
                    "tech_support":     {"type": "string",   "enum": ["Yes","No","No internet service"]},
                    "return_proba":     {"type": "boolean",  "default": False},
                },
                "required": ["tenure", "monthly_charges", "total_charges",
                             "contract_type", "internet_service", "tech_support"],
            },
        },
    }

@app.post("/agent/invoke", response_model=AgentInvokeResponse, tags=["Agentic-Ops"])
def agent_invoke(req: AgentInvokeRequest, background_tasks: BackgroundTasks):
    """
    LLM-callable inference endpoint.
    Returns a lightweight explanation hint so the agent can
    narrate the result to the end-user naturally.
    """
    _state["request_count"] += 1
    try:
        prediction, latency_ms = _run_inference(req.features, req.return_proba)
    except HTTPException:
        raise
    except Exception as exc:
        _state["error_count"] += 1
        raise HTTPException(status_code=500, detail=str(exc))

    trace_id = str(uuid.uuid4())
    background_tasks.add_task(_store_trace, trace_id, req.features, prediction, latency_ms)

    # Lightweight explanation hint for the LLM to use in its reply
    hint = _build_explanation_hint(req.features, prediction)

    return AgentInvokeResponse(
        call_id=req.call_id,
        trace_id=trace_id,
        result=prediction,
        model_version=_state["model_version"],
        latency_ms=round(latency_ms, 2),
        explanation_hint=hint,
    )

@app.post("/agent/feedback", tags=["Agentic-Ops"])
def agent_feedback(payload: FeedbackPayload):
    """
    Human-in-the-loop feedback endpoint.
    Stores (trace_id, correct_label) pairs for:
      • online RLHF / fine-tuning pipelines
      • model drift monitoring
      • guardrail system training
    """
    if payload.trace_id not in _state["traces"]:
        raise HTTPException(status_code=404, detail=f"trace_id {payload.trace_id!r} not found.")
    _state["feedback"].append(payload)
    log.info("Feedback recorded: trace=%s label=%s", payload.trace_id, payload.correct_label)
    return {"status": "recorded", "feedback_count": len(_state["feedback"])}

@app.get("/agent/trace/{trace_id}", response_model=PredictionTrace, tags=["Agentic-Ops"])
def get_trace(trace_id: str):
    """
    Retrieve a stored prediction trace.
    Used by:
      • guardrail / safety layers that need to inspect what was predicted
      • explainability pipelines (attach SHAP values post-hoc)
      • audit logs for regulated industries
    """
    trace = _state["traces"].get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace

# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_explanation_hint(features: dict, prediction: Any) -> str | None:
    """
    Stub for a feature-importance explanation.
    In production replace with SHAP / LIME or mlflow.models.predict with
    model signatures that include explanation columns.
    """
    try:
        # Simple heuristic hints so the LLM can reason about the prediction
        hints = []
        if features.get("contract_type") == "Month-to-month":
            hints.append("month-to-month contracts are a strong churn driver")
        if features.get("tenure", 999) < 12:
            hints.append("short tenure increases churn risk")
        if features.get("monthly_charges", 0) > 70:
            hints.append("above-average monthly charges correlate with churn")
        return "; ".join(hints) if hints else None
    except Exception:
        return None
```

register_model.py

```
#!/usr/bin/env python
"""
scripts/register_dummy_model.py
================================
Run this ONCE before starting the server when you don't yet have a real
registered model from Exercise 6.

It trains a tiny Logistic Regression on synthetic data, logs it to MLflow,
registers it as 'churn_classifier', and sets the 'champion' alias.

Usage:
    python scripts/register_dummy_model.py
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

TRACKING_URI    = "sqlite:///mlflow.db"
MODEL_NAME      = "churn_classifier"
CHAMPION_ALIAS  = "champion"

mlflow.set_tracking_uri(TRACKING_URI)

# ── Synthetic data ────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
n   = 400

df = pd.DataFrame({
    "tenure":           rng.integers(1, 72, n),
    "monthly_charges":  rng.uniform(20, 120, n).round(2),
    "total_charges":    rng.uniform(20, 8000, n).round(2),
    "contract_type":    rng.choice(["Month-to-month", "One year", "Two year"], n),
    "internet_service": rng.choice(["DSL", "Fiber optic", "No"], n),
    "tech_support":     rng.choice(["Yes", "No", "No internet service"], n),
})

# Simple churn rule: high charges + short tenure + month-to-month
df["churn"] = (
    (df["monthly_charges"] > 70)
    & (df["tenure"] < 18)
    & (df["contract_type"] == "Month-to-month")
).astype(int)

X = df.drop("churn", axis=1)
y = df["churn"]

# ── Pipeline ──────────────────────────────────────────────────────────────────
cat_cols = ["contract_type", "internet_service", "tech_support"]
num_cols = ["tenure", "monthly_charges", "total_charges"]

pre = ColumnTransformer([
    ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols),
], remainder="passthrough")

pipe = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=500))])

# ── MLflow run ────────────────────────────────────────────────────────────────
mlflow.set_experiment("churn_serving_demo")

with mlflow.start_run(run_name="dummy_lr_for_ex7") as run:
    pipe.fit(X, y)
    acc = (pipe.predict(X) == y).mean()
    mlflow.log_metric("train_accuracy", acc)
    mlflow.log_param("model_type", "LogisticRegression")

    # Log with input_example so the model signature is inferred automatically
    mlflow.sklearn.log_model(
        pipe,
        artifact_path="model",
        input_example=X.head(3),
        registered_model_name=MODEL_NAME,
    )
    print(f"  Run id  : {run.info.run_id}")
    print(f"  Accuracy: {acc:.3f}")

# ── Set champion alias ────────────────────────────────────────────────────────
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_v = max(int(v.version) for v in versions)

client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, str(latest_v))
print(f"\n✓  Model '{MODEL_NAME}' v{latest_v} registered and alias '{CHAMPION_ALIAS}' set.")
print("  Start the server: uvicorn app.main:app --reload")
```


```
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exercise 7 – FastAPI Model Serving\n",
    "\n",
    "**What you'll build**\n",
    "\n",
    "| Layer | Endpoint | Purpose |\n",
    "|---|---|---|\n",
    "| REST | `POST /predict` | Single-row inference |\n",
    "| REST | `POST /predict/batch` | Batch inference (up to 512 rows) |\n",
    "| REST | `GET /model/info` | Registered-model metadata |\n",
    "| REST | `GET /health` | Liveness probe |\n",
    "| REST | `GET /metrics` | Prometheus-style counters |\n",
    "| **Agentic-Ops** | `GET /agent/schema` | OpenAI-compatible tool schema |\n",
    "| **Agentic-Ops** | `POST /agent/invoke` | LLM-callable inference |\n",
    "| **Agentic-Ops** | `POST /agent/feedback` | Human-in-the-loop labelling |\n",
    "| **Agentic-Ops** | `GET /agent/trace/{id}` | Prediction audit log |\n",
    "\n",
    "---\n",
    "\n",
    "## 0 — Prerequisites"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install deps (run once)\n",
    "%pip install fastapi uvicorn[standard] mlflow scikit-learn pandas httpx pytest -q"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1 — Register a model (skip if Ex 6 already did this)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%run scripts/register_dummy_model.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2 — Start the server\n",
    "\n",
    "Open a terminal in the project root and run:\n",
    "\n",
    "```bash\n",
    "uvicorn app.main:app --reload --port 8000\n",
    "```\n",
    "\n",
    "Watch for:\n",
    "```\n",
    "INFO  Loading model 'churn_classifier' @ alias 'champion' …\n",
    "INFO  Model v42 loaded OK.\n",
    "INFO  Uvicorn running on http://127.0.0.1:8000\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3 — Smoke-test with httpx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import httpx, json\n",
    "\n",
    "BASE = \"http://localhost:8000\"\n",
    "\n",
    "# Health check\n",
    "r = httpx.get(f\"{BASE}/health\")\n",
    "print(\"Health:\", r.json())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Single prediction\n",
    "payload = {\n",
    "    \"features\": {\n",
    "        \"tenure\": 6,\n",
    "        \"monthly_charges\": 89.0,\n",
    "        \"total_charges\": 534.0,\n",
    "        \"contract_type\": \"Month-to-month\",\n",
    "        \"internet_service\": \"Fiber optic\",\n",
    "        \"tech_support\": \"No\",\n",
    "    }\n",
    "}\n",
    "\n",
    "r = httpx.post(f\"{BASE}/predict\", json=payload)\n",
    "resp = r.json()\n",
    "print(json.dumps(resp, indent=2))\n",
    "\n",
    "TRACE_ID = resp[\"trace_id\"]   # save for later cells"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Batch prediction\n",
    "batch_payload = {\n",
    "    \"rows\": [\n",
    "        {\"tenure\": 6,  \"monthly_charges\": 89.0, \"total_charges\": 534.0,\n",
    "         \"contract_type\": \"Month-to-month\", \"internet_service\": \"Fiber optic\", \"tech_support\": \"No\"},\n",
    "        {\"tenure\": 48, \"monthly_charges\": 45.0, \"total_charges\": 2160.0,\n",
    "         \"contract_type\": \"Two year\",       \"internet_service\": \"DSL\",         \"tech_support\": \"Yes\"},\n",
    "    ]\n",
    "}\n",
    "\n",
    "r = httpx.post(f\"{BASE}/predict/batch\", json=batch_payload)\n",
    "print(json.dumps(r.json(), indent=2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 4 — Agentic-Ops walkthrough\n",
    "\n",
    "### 4a  Fetch the tool schema (what an LLM agent does first)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "schema = httpx.get(f\"{BASE}/agent/schema\").json()\n",
    "print(json.dumps(schema, indent=2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The schema is **OpenAI tool-call compatible** – paste it into any LLM's `tools=` parameter and the model will know how to call `/agent/invoke` without any hand-written wrapper.\n",
    "\n",
    "### 4b  Simulate an LLM agent calling the tool"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simulated LLM agent call\n",
    "agent_payload = {\n",
    "    \"call_id\": \"llm-call-001\",\n",
    "    \"features\": {\n",
    "        \"tenure\": 3,\n",
    "        \"monthly_charges\": 95.0,\n",
    "        \"total_charges\": 285.0,\n",
    "        \"contract_type\": \"Month-to-month\",\n",
    "        \"internet_service\": \"Fiber optic\",\n",
    "        \"tech_support\": \"No\",\n",
    "    },\n",
    "    \"return_proba\": False,\n",
    "}\n",
    "\n",
    "r = httpx.post(f\"{BASE}/agent/invoke\", json=agent_payload)\n",
    "agent_resp = r.json()\n",
    "print(json.dumps(agent_resp, indent=2))\n",
    "\n",
    "print(\"\\n💡 Explanation hint the LLM can narrate to the user:\")\n",
    "print(\" \", agent_resp.get(\"explanation_hint\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4c  Submit human feedback (RLHF data collection)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time; time.sleep(0.2)   # let background task store the trace\n",
    "\n",
    "feedback_payload = {\n",
    "    \"trace_id\": agent_resp[\"trace_id\"],\n",
    "    \"correct_label\": 1,           # human confirms: yes, this customer did churn\n",
    "    \"annotator\": \"alice@company.com\",\n",
    "    \"notes\": \"Cancelled the next day.\",\n",
    "}\n",
    "\n",
    "r = httpx.post(f\"{BASE}/agent/feedback\", json=feedback_payload)\n",
    "print(r.json())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4d  Retrieve the prediction trace (auditability)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "r = httpx.get(f\"{BASE}/agent/trace/{agent_resp['trace_id']}\")\n",
    "print(json.dumps(r.json(), indent=2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 5 — Run the test suite"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pytest tests/ -v --tb=short"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 6 — Interactive API docs\n",
    "\n",
    "With the server running, open:\n",
    "\n",
    "- **Swagger UI** → http://localhost:8000/docs\n",
    "- **ReDoc**       → http://localhost:8000/redoc\n",
    "\n",
    "---\n",
    "## 7 — Exercises\n",
    "\n",
    "1. **Model hot-reload** — Add a `POST /model/reload` endpoint that re-fetches the `champion` alias from the registry without restarting the server.  \n",
    "   *Hint:* guard against concurrent requests with `asyncio.Lock`.\n",
    "\n",
    "2. **Real SHAP explanations** — Replace `_build_explanation_hint` with `shap.TreeExplainer` (for tree models) or `shap.LinearExplainer` (for linear models), returning the top-3 feature contributions in `explanation_hint`.\n",
    "\n",
    "3. **Prometheus integration** — Swap the hand-rolled `/metrics` for `prometheus_fastapi_instrumentator` and add a histogram for latency.\n",
    "\n",
    "4. **LLM agent demo** — Wire `/agent/schema` + `/agent/invoke` into a real LLM (OpenAI or Anthropic) using function-calling mode. The LLM should be given a customer description in natural language and autonomously decide when to call the churn tool.\n",
    "\n",
    "5. **Feedback pipeline** — Drain `/agent/feedback` records into a DataFrame and retrain the model with the corrected labels (online learning loop).\n",
    "\n",
    "---\n",
    "## 8 — Key concepts\n",
    "\n",
    "| Concept | Where it appears | Why it matters |\n",
    "|---|---|---|\n",
    "| **MLflow Model Alias** | `models:/name@alias` URI | Decouple deployment from version numbers |\n",
    "| **Lifespan events** | `@asynccontextmanager` | Load model once; clean shutdown |\n",
    "| **BackgroundTasks** | `/predict`, `/predict/batch` | Store traces without adding latency |\n",
    "| **OpenAI tool schema** | `/agent/schema` | Makes service discoverable by any LLM |\n",
    "| **Prediction traces** | `/agent/trace/{id}` | Auditability, guardrails, explainability |\n",
    "| **HITL feedback** | `/agent/feedback` | Closes the human-in-the-loop data flywheel |\n",
    "| **Pydantic v2** | All request/response models | Runtime validation + auto-docs |\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
} 
```

ex06_walkthrough.ipynb

```
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exercise 7 – FastAPI Model Serving\n",
    "\n",
    "**What you'll build**\n",
    "\n",
    "| Layer | Endpoint | Purpose |\n",
    "|---|---|---|\n",
    "| REST | `POST /predict` | Single-row inference |\n",
    "| REST | `POST /predict/batch` | Batch inference (up to 512 rows) |\n",
    "| REST | `GET /model/info` | Registered-model metadata |\n",
    "| REST | `GET /health` | Liveness probe |\n",
    "| REST | `GET /metrics` | Prometheus-style counters |\n",
    "| **Agentic-Ops** | `GET /agent/schema` | OpenAI-compatible tool schema |\n",
    "| **Agentic-Ops** | `POST /agent/invoke` | LLM-callable inference |\n",
    "| **Agentic-Ops** | `POST /agent/feedback` | Human-in-the-loop labelling |\n",
    "| **Agentic-Ops** | `GET /agent/trace/{id}` | Prediction audit log |\n",
    "\n",
    "---\n",
    "\n",
    "## 0 — Prerequisites"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install deps (run once)\n",
    "%pip install fastapi uvicorn[standard] mlflow scikit-learn pandas httpx pytest -q"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1 — Register a model (skip if Ex 6 already did this)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%run scripts/register_dummy_model.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2 — Start the server\n",
    "\n",
    "Open a terminal in the project root and run:\n",
    "\n",
    "```bash\n",
    "uvicorn app.main:app --reload --port 8000\n",
    "```\n",
    "\n",
    "Watch for:\n",
    "```\n",
    "INFO  Loading model 'churn_classifier' @ alias 'champion' …\n",
    "INFO  Model v42 loaded OK.\n",
    "INFO  Uvicorn running on http://127.0.0.1:8000\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3 — Smoke-test with httpx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import httpx, json\n",
    "\n",
    "BASE = \"http://localhost:8000\"\n",
    "\n",
    "# Health check\n",
    "r = httpx.get(f\"{BASE}/health\")\n",
    "print(\"Health:\", r.json())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Single prediction\n",
    "payload = {\n",
    "    \"features\": {\n",
    "        \"tenure\": 6,\n",
    "        \"monthly_charges\": 89.0,\n",
    "        \"total_charges\": 534.0,\n",
    "        \"contract_type\": \"Month-to-month\",\n",
    "        \"internet_service\": \"Fiber optic\",\n",
    "        \"tech_support\": \"No\",\n",
    "    }\n",
    "}\n",
    "\n",
    "r = httpx.post(f\"{BASE}/predict\", json=payload)\n",
    "resp = r.json()\n",
    "print(json.dumps(resp, indent=2))\n",
    "\n",
    "TRACE_ID = resp[\"trace_id\"]   # save for later cells"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Batch prediction\n",
    "batch_payload = {\n",
    "    \"rows\": [\n",
    "        {\"tenure\": 6,  \"monthly_charges\": 89.0, \"total_charges\": 534.0,\n",
    "         \"contract_type\": \"Month-to-month\", \"internet_service\": \"Fiber optic\", \"tech_support\": \"No\"},\n",
    "        {\"tenure\": 48, \"monthly_charges\": 45.0, \"total_charges\": 2160.0,\n",
    "         \"contract_type\": \"Two year\",       \"internet_service\": \"DSL\",         \"tech_support\": \"Yes\"},\n",
    "    ]\n",
    "}\n",
    "\n",
    "r = httpx.post(f\"{BASE}/predict/batch\", json=batch_payload)\n",
    "print(json.dumps(r.json(), indent=2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 4 — Agentic-Ops walkthrough\n",
    "\n",
    "### 4a  Fetch the tool schema (what an LLM agent does first)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "schema = httpx.get(f\"{BASE}/agent/schema\").json()\n",
    "print(json.dumps(schema, indent=2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The schema is **OpenAI tool-call compatible** – paste it into any LLM's `tools=` parameter and the model will know how to call `/agent/invoke` without any hand-written wrapper.\n",
    "\n",
    "### 4b  Simulate an LLM agent calling the tool"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simulated LLM agent call\n",
    "agent_payload = {\n",
    "    \"call_id\": \"llm-call-001\",\n",
    "    \"features\": {\n",
    "        \"tenure\": 3,\n",
    "        \"monthly_charges\": 95.0,\n",
    "        \"total_charges\": 285.0,\n",
    "        \"contract_type\": \"Month-to-month\",\n",
    "        \"internet_service\": \"Fiber optic\",\n",
    "        \"tech_support\": \"No\",\n",
    "    },\n",
    "    \"return_proba\": False,\n",
    "}\n",
    "\n",
    "r = httpx.post(f\"{BASE}/agent/invoke\", json=agent_payload)\n",
    "agent_resp = r.json()\n",
    "print(json.dumps(agent_resp, indent=2))\n",
    "\n",
    "print(\"\\n💡 Explanation hint the LLM can narrate to the user:\")\n",
    "print(\" \", agent_resp.get(\"explanation_hint\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4c  Submit human feedback (RLHF data collection)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time; time.sleep(0.2)   # let background task store the trace\n",
    "\n",
    "feedback_payload = {\n",
    "    \"trace_id\": agent_resp[\"trace_id\"],\n",
    "    \"correct_label\": 1,           # human confirms: yes, this customer did churn\n",
    "    \"annotator\": \"alice@company.com\",\n",
    "    \"notes\": \"Cancelled the next day.\",\n",
    "}\n",
    "\n",
    "r = httpx.post(f\"{BASE}/agent/feedback\", json=feedback_payload)\n",
    "print(r.json())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4d  Retrieve the prediction trace (auditability)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "r = httpx.get(f\"{BASE}/agent/trace/{agent_resp['trace_id']}\")\n",
    "print(json.dumps(r.json(), indent=2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 5 — Run the test suite"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pytest tests/ -v --tb=short"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 6 — Interactive API docs\n",
    "\n",
    "With the server running, open:\n",
    "\n",
    "- **Swagger UI** → http://localhost:8000/docs\n",
    "- **ReDoc**       → http://localhost:8000/redoc\n",
    "\n",
    "---\n",
    "## 7 — Exercises\n",
    "\n",
    "1. **Model hot-reload** — Add a `POST /model/reload` endpoint that re-fetches the `champion` alias from the registry without restarting the server.  \n",
    "   *Hint:* guard against concurrent requests with `asyncio.Lock`.\n",
    "\n",
    "2. **Real SHAP explanations** — Replace `_build_explanation_hint` with `shap.TreeExplainer` (for tree models) or `shap.LinearExplainer` (for linear models), returning the top-3 feature contributions in `explanation_hint`.\n",
    "\n",
    "3. **Prometheus integration** — Swap the hand-rolled `/metrics` for `prometheus_fastapi_instrumentator` and add a histogram for latency.\n",
    "\n",
    "4. **LLM agent demo** — Wire `/agent/schema` + `/agent/invoke` into a real LLM (OpenAI or Anthropic) using function-calling mode. The LLM should be given a customer description in natural language and autonomously decide when to call the churn tool.\n",
    "\n",
    "5. **Feedback pipeline** — Drain `/agent/feedback` records into a DataFrame and retrain the model with the corrected labels (online learning loop).\n",
    "\n",
    "---\n",
    "## 8 — Key concepts\n",
    "\n",
    "| Concept | Where it appears | Why it matters |\n",
    "|---|---|---|\n",
    "| **MLflow Model Alias** | `models:/name@alias` URI | Decouple deployment from version numbers |\n",
    "| **Lifespan events** | `@asynccontextmanager` | Load model once; clean shutdown |\n",
    "| **BackgroundTasks** | `/predict`, `/predict/batch` | Store traces without adding latency |\n",
    "| **OpenAI tool schema** | `/agent/schema` | Makes service discoverable by any LLM |\n",
    "| **Prediction traces** | `/agent/trace/{id}` | Auditability, guardrails, explainability |\n",
    "| **HITL feedback** | `/agent/feedback` | Closes the human-in-the-loop data flywheel |\n",
    "| **Pydantic v2** | All request/response models | Runtime validation + auto-docs |\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
  ```

Testing the Run

```
"""
tests/test_api.py
==================
Pytest test suite for Exercise 7 – FastAPI Model Serving.

Run:
    pytest tests/ -v

The tests use a lightweight mock model so they run offline
(no MLflow server required).
"""

import time
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── Patch model loading before importing app ──────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def mock_mlflow():
    """Inject a fake sklearn-style model so tests don't need a real MLflow server."""
    fake_model = MagicMock()
    # Side-effect: return one prediction per input row
    fake_model.predict.side_effect = lambda df: np.ones(len(df), dtype=int)
    fake_model.predict_proba.side_effect = lambda df: np.tile([0.3, 0.7], (len(df), 1))

    fake_mv = MagicMock()
    fake_mv.version = "42"

    with (
        patch("mlflow.pyfunc.load_model", return_value=fake_model),
        patch(
            "mlflow.tracking.MlflowClient.get_model_version_by_alias",
            return_value=fake_mv,
        ),
    ):
        yield fake_model


@pytest.fixture(scope="session")
def client(mock_mlflow):
    from app.main import app, _state

    # Manually populate state (lifespan ran during import with mock)
    _state["model"] = mock_mlflow
    _state["model_version"] = "42"
    _state["loaded_at"] = time.time()

    with TestClient(app) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# Ops endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_uptime_positive(self, client):
        r = client.get("/health")
        assert r.json()["uptime_s"] > 0


class TestMetrics:
    def test_metrics_text(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "model_requests_total" in r.text
        assert "model_version_info" in r.text


class TestModelInfo:
    def test_version_present(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        assert r.json()["version"] == "42"


# ─────────────────────────────────────────────────────────────────────────────
# Inference endpoints
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_FEATURES = {
    "tenure": 6,
    "monthly_charges": 85.0,
    "total_charges": 510.0,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "tech_support": "No",
}


class TestPredict:
    def test_basic(self, client):
        r = client.post("/predict", json={"features": SAMPLE_FEATURES})
        assert r.status_code == 200
        body = r.json()
        assert "prediction" in body
        assert "trace_id" in body
        assert "latency_ms" in body

    def test_trace_id_is_uuid(self, client):
        r = client.post("/predict", json={"features": SAMPLE_FEATURES})
        tid = r.json()["trace_id"]
        uuid.UUID(tid)  # raises if not valid UUID

    def test_return_proba(self, client):
        r = client.post("/predict", json={"features": SAMPLE_FEATURES, "return_proba": True})
        assert r.status_code == 200

    def test_missing_features_field(self, client):
        r = client.post("/predict", json={"not_features": {}})
        assert r.status_code == 422   # Pydantic validation error


class TestBatchPredict:
    def test_basic(self, client):
        mock_model = client.app.state  # not used; mock is patched globally
        rows = [SAMPLE_FEATURES, {**SAMPLE_FEATURES, "tenure": 36}]
        r = client.post("/predict/batch", json={"rows": rows})
        assert r.status_code == 200
        body = r.json()
        assert len(body["trace_ids"]) == len(rows)
        assert len(body["predictions"]) == len(rows)

    def test_empty_rows_rejected(self, client):
        r = client.post("/predict/batch", json={"rows": []})
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Agentic-Ops endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentSchema:
    def test_returns_openai_tool_format(self, client):
        r = client.get("/agent/schema")
        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "function"
        assert "function" in body
        fn = body["function"]
        assert fn["name"] == "predict_churn"
        assert "parameters" in fn
        # Required fields present
        for field in ["tenure", "monthly_charges", "total_charges"]:
            assert field in fn["parameters"]["required"]

    def test_schema_has_descriptions(self, client):
        r = client.get("/agent/schema")
        props = r.json()["function"]["parameters"]["properties"]
        for key, val in props.items():
            assert "description" in val or "type" in val, f"{key} missing type/description"


class TestAgentInvoke:
    def test_basic(self, client):
        r = client.post("/agent/invoke", json={"features": SAMPLE_FEATURES})
        assert r.status_code == 200
        body = r.json()
        assert "result" in body
        assert "trace_id" in body
        assert "call_id" in body

    def test_custom_call_id_preserved(self, client):
        cid = "test-call-001"
        r = client.post("/agent/invoke", json={"features": SAMPLE_FEATURES, "call_id": cid})
        assert r.json()["call_id"] == cid

    def test_explanation_hint_present_for_high_risk(self, client):
        high_risk = {**SAMPLE_FEATURES, "tenure": 5, "monthly_charges": 95.0}
        r = client.post("/agent/invoke", json={"features": high_risk})
        hint = r.json().get("explanation_hint")
        assert hint is not None and len(hint) > 0


class TestAgentFeedback:
    def test_record_feedback(self, client):
        # First make a prediction to get a trace_id
        pred_r = client.post("/predict", json={"features": SAMPLE_FEATURES})
        trace_id = pred_r.json()["trace_id"]

        # Small sleep to let background task store the trace
        time.sleep(0.05)

        r = client.post(
            "/agent/feedback",
            json={"trace_id": trace_id, "correct_label": 1, "annotator": "tester"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "recorded"

    def test_feedback_unknown_trace(self, client):
        r = client.post(
            "/agent/feedback",
            json={"trace_id": "nonexistent-id", "correct_label": 0},
        )
        assert r.status_code == 404


class TestAgentTrace:
    def test_retrieve_trace(self, client):
        pred_r = client.post("/predict", json={"features": SAMPLE_FEATURES})
        trace_id = pred_r.json()["trace_id"]
        time.sleep(0.05)  # background task

        r = client.get(f"/agent/trace/{trace_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["trace_id"] == trace_id
        assert "features" in body
        assert "prediction" in body
        assert "timestamp" in body

    def test_missing_trace_404(self, client):
        r = client.get("/agent/trace/does-not-exist")
        assert r.status_code == 404
```


## Setup

### Prerequisites

- Python 3.11+
- Exercise 6 completed **or** the dummy model script below

### Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins:

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
mlflow>=2.13.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
pydantic>=2.7.0
httpx>=0.27.0
pytest>=8.2.0
```

### Register a model

If Exercise 6 already registered `churn_classifier` with the `champion` alias, skip this step.

Otherwise, run the bootstrap script which trains a tiny Logistic Regression on synthetic data, logs it to MLflow, and sets the alias:

```bash
python scripts/register_dummy_model.py
```

Expected output:

```
  Run id  : abc123...
  Accuracy: 0.873

✓  Model 'churn_classifier' v1 registered and alias 'champion' set.
   Start the server: uvicorn app.main:app --reload
```

### Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

Watch for the model-loaded confirmation:

```
INFO  Loading model 'churn_classifier' @ alias 'champion' …
INFO  Model v1 loaded OK.
INFO  Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Explore the docs

- **Swagger UI** → http://localhost:8000/docs
- **ReDoc** → http://localhost:8000/redoc

---

## API Reference

### Ops Endpoints

#### `GET /health`

Liveness probe. Returns `"ok"` when the model is loaded, `"degraded"` if the model failed to load (server still runs; inference returns 503).

```json
{
  "status": "ok",
  "model_loaded": true,
  "uptime_s": 142.7
}
```

#### `GET /metrics`

Prometheus-compatible plain-text counters. Designed to be scraped by a Prometheus instance or read directly.

```
model_requests_total 48
model_errors_total 0
model_version_info{version="3"} 1
traces_stored 48
feedback_records 2
```

In production, replace with `prometheus_fastapi_instrumentator` for histogram latency tracking (see Exercise 3 below).

#### `GET /model/info`

Returns metadata about the currently loaded model version.

```json
{
  "name": "churn_classifier",
  "alias": "champion",
  "version": "3",
  "loaded_at": 1710000000.0,
  "request_count": 48,
  "error_count": 0
}
```

---

### Inference Endpoints

#### `POST /predict`

Single-row inference. The `features` dict must contain the keys the model was trained on.

**Request:**

```json
{
  "features": {
    "tenure": 6,
    "monthly_charges": 89.0,
    "total_charges": 534.0,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "tech_support": "No"
  },
  "return_proba": false
}
```

**Response:**

```json
{
  "trace_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "prediction": 1,
  "model_version": "3",
  "latency_ms": 4.21
}
```

Set `return_proba: true` to get class probabilities instead of the hard label.

#### `POST /predict/batch`

Batch inference for 1–512 rows. A separate `trace_id` is stored for each row.

**Request:**

```json
{
  "rows": [
    {
      "tenure": 6,  "monthly_charges": 89.0, "total_charges": 534.0,
      "contract_type": "Month-to-month", "internet_service": "Fiber optic", "tech_support": "No"
    },
    {
      "tenure": 48, "monthly_charges": 45.0, "total_charges": 2160.0,
      "contract_type": "Two year", "internet_service": "DSL", "tech_support": "Yes"
    }
  ],
  "return_proba": false
}
```

**Response:**

```json
{
  "trace_ids": ["f47ac10b-...", "a1b2c3d4-..."],
  "predictions": [1, 0],
  "model_version": "3",
  "latency_ms": 6.88
}
```

---

### Agentic-Ops Endpoints

#### `GET /agent/schema`

Returns an **OpenAI-compatible tool specification** describing the `predict_churn` function. Any function-calling LLM fetches this once and can then autonomously decide when and how to call `/agent/invoke`.

Compatible with: OpenAI function-calling, Anthropic tool-use, LangChain `Tool`, LlamaIndex `FunctionTool`, AutoGen agents.

```json
{
  "type": "function",
  "function": {
    "name": "predict_churn",
    "description": "Predict customer churn probability using the registered ML model. Call this whenever you need a churn score for a single customer.",
    "parameters": {
      "type": "object",
      "properties": {
        "tenure":           { "type": "integer", "description": "Months as customer" },
        "monthly_charges":  { "type": "number",  "description": "USD per month" },
        "total_charges":    { "type": "number",  "description": "Lifetime USD spend" },
        "contract_type":    { "type": "string",  "enum": ["Month-to-month", "One year", "Two year"] },
        "internet_service": { "type": "string",  "enum": ["DSL", "Fiber optic", "No"] },
        "tech_support":     { "type": "string",  "enum": ["Yes", "No", "No internet service"] },
        "return_proba":     { "type": "boolean", "default": false }
      },
      "required": ["tenure", "monthly_charges", "total_charges", "contract_type", "internet_service", "tech_support"]
    }
  }
}
```

#### `POST /agent/invoke`

LLM-callable inference. Mirrors `/predict` but adds:

- `call_id` — passed through from the agent's tool-call ID so the LLM can correlate responses.
- `explanation_hint` — a plain-English string the LLM can incorporate into its reply to the user without needing a separate explainability call.

**Request:**

```json
{
  "call_id": "call_abc123",
  "features": {
    "tenure": 3,
    "monthly_charges": 95.0,
    "total_charges": 285.0,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "tech_support": "No"
  }
}
```

**Response:**

```json
{
  "call_id": "call_abc123",
  "trace_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
  "result": 1,
  "model_version": "3",
  "latency_ms": 3.94,
  "explanation_hint": "month-to-month contracts are a strong churn driver; short tenure increases churn risk; above-average monthly charges correlate with churn"
}
```

The `explanation_hint` field is a stub in this exercise. Replace it with SHAP or LIME values in production (see Exercise 2 below).

#### `POST /agent/feedback`

Stores a human-corrected label against a prediction trace. This is the entry point for the HITL data flywheel: corrected labels accumulate here and can be drained into a retraining dataset.

**Request:**

```json
{
  "trace_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
  "correct_label": 1,
  "annotator": "alice@company.com",
  "notes": "Customer cancelled subscription the next day."
}
```

**Response:**

```json
{
  "status": "recorded",
  "feedback_count": 7
}
```

Returns 404 if `trace_id` is unknown — enforces that feedback can only be submitted for predictions this server actually made.

#### `GET /agent/trace/{trace_id}`

Retrieves the full prediction trace: inputs, output, model version, timestamp, and latency. Used by guardrail systems that need to inspect what the model decided before taking action, explainability pipelines that attach SHAP values post-hoc, and audit logs in regulated industries.

**Response:**

```json
{
  "trace_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
  "features": {
    "tenure": 3,
    "monthly_charges": 95.0,
    "total_charges": 285.0,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "tech_support": "No"
  },
  "prediction": 1,
  "model_version": "3",
  "timestamp": 1710000123.45,
  "latency_ms": 3.94
}
```

---

## Architecture Deep-Dive

### Startup: Model Loading via Lifespan

FastAPI's `@asynccontextmanager` lifespan replaces the older `@app.on_event("startup")` pattern (deprecated in FastAPI 0.103+). The lifespan function runs setup before `yield` and teardown after:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    _state["model"] = mlflow.pyfunc.load_model(model_uri)
    yield
    # --- shutdown ---
    log.info("Shutting down – bye!")
```

Key design decision: if model loading fails, the server **does not crash**. It logs the error and sets `_state["model"] = None`. The `/health` endpoint reports `"degraded"` and all inference endpoints return `503`. This is preferable to a crash loop in orchestrated environments (Kubernetes, ECS) where the pod will restart but you want to preserve logs and metrics.

### Request Flow

```
Client Request
      │
      ▼
FastAPI Route Handler
      │
      ├── Pydantic validates request body (422 on failure, never reaches inference)
      │
      ├── _require_model() — fast 503 guard if model not loaded
      │
      ├── _run_inference() — builds DataFrame, calls model.predict()
      │
      ├── Increment _state["request_count"]
      │
      ├── generate trace_id (UUID4)
      │
      ├── Return PredictResponse immediately  ◄── client gets response here
      │
      └── BackgroundTask: _store_trace()      ◄── runs after response is sent
```

### Pydantic Schemas

All request and response bodies are typed Pydantic models. This gives you:

- Runtime validation with structured error messages
- Automatic JSON serialisation/deserialisation
- OpenAPI schema generation for `/docs`

The key schema pairs are:

| Request | Response |
|---|---|
| `PredictRequest` | `PredictResponse` |
| `BatchPredictRequest` | `BatchPredictResponse` |
| `AgentInvokeRequest` | `AgentInvokeResponse` |
| `FeedbackPayload` | *(plain dict)* |
| *(path param)* | `PredictionTrace` |

### Background Tasks and Trace Storage

`BackgroundTasks` in FastAPI lets you run a function *after the response has been sent to the client*. This is used for trace storage:

```python
def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    prediction, latency_ms = _run_inference(req.features, req.return_proba)
    trace_id = str(uuid.uuid4())
    background_tasks.add_task(_store_trace, trace_id, req.features, prediction, latency_ms)
    return PredictResponse(...)   # ← client receives this immediately
    # _store_trace() runs after this return
```

The client sees zero added latency from storage. The trade-off is that if the server crashes between the response and the background task completing, the trace is lost. For production, push traces to a Redis queue or a write-ahead log instead.

The in-memory `_state["traces"]` dict is fine for a single-process dev server. Replace with Redis or PostgreSQL when you have multiple workers or need persistence across restarts.

### The Agentic-Ops Layer

The four `/agent/*` endpoints implement a complete **tool-calling contract** that any LLM framework understands:

```
┌─────────────────────────────────────────────────────────────┐
│                        LLM Agent                            │
│                                                             │
│  1. GET /agent/schema  ──► reads tool spec                  │
│                            (done once; cached by the agent) │
│                                                             │
│  2. POST /agent/invoke ──► calls model as function          │
│          │                 receives result + hint           │
│          │                                                  │
│          └──► narrates result to user using hint            │
│                                                             │
│  3. POST /agent/feedback ◄── human says "actually wrong"    │
│                               label stored for retraining   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Guardrail / Audit System                    │
│                                                             │
│  GET /agent/trace/{id} ──► retrieves exact inputs+outputs   │
│                            for any past inference           │
└─────────────────────────────────────────────────────────────┘
```

The `explanation_hint` field in `/agent/invoke` responses deserves attention. Rather than forcing the agent to make a second call to an explainability API, the serving layer returns a pre-computed plain-English string the LLM can splice directly into its reply. This keeps the agent's reasoning loop tight and reduces total latency.
---

'''


## Code Walkthrough

### Configuration

```python
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME          = os.getenv("REGISTERED_MODEL_NAME", "churn_classifier")
MODEL_ALIAS         = os.getenv("MODEL_ALIAS", "champion")
```

All three are overridable via environment variables, so the same image can serve different model versions in different environments without rebuilding.

### State Management

```python
_state: dict[str, Any] = {
    "model": None,          # mlflow.pyfunc.PyFuncModel
    "model_version": None,  # str, e.g. "3"
    "loaded_at": None,      # float (Unix timestamp)
    "request_count": 0,
    "error_count": 0,
    "traces": {},           # trace_id → PredictionTrace
    "feedback": [],         # list[FeedbackPayload]
}
```

A module-level dict is the simplest way to share state across route handlers in a single-process FastAPI app. The comment on every field says what to swap it for in production: `traces` → Redis, `feedback` → PostgreSQL, `request_count` / `error_count` → Prometheus counters.

### `/predict` — Single-row inference

The handler is intentionally thin. All the interesting logic lives in `_run_inference()`:

```python
def _run_inference(features: dict, return_proba: bool) -> tuple[Any, float]:
    _require_model()
    t0 = time.perf_counter()
    df = pd.DataFrame([features])           # model expects a DataFrame
    if return_proba and hasattr(_state["model"], "predict_proba"):
        result = _state["model"].predict_proba(df).tolist()
    else:
        raw = _state["model"].predict(df)
        result = raw.tolist() if hasattr(raw, "tolist") else raw
        result = result[0] if isinstance(result, list) and len(result) == 1 else result
    latency_ms = (time.perf_counter() - t0) * 1000
    return result, latency_ms
```

The `hasattr(model, "predict_proba")` guard matters: MLflow's `pyfunc` wrapper does not always expose `predict_proba` even for classifiers, depending on how the model was logged. The guard prevents a silent fallback to `predict` without the caller knowing.

### `/predict/batch` — Batch inference

Batch inference builds a DataFrame from all rows in one call, which is far more efficient than looping over single-row predictions:

```python
df = pd.DataFrame(req.rows)          # shape: (n_rows, n_features)
raw = _state["model"].predict(df)    # single vectorised call
preds = raw.tolist()
```

The `max_length=512` constraint on `BatchPredictRequest.rows` prevents runaway memory usage from huge payloads. Tune this based on your model's memory footprint.

### `/agent/schema` — Tool discovery

The response is hand-crafted JSON that matches the OpenAI function-calling spec exactly. The `enum` constraints on categorical features are important: they prevent the LLM from hallucinating invalid values like `"Monthly"` for `contract_type`.

In a more sophisticated setup, generate this schema programmatically from the MLflow model signature:

```python
# Future enhancement
from mlflow.models import ModelSignature
sig: ModelSignature = mlflow.models.get_model_info(model_uri).signature
```

### `/agent/invoke` — LLM-callable inference

This endpoint is functionally identical to `/predict` with two additions:

- The `call_id` is echoed back so the LLM can match responses to its tool-call IDs (required by OpenAI's tool-use protocol).
- `explanation_hint` is computed by `_build_explanation_hint()`, a stub that applies heuristic rules. Replace this with a real explainer:

```python
# Production replacement — SHAP for tree models
import shap
explainer = shap.TreeExplainer(underlying_model)
shap_values = explainer.shap_values(df)
top_features = sorted(zip(feature_names, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)[:3]
hint = "; ".join(f"{name} (SHAP: {val:+.3f})" for name, val in top_features)
```

### `/agent/feedback` — HITL data collection

The 404 guard is the critical piece:

```python
if payload.trace_id not in _state["traces"]:
    raise HTTPException(status_code=404, detail=f"trace_id {payload.trace_id!r} not found.")
```

This enforces referential integrity: you cannot submit feedback for a prediction that wasn't made by this server in this session. In production, move the trace store to a database so the check persists across restarts and multiple workers.

### `/agent/trace/{id}` — Audit log

A simple dict lookup. The design choice here is that traces are stored by the inference endpoints as a background task (low overhead) and retrieved on-demand. This is appropriate when retrieval is infrequent (audits, investigations). If a guardrail system needs to intercept *every* prediction synchronously, push traces to a queue and have the guardrail consume from it instead.

---

## Running the Tests

```bash
pytest tests/ -v
```

The test suite uses `unittest.mock.patch` to intercept the two MLflow calls that happen at import time (`mlflow.pyfunc.load_model` and `MlflowClient.get_model_version_by_alias`), injecting a `MagicMock` model whose `predict` and `predict_proba` methods return deterministic results sized to match the input DataFrame.

```
tests/test_api.py::TestHealth::test_ok                              PASSED
tests/test_api.py::TestHealth::test_uptime_positive                 PASSED
tests/test_api.py::TestMetrics::test_metrics_text                   PASSED
tests/test_api.py::TestModelInfo::test_version_present              PASSED
tests/test_api.py::TestPredict::test_basic                          PASSED
tests/test_api.py::TestPredict::test_trace_id_is_uuid               PASSED
tests/test_api.py::TestPredict::test_return_proba                   PASSED
tests/test_api.py::TestPredict::test_missing_features_field         PASSED
tests/test_api.py::TestBatchPredict::test_basic                     PASSED
tests/test_api.py::TestBatchPredict::test_empty_rows_rejected       PASSED
tests/test_api.py::TestAgentSchema::test_returns_openai_tool_format PASSED
tests/test_api.py::TestAgentSchema::test_schema_has_descriptions    PASSED
tests/test_api.py::TestAgentInvoke::test_basic                      PASSED
tests/test_api.py::TestAgentInvoke::test_custom_call_id_preserved   PASSED
tests/test_api.py::TestAgentInvoke::test_explanation_hint_present_for_high_risk PASSED
tests/test_api.py::TestAgentFeedback::test_record_feedback          PASSED
tests/test_api.py::TestAgentFeedback::test_feedback_unknown_trace   PASSED
tests/test_api.py::TestAgentTrace::test_retrieve_trace              PASSED
tests/test_api.py::TestAgentTrace::test_missing_trace_404           PASSED

19 passed in 7.51s
```

No live MLflow server, no live FastAPI server — `TestClient` runs the ASGI app in-process.

---

## Exercises

These build on each other. Complete them in order.

### Exercise 1 — Model hot-reload

Add a `POST /model/reload` endpoint that re-fetches the `@champion` alias from the registry and swaps the in-memory model, without restarting the server.

Things to handle:

- Concurrent requests arriving during the reload window — use `asyncio.Lock` to prevent two reloads running simultaneously.
- The old model should remain in service until the new one is fully loaded (`_state["model"]` should only be replaced after `mlflow.pyfunc.load_model()` returns successfully).
- The endpoint should return the old and new version numbers so the caller can confirm the swap happened.

```python
# Skeleton
import asyncio
_reload_lock = asyncio.Lock()

@app.post("/model/reload", tags=["Ops"])
async def reload_model():
    async with _reload_lock:
        old_version = _state["model_version"]
        # ... load new model ...
        return {"old_version": old_version, "new_version": _state["model_version"]}
```

### Exercise 2 — Real SHAP explanations

Replace the heuristic stubs in `_build_explanation_hint()` with actual SHAP values.

```bash
pip install shap
```

For the Logistic Regression pipeline produced by `register_dummy_model.py`, use `shap.LinearExplainer`. For tree-based models (XGBoost, LightGBM, RandomForest), use `shap.TreeExplainer`.

The explanation hint should return the top-3 features by absolute SHAP value, formatted as a string the LLM can read naturally:

```
"monthly_charges (+0.42); contract_type (+0.38); tenure (-0.21)"
```

Pre-compute the explainer at startup (it's expensive to initialise) and store it in `_state`.

### Exercise 3 — Prometheus integration

Replace the hand-rolled `/metrics` endpoint with `prometheus_fastapi_instrumentator`:

```bash
pip install prometheus-fastapi-instrumentator
```

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

This gives you a proper `histogram` for request latency (not just counters), compatible with Grafana dashboards. After adding it, configure a local Prometheus instance to scrape `http://localhost:8000/metrics` and visualise the latency histogram.

### Exercise 4 — Wire a real LLM agent

Use the OpenAI Python SDK (or Anthropic's) to build a simple agent that:

1. Fetches the tool schema from `GET /agent/schema`.
2. Accepts a natural-language customer description (e.g. *"New customer, 3 months in, paying $95/month, month-to-month contract, fibre internet, no tech support"*).
3. Extracts the structured features and calls `POST /agent/invoke`.
4. Returns a natural-language churn assessment to the user, incorporating the `explanation_hint`.

```python
# OpenAI function-calling skeleton
import openai, httpx, json

schema = httpx.get("http://localhost:8000/agent/schema").json()
client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Customer description ..."}],
    tools=[schema],
    tool_choice="auto",
)

# Handle tool_calls in response.choices[0].message.tool_calls
```

### Exercise 5 — Online retraining loop

Drain the feedback records accumulated by `POST /agent/feedback` and use them to retrain the model:

1. Add a `GET /agent/feedback/export` endpoint that returns all stored feedback as a JSON array.
2. Write a script (`scripts/retrain_from_feedback.py`) that:
   - Fetches feedback from the export endpoint
   - Fetches the corresponding traces from `GET /agent/trace/{id}`
   - Joins them into a labelled DataFrame (features + correct_label)
   - Retrains the model on the combined original training data + corrections
   - Registers the new version in MLflow and sets the `@champion` alias
3. Trigger a hot-reload via `POST /model/reload`.

This closes the full human-in-the-loop cycle: inference → human feedback → retraining → deployment → inference.

---

## Production Upgrade Path

The in-memory design of this exercise is intentional — it keeps the code readable. Here is the upgrade path for each component:

| Component | Dev (this exercise) | Production |
|---|---|---|
| Trace store | `dict` in `_state` | Redis (TTL-based) or PostgreSQL |
| Feedback store | `list` in `_state` | PostgreSQL + async writer |
| Metrics | Hand-rolled plain text | `prometheus_fastapi_instrumentator` |
| Model state | Module-level `_state` dict | Single worker with `asyncio.Lock`, or Redis for multi-worker |
| Model loading | Synchronous at startup | Async with `asyncio.to_thread()` for large models |
| Batch size limit | Pydantic `max_length=512` | Queue-based batching with dynamic max wait (Triton, Ray Serve) |
| Explanation hints | Heuristic stubs | SHAP pre-computed at load time, cached per feature hash |
| Authentication | None | OAuth2 / API key middleware |
| TLS | None | Terminate at load balancer or use `uvicorn --ssl-*` |

---

## Key Takeaways

**MLflow aliases decouple serving from versioning.** The model URI `models:/churn_classifier@champion` stays constant in your serving code; the MLOps team moves the alias to promote or roll back versions.

**FastAPI lifespan is the right place for one-time setup.** Loading a model in `lifespan()` means it happens once, before the first request, and the reference lives for the whole process lifetime.

**BackgroundTasks keep inference latency clean.** Trace storage, logging, and metrics updates that don't affect the response should always be backgrounded.

**Fail gracefully on model load errors.** A server that reports `"degraded"` health is easier to debug than one that crash-loops. The `/health` endpoint is your first debugging surface.

**Agent-native design is additive, not a replacement.** The `/agent/*` endpoints sit alongside the standard `/predict` endpoints. Human users and application code use the REST layer; LLM agents use the agentic layer. The same model, the same inference logic, two different calling contracts.

**The tool schema is the contract.** The most important thing in the agentic layer is that `/agent/schema` is accurate, versioned alongside the model, and includes `enum` constraints for all categorical inputs. An LLM with a bad schema will hallucinate invalid inputs.

**The feedback endpoint is where the data flywheel starts.** Every prediction is traceable; every incorrect prediction can be corrected; those corrections accumulate into a retraining dataset. Designing this flow from day one is far cheaper than retrofitting it later.

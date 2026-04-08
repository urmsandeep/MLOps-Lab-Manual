# Exercise 10 — End-to-End ML Pipeline

## Scenario: You are an MLOps Engineer at QuickFoods

You have built every piece of the QuickFoods ML system across Exercises 1–9:

* A reproducible training pipeline
* Experiment tracking and model comparison
* Hyperparameter tuning
* Docker packaging
* FastAPI serving
* Prediction logging and monitoring
* Model versioning and promotion
* A retraining pipeline

The CTO now asks:

> "Show me the full lifecycle in one run. I want to see: train the model, track the experiment, register it, serve it, send traffic, monitor it, detect drift, retrain, and promote — all in sequence. This is what I want to show the board as proof that we have a production-grade ML system."

Your task:

1. Build a **single orchestration script** that runs the full MLOps lifecycle end-to-end
2. Each stage calls the scripts built in previous exercises
3. The output is a complete audit trail: experiments, registry versions, prediction logs, monitoring report

This is the capstone exercise. It ties everything together.

---

## Learning Objectives

By the end of this exercise you will be able to:

* Run the entire MLOps lifecycle from a single entry point
* Explain each stage and its purpose in sequence
* Demonstrate a working ML system in an interview or review
* Identify which stage to debug when something goes wrong in production

---

## Prerequisites

* Exercises 1–9 completed (all scripts exist and have been run at least once)
* MLflow tracking running locally
* Docker installed

---

## The Full Lifecycle — What Happens in Sequence

```
┌─────────────────────────────────────────────────────────┐
│                  MLOps Full Lifecycle                    │
│                                                         │
│  1. TRAIN        → Train baseline model                 │
│  2. TRACK        → Log experiment to MLflow             │
│  3. EVALUATE     → Compare models, select best          │
│  4. TUNE         → Hyperparameter search                │
│  5. REGISTER     → Push best model to registry          │
│  6. SERVE        → Start FastAPI with champion model    │
│  7. PREDICT      → Send traffic, log predictions        │
│  8. MONITOR      → Check drift and prediction stats     │
│  9. RETRAIN      → Train on new data, compare           │
│  10. PROMOTE     → Swap champion if improved            │
│                                                         │
│  Then: loop back to step 7 (serve → predict → monitor)  │
└─────────────────────────────────────────────────────────┘
```

---

## Step 1 — Create the Orchestration Script

Create `src/run_full_pipeline.py`:

```python
import os
import sys
import subprocess
import time
import json
import signal
import requests

# ── Config ─────────────────────────────────────────────────────────────────────

API_URL         = "http://127.0.0.1:8000"
PREDICT_URL     = f"{API_URL}/predict"
HEALTH_URL      = f"{API_URL}/health"
LOG_PATH        = "logs/predictions.jsonl"


def run_script(description: str, script: str):
    """Run a Python script and print its output."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {description}")
    print(f"  Running: python {script}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        print(f"\n❌ FAILED: {script} exited with code {result.returncode}")
        sys.exit(1)

    print(f"\n✅ {description} — complete\n")


def start_api():
    """Start the FastAPI server in the background."""
    print(f"\n{'='*60}")
    print(f"  STAGE: Start API Server")
    print(f"{'='*60}\n")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for the server to be ready
    for attempt in range(20):
        time.sleep(1)
        try:
            resp = requests.get(HEALTH_URL, timeout=2)
            if resp.status_code == 200:
                print(f"✅ API is healthy: {resp.json()}")
                return proc
        except requests.ConnectionError:
            pass

    print("❌ API failed to start within 20 seconds")
    proc.terminate()
    sys.exit(1)


def send_test_traffic(n_normal=15, n_drifted=10):
    """Send a mix of normal and drifted requests."""
    print(f"\n{'='*60}")
    print(f"  STAGE: Send Production Traffic ({n_normal} normal + {n_drifted} drifted)")
    print(f"{'='*60}\n")

    import random
    random.seed(42)

    for i in range(n_normal):
        payload = {
            "distance_km": round(random.uniform(0.5, 10.0), 1),
            "items_count": random.randint(1, 6),
            "is_peak_hour": random.choice([0, 1]),
            "traffic_level": random.choice([1, 2, 3]),
        }
        resp = requests.post(PREDICT_URL, json=payload)
        pred = resp.json()["delivery_time_min"]
        print(f"  [Normal  {i+1:02d}] dist={payload['distance_km']:5.1f} → {pred:.1f} min")

    print("\n  --- Drift begins ---\n")

    for i in range(n_drifted):
        payload = {
            "distance_km": round(random.uniform(15.0, 25.0), 1),
            "items_count": random.randint(7, 12),
            "is_peak_hour": 1,
            "traffic_level": 3,
        }
        resp = requests.post(PREDICT_URL, json=payload)
        pred = resp.json()["delivery_time_min"]
        print(f"  [Drifted {i+1:02d}] dist={payload['distance_km']:5.1f} → {pred:.1f} min")

    print(f"\n✅ Sent {n_normal + n_drifted} predictions\n")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     QuickFoods MLOps — Full Lifecycle Pipeline          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Clean up previous prediction logs
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
        print(f"Cleaned previous log: {LOG_PATH}")

    # ── Stage 1: Train baseline ───────────────────────────────────────
    run_script(
        "Train Baseline Model (Exercise 1)",
        "src/train.py"
    )

    # ── Stage 2: Train with MLflow tracking ───────────────────────────
    run_script(
        "Train with MLflow Tracking (Exercise 2)",
        "src/train_with_mlflow.py"
    )

    # ── Stage 3: Multi-metric comparison ──────────────────────────────
    run_script(
        "Multi-Metric Model Comparison (Exercise 3)",
        "src/train_multi_metrics_with_mlflow.py"
    )

    # ── Stage 4: Hyperparameter tuning ────────────────────────────────
    run_script(
        "Hyperparameter Tuning (Exercise 5)",
        "src/train_hyperparameter_tuning.py"
    )

    # ── Stage 5: Register best model ──────────────────────────────────
    run_script(
        "Register Best Model (Exercise 5)",
        "src/register_best_model.py"
    )

    # ── Stage 6: Promote to champion ──────────────────────────────────
    run_script(
        "Assign Champion Alias (Exercise 8)",
        "src/promote_model.py"
    )

    # ── Stage 7: Start API and serve ──────────────────────────────────
    api_proc = start_api()

    try:
        # ── Stage 8: Send traffic ─────────────────────────────────────
        send_test_traffic(n_normal=15, n_drifted=10)

        # ── Stage 9: Monitor ──────────────────────────────────────────
        run_script(
            "Monitor Predictions and Check Drift (Exercise 7)",
            "src/monitor.py"
        )

    finally:
        # Stop the API server
        print("\nStopping API server...")
        api_proc.terminate()
        api_proc.wait(timeout=5)
        print("API server stopped.\n")

    # ── Stage 10: Retrain ─────────────────────────────────────────────
    run_script(
        "Retrain on New Data and Promote (Exercise 9)",
        "src/retrain.py"
    )

    # ── Summary ───────────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════╗")
    print("║               Pipeline Complete                        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║                                                        ║")
    print("║  ✅ Trained baseline model                              ║")
    print("║  ✅ Tracked experiments in MLflow                       ║")
    print("║  ✅ Compared 3 models on multiple metrics               ║")
    print("║  ✅ Tuned hyperparameters (grid + random search)        ║")
    print("║  ✅ Registered best model in MLflow registry            ║")
    print("║  ✅ Served model via FastAPI                            ║")
    print("║  ✅ Logged predictions (normal + drifted traffic)       ║")
    print("║  ✅ Detected input drift via monitoring                 ║")
    print("║  ✅ Retrained on combined data                          ║")
    print("║  ✅ Promoted new champion (if improved)                 ║")
    print("║                                                        ║")
    print("║  View full history: mlflow ui → http://127.0.0.1:5000  ║")
    print("║  Prediction log: logs/predictions.jsonl                ║")
    print("║                                                        ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
```

---

## Step 2 — Run the Full Pipeline

```
python src/run_full_pipeline.py
```

This will take 1–2 minutes. Watch each stage execute in sequence.

---

## Step 3 — Inspect the Results

After the pipeline completes:

**MLflow UI:**

```
mlflow ui
```

Open `http://127.0.0.1:5000` and explore:

* **Experiments tab** — all training runs across exercises, nested sweep runs, retraining run
* **Models tab** — `quickfoods-delivery-predictor` with multiple versions, the latest marked as `champion`

**Prediction log:**

```
cat logs/predictions.jsonl | head -5
```

Each line is a JSON record with timestamp, input features, prediction, and model info.

**Model Registry:**

```
python -c "
from mlflow.tracking import MlflowClient
client = MlflowClient()
for v in client.search_model_versions(\"name='quickfoods-delivery-predictor'\"):
    print(f'Version {v.version} | Aliases: {v.aliases}')
"
```

---

## The Five Questions You Can Now Answer

From the Learning Map, the end goal was to be able to answer these:

**1. Which model is currently deployed?**

→ The `champion` alias in the MLflow Model Registry. Check with `/health` endpoint or `MlflowClient`.

**2. How accurate is it in production?**

→ Read `logs/predictions.jsonl`, compute MAE against actual delivery times (when available). The monitoring script reports prediction distribution stats.

**3. How do you detect model issues?**

→ The monitoring script (`src/monitor.py`) compares live feature distributions to training data and raises alerts when thresholds are breached.

**4. When should you retrain the model?**

→ When monitoring alerts are sustained — not one-off spikes. The retraining script compares old vs new performance and only promotes if metrics improve.

**5. What changed between model versions?**

→ Every version in the registry links to its MLflow run, which records the exact parameters, metrics, data, and code that produced it.

---

## What We Learned

* The MLOps lifecycle is a loop, not a straight line: train → deploy → monitor → retrain → deploy
* Every stage produces traceable artifacts: models, metrics, logs, registry versions
* Automation does not mean unattended — humans make the promotion decision based on metrics
* This exercise demonstrates a complete, working ML system that can be explained in any technical interview

---

## What This Exercise Is NOT

This pipeline runs locally and sequentially. In a production environment:

* Training runs on dedicated GPU/CPU clusters
* The API runs in Kubernetes or a managed service
* Monitoring is continuous (Prometheus + Grafana, Evidently)
* Retraining is triggered by an orchestrator (Airflow, Prefect)
* Promotion goes through CI/CD with automated tests

But the **logic and workflow** are exactly the same. What you built here scales — the concepts do not change, only the infrastructure.

---

## Key Questions and Answers

**Q: Can this pipeline run on a schedule?**

A: Yes. Wrap `run_full_pipeline.py` in a cron job or Airflow DAG. In practice, you would not retrain every time — only when monitoring signals indicate the need.

**Q: What is the most common failure point in production ML?**

A: Silent model degradation — the model still returns predictions but they become less accurate over time. This is exactly what monitoring (Exercise 7) catches.

**Q: How would you explain this system in a 5-minute interview answer?**

A: "We train models using scikit-learn and track every experiment in MLflow. The best model goes through hyperparameter tuning, gets registered in the model registry, and is served via FastAPI. Every prediction is logged. A monitoring script compares live inputs to training distributions and alerts us when drift is detected. When alerts fire, we retrain on combined old + new data, compare against the current champion, and promote only if metrics improve. The full lifecycle is automated and auditable."

**Q: What would you add next?**

A: In order of priority: automated tests for the API, CI/CD pipeline for model deployment, a proper database for prediction logs, A/B testing framework for comparing models on live traffic, and infrastructure-as-code for the serving environment.

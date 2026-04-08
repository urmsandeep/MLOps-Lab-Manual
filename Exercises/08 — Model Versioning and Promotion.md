# Exercise 08 — Model Versioning and Promotion

## Scenario: You are an MLOps Engineer at QuickFoods

QuickFoods now has:

* Multiple trained models tracked in MLflow (Exercises 2, 3, 5)
* A registered model `quickfoods-delivery-predictor` in the MLflow Model Registry (Exercise 5)
* A FastAPI service serving predictions (Exercise 6)
* A monitoring system detecting drift (Exercise 7)

The Engineering Lead asks:

> "We have three model versions now. How do we safely promote a new model to production without breaking the live service? And how do we roll back if something goes wrong?"

Your task:

1. Register multiple model versions in the MLflow Model Registry
2. Use **aliases** (or stage transitions) to manage lifecycle: `champion` and `challenger`
3. Update the FastAPI service to **load the model from the registry** instead of a hardcoded file path
4. Demonstrate a safe promotion workflow: challenger → champion

---

## Learning Objectives

By the end of this exercise you will be able to:

* Register multiple model versions under one registered model name
* Assign aliases (`champion`, `challenger`) to control which version is active
* Load a model from the MLflow registry in a serving application
* Swap the production model without redeploying the service
* Explain model versioning and promotion in an interview

---

## Prerequisites

* Exercises 1–7 completed
* MLflow tracking server running locally (default: `mlruns/` directory)
* At least one model registered from Exercise 5

---

## Important Concept: Model Registry Lifecycle

The MLflow Model Registry stores models by **name** and **version**:

```
quickfoods-delivery-predictor
├── Version 1  — GradientBoosting (from Exercise 5)   alias: champion
├── Version 2  — RandomForest (retrained)              alias: challenger
└── Version 3  — GradientBoosting (retrained on new data)
```

**Aliases** (MLflow 2.x+) replace the old stage-based system (Staging/Production/Archived):

* `champion` — the version currently serving production traffic
* `challenger` — a candidate being evaluated before promotion

Only one version can hold a given alias at a time. Moving the `champion` alias to a new version is an atomic swap.

---

## Step 1 — Register a Second Model Version

In Exercise 5, we registered version 1. Now let us train a slightly different model and register it as version 2.

Create `src/train_v2.py`:

```python
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

DATA_PATH       = "data/delivery_times.csv"
MODEL_DIR       = "models"
EXPERIMENT_NAME = "quickfoods-delivery-time"
REGISTERED_NAME = "quickfoods-delivery-predictor"


def main():
    print("=== QuickFoods: Train and Register Model V2 ===")

    df = pd.read_csv(DATA_PATH)
    X  = df[["distance_km", "items_count", "is_peak_hour", "traffic_level"]]
    y  = df["delivery_time_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=99  # different split
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="RandomForest-V2"):
        mlflow.set_tag("project", "QuickFoods")
        mlflow.set_tag("purpose", "version_comparison")

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=3,
            random_state=99,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae  = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2   = r2_score(y_test, preds)

        mlflow.log_param("model_name", "RandomForest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 8)
        mlflow.log_param("min_samples_split", 3)
        mlflow.log_param("random_state", 99)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model in MLflow format
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

        # Register as new version
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/sklearn-model"
        mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_NAME)

        print(f"MAE  : {mae:.3f}")
        print(f"RMSE : {rmse:.3f}")
        print(f"R²   : {r2:.3f}")
        print(f"\n✅ Registered '{REGISTERED_NAME}' version {mv.version}")


if __name__ == "__main__":
    main()
```

Run:

```
python src/train_v2.py
```

Expected output:

```
=== QuickFoods: Train and Register Model V2 ===
MAE  : 4.XXX
RMSE : 5.XXX
R²   : 0.9XX

✅ Registered 'quickfoods-delivery-predictor' version 2
```

---

## Step 2 — Assign Aliases

Create `src/promote_model.py`:

```python
from mlflow.tracking import MlflowClient

REGISTERED_NAME = "quickfoods-delivery-predictor"


def list_versions(client):
    """Print all versions of the registered model."""
    print(f"\nAll versions of '{REGISTERED_NAME}':")
    # search_model_versions returns all versions
    versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    for v in versions:
        aliases = v.aliases if hasattr(v, "aliases") else []
        print(f"  Version {v.version} | Run ID: {v.run_id[:8]}... | Aliases: {aliases}")
    return versions


def main():
    print("=== QuickFoods: Model Promotion Workflow ===")

    client = MlflowClient()

    versions = list_versions(client)

    if len(versions) < 2:
        print("\nNeed at least 2 versions. Run train_v2.py first.")
        return

    # Assign version 1 as champion (current production)
    client.set_registered_model_alias(REGISTERED_NAME, "champion", "1")
    print("\n→ Set version 1 as 'champion' (current production model)")

    # Assign version 2 as challenger (candidate for promotion)
    client.set_registered_model_alias(REGISTERED_NAME, "challenger", "2")
    print("→ Set version 2 as 'challenger' (candidate under evaluation)")

    list_versions(client)

    print("\n--- Simulating promotion after evaluation ---\n")

    # After testing: promote challenger to champion
    client.set_registered_model_alias(REGISTERED_NAME, "champion", "2")
    print("→ Promoted version 2 to 'champion'")

    # Remove challenger alias (version 2 is now champion, no longer challenger)
    client.delete_registered_model_alias(REGISTERED_NAME, "challenger")
    print("→ Removed 'challenger' alias from version 2")

    list_versions(client)

    print("\nThe FastAPI service can now load the 'champion' alias to always get the current best model.")
    print("No redeployment needed — just restart the service or implement hot-reload.")


if __name__ == "__main__":
    main()
```

Run:

```
python src/promote_model.py
```

Expected output:

```
=== QuickFoods: Model Promotion Workflow ===

All versions of 'quickfoods-delivery-predictor':
  Version 1 | Run ID: a3f8c1... | Aliases: []
  Version 2 | Run ID: 7b2e09... | Aliases: []

→ Set version 1 as 'champion' (current production model)
→ Set version 2 as 'challenger' (candidate under evaluation)

All versions of 'quickfoods-delivery-predictor':
  Version 1 | Run ID: a3f8c1... | Aliases: ['champion']
  Version 2 | Run ID: 7b2e09... | Aliases: ['challenger']

--- Simulating promotion after evaluation ---

→ Promoted version 2 to 'champion'
→ Removed 'challenger' alias from version 2

All versions of 'quickfoods-delivery-predictor':
  Version 1 | Run ID: a3f8c1... | Aliases: []
  Version 2 | Run ID: 7b2e09... | Aliases: ['champion']

The FastAPI service can now load the 'champion' alias to always get the current best model.
No redeployment needed — just restart the service or implement hot-reload.
```

---

## Step 3 — Update FastAPI to Load from the Registry

Update `src/api.py` to load the champion model from the registry instead of a hardcoded `.pkl` path:

```python
import os
import json
import mlflow
import pandas as pd
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


REGISTERED_NAME = "quickfoods-delivery-predictor"
CHAMPION_ALIAS  = "champion"
LOG_DIR         = "logs"
LOG_PATH        = os.path.join(LOG_DIR, "predictions.jsonl")

app = FastAPI(
    title="QuickFoods Delivery Time Prediction API",
    description="Serves the champion model from MLflow Registry",
    version="3.0.0"
)

os.makedirs(LOG_DIR, exist_ok=True)

# Load the champion model from the registry
model_uri = f"models:/{REGISTERED_NAME}@{CHAMPION_ALIAS}"
print(f"Loading model from: {model_uri}")
model = mlflow.sklearn.load_model(model_uri)
print("Model loaded successfully.")


class DeliveryRequest(BaseModel):
    distance_km: float = Field(..., gt=0)
    items_count: int = Field(..., gt=0)
    is_peak_hour: int = Field(..., ge=0, le=1)
    traffic_level: int = Field(..., ge=1, le=3)


class PredictionResponse(BaseModel):
    delivery_time_min: float


def log_prediction(request_data: dict, prediction: float):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": request_data,
        "prediction": prediction,
        "model": REGISTERED_NAME,
        "alias": CHAMPION_ALIAS,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": REGISTERED_NAME,
        "alias": CHAMPION_ALIAS,
        "model_uri": model_uri,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: DeliveryRequest):
    try:
        input_dict = {
            "distance_km": request.distance_km,
            "items_count": request.items_count,
            "is_peak_hour": request.is_peak_hour,
            "traffic_level": request.traffic_level,
        }
        input_df = pd.DataFrame([input_dict])
        prediction = round(float(model.predict(input_df)[0]), 2)

        log_prediction(input_dict, prediction)

        return PredictionResponse(delivery_time_min=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
```

Run:

```
uvicorn src.api:app --reload
```

Test:

```
curl http://127.0.0.1:8000/health
```

The `/health` response now shows which registered model and alias is being served.

---

## Step 4 — Rollback Demonstration

If the new champion (version 2) performs poorly in production, rolling back is one command:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.set_registered_model_alias("quickfoods-delivery-predictor", "champion", "1")
print("Rolled back to version 1")
```

Restart the FastAPI service and it loads version 1 again. No code change. No redeployment.

---

## What We Learned

* The MLflow Model Registry provides a central place to manage model versions
* **Aliases** (`champion`, `challenger`) decouple model identity from version numbers
* The serving layer loads from the registry — swapping models is an alias change, not a code change
* Rollback is instant: move the alias back to the previous version
* This workflow answers the interview question: "How do you safely update a production model?"

---

## Key Questions and Answers

**Q: What is the difference between a logged model and a registered model?**

A: A **logged model** is an artifact inside a run — it exists in the experiment's artifact store. A **registered model** has a name, version number, and aliases. Only registered models support lifecycle management (promotion, rollback, serving by alias).

**Q: Why use aliases instead of the old Staging/Production stages?**

A: MLflow 2.x introduced aliases as a more flexible system. Stages were limited to a fixed set (Staging, Production, Archived). Aliases are user-defined strings — you can have `champion`, `challenger`, `shadow`, `canary`, or any label that fits your workflow.

**Q: Does changing an alias require restarting the API?**

A: With the code above, yes — the model is loaded at startup. In a production system, you would implement periodic model reloading (e.g. check the registry every N minutes) or use a webhook to trigger a reload when an alias changes.

**Q: What if two people promote different versions at the same time?**

A: The last write wins — alias assignment is atomic. This is why promotion should go through a controlled process (a script with approvals) rather than ad-hoc manual changes.

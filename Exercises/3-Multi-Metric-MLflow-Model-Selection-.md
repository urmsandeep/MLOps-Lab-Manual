# Exercise 3 — Multi-Metric MLflow Tracking and Model Selection

## Scenario: You are an MLOps Engineer at QuickFoods

QuickFoods is a food delivery company. Your team already built a baseline model to predict `delivery_time_min` from order features.

Now the Product Manager says:

> “We tried different model ideas last week. Nobody remembers which model was better and we can’t reproduce the results. We need a proper experiment log so we can compare models and pick the best one confidently.”

As the MLOps Engineer, your task is to:
1. Train multiple models
2. Track each run with parameters, metrics and model artifacts
3. Compare results in the MLflow UI
4. Select the best model based on MAE

This is the first step toward production-grade ML, because real teams must answer:
- Which model version was deployed
- What data and settings produced it
- What metrics justified it

## MLOps Insights and Learning outcomes

Model selection is not:
   **“Which algorithm is cooler?”**
It is:
  **“Which model performs best under agreed metrics and constraints?”**

In production systems, we will:
- Monitor MAE over time
- Trigger retraining when MAE increases
- Set alert thresholds based on business tolerance

By the end of this exercise, you will be able to:
- Use MLflow to track ML experiments locally
- Log parameters and metrics for each training run
- Save and log model artifacts
- Compare runs in the MLflow UI
- Pick the best model using a measurable metric

## Prerequisites

- Python 3.9+
- Git
- A code editor (VS Code recommended)

## Continuation from Exercise 1 and 2

Use the same project file structure built earlier.
https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/2-Tracking-Model-Comparision.md#use-python-virtual-environment


## Log Multiple Regression Metrics
Log multiple regression metrics (MAE, MSE, RMSE, R²), logs model size and inference latency.
For each run, MLflow will show metrics like: mae, mse, rmse, r2, model_size_kb, avg_inference_latency_ms

Note:
- Create a new file: **src/train_multi_metrics_with_mlflow.py** and install required pip/import dependencies.
- Use the data set in: data/delivery_times.csv

```
import os
import json
import time
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


DATA_PATH = "data/delivery_times.csv"
MODEL_DIR = "models"
EXPERIMENT_NAME = "quickfoods-delivery-time"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def split(df: pd.DataFrame):
    X = df[["distance_km", "items_count", "is_peak_hour", "traffic_level"]]
    y = df["delivery_time_min"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def evaluate_regression(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "mse": float(mse), "rmse": rmse, "r2": float(r2)}


def measure_inference_latency_ms(model, X_sample: pd.DataFrame, repeats: int = 200) -> float:
    # Warm up
    _ = model.predict(X_sample)

    start = time.perf_counter()
    for _ in range(repeats):
        _ = model.predict(X_sample)
    end = time.perf_counter()

    avg_sec = (end - start) / repeats
    return float(avg_sec * 1000.0)


def train_and_log(model_name: str, model, params: dict, X_train, X_test, y_train, y_test) -> dict:
    with mlflow.start_run(run_name=model_name):
        # Tags help in filtering runs later
        mlflow.set_tag("project", "QuickFoods")
        mlflow.set_tag("problem_type", "regression")
        mlflow.set_tag("dataset", "delivery_times_v1")

        # Log parameters
        mlflow.log_param("model_name", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Train
        model.fit(X_train, y_train)

        # Predict & metrics
        preds = model.predict(X_test)
        metrics = evaluate_regression(y_test, preds)

        # Log multiple metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Save model artifact locally
        ensure_dir(MODEL_DIR)
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(model, model_path)

        # Compute model size
        model_size_bytes = os.path.getsize(model_path)
        model_size_kb = model_size_bytes / 1024.0
        mlflow.log_metric("model_size_kb", float(model_size_kb))

        # Measure basic inference latency on 1 sample (avg over repeats)
        X_one = X_test.iloc[[0]] if len(X_test) > 0 else X_train.iloc[[0]]
        latency_ms = measure_inference_latency_ms(model, X_one, repeats=200)
        mlflow.log_metric("avg_inference_latency_ms", latency_ms)

        # Log artifacts: model file + JSON report
        mlflow.log_artifact(model_path)

        report = {
            "model_name": model_name,
            "params": params,
            "metrics": metrics,
            "model_size_kb": model_size_kb,
            "avg_inference_latency_ms": latency_ms,
        }
        report_path = os.path.join(MODEL_DIR, f"{model_name}_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)

        # Log model in MLflow format (useful later for registry/deploy)
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

        print(
            f"[OK] {model_name} | "
            f"MAE={metrics['mae']:.3f} | RMSE={metrics['rmse']:.3f} | R2={metrics['r2']:.3f} | "
            f"Size={model_size_kb:.1f}KB | Latency={latency_ms:.3f}ms"
        )

        return {"model_name": model_name, **metrics, "model_size_kb": model_size_kb, "avg_inference_latency_ms": latency_ms}


def main():
    print("=== Exercise 03: MLflow Multi-Metric Tracking (QuickFoods) ===")

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split(df)

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    # Model 1: Baseline
    results.append(
        train_and_log(
            model_name="LinearRegression",
            model=LinearRegression(),
            params={},
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        )
    )

    # Model 2: RandomForest (example parameters)
    results.append(
        train_and_log(
            model_name="RandomForest",
            model=RandomForestRegressor(n_estimators=150, random_state=42),
            params={"n_estimators": 150},
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        )
    )

    # Model 3: GradientBoosting
    results.append(
        train_and_log(
            model_name="GradientBoosting",
            model=GradientBoostingRegressor(random_state=42),
            params={},
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        )
    )

    # Choose "best" model by MAE (business-friendly metric)
    best = sorted(results, key=lambda x: x["mae"])[0]

    print("\n=== Best model by MAE (lower is better) ===")
    print(
        f"Best: {best['model_name']} | "
        f"MAE={best['mae']:.3f} | RMSE={best['rmse']:.3f} | R2={best['r2']:.3f} | "
        f"Size={best['model_size_kb']:.1f}KB | Latency={best['avg_inference_latency_ms']:.3f}ms"
    )

    print("\nNext:")
    print("1) Start MLflow UI: mlflow ui")
    print("2) Open: http://127.0.0.1:5000")
    print("3) Sort by MAE / RMSE and compare trade-offs.")


if __name__ == "__main__":
    main()
```

# How to Interpret MAE vs RMSE vs R²

In this exercise, we logged multiple regression metrics:
- MAE
- MSE
- RMSE
- R²

Understanding what they mean is critical for real-world MLOps decisions.

## 1️⃣ MAE — Mean Absolute Error

Formula:
\[
MAE = \frac{1}{n} \sum |y_{true} - y_{pred}|
\]

### What it tells you:
- On average, how far off are we?
- Same unit as target (minutes)

### Example:
If MAE = 4.2

We can say:
> “On average, our prediction is off by 4.2 minutes.”

This is **business-friendly and interpretable** and Product Manager/Leadership understand this.

### When to use MAE:
- Determine understandable errors
- Business stakeholders care about average error

## 2️⃣ RMSE — Root Mean Squared Error

Formula:
\[
RMSE = \sqrt{\frac{1}{n} \sum (y_{true} - y_{pred})^2}
\]

### What it tells you:
- Penalizes large errors more heavily
- Sensitive to outliers

### Example:
If one prediction is off by 30 minutes,
RMSE increases significantly.

### When to use RMSE:
- Large mistakes are unacceptable
- Want to penalize extreme predictions

## 3️⃣ R² — Coefficient of Determination

Formula:
\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

### What it tells you:
- How much variance in data is explained by the model
- Value ranges:
  - 1 → Perfect fit
  - 0 → Model no better than mean
  - <0 → Worse than baseline

### Example:
If R² = 0.82

It means:
> “The model explains 82% of the variability in delivery time.”

### Important:
R² is useful for technical evaluation but not always intuitive for business users.

# Why We Sort by MAE 

For QuickFoods, the business question is:

> “How many minutes off are we?”

That makes MAE a natural primary metric.

However, in real systems:
- We compare multiple metrics
- We consider latency
- We consider model size
- We consider deployment constraints

# Q & A

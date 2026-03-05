# MLOps Lab 6  Hyperparameter Tuning with MLflow

## Scenario: You are an MLOps Engineer at QuickFoods

QuickFoods now has three tracked models from Exercise 3.
The Product Manager asks:

> "RandomForest gave us the best MAE, but did we try different settings for it?
> Maybe a better configuration exists and we are leaving performance on the table."

As the MLOps Engineer, your task is to:
1. Run a **Grid Search** sweep over RandomForestRegressor hyperparameters
2. Run a **Random Search** sweep over GradientBoostingRegressor hyperparameters
3. Log every candidate as a **nested child run** inside MLflow
4. Programmatically identify and register the best model

This is how production ML teams answer:
- "Did we actually explore the parameter space?"
- "Which exact configuration is in the registry?"
- "Can we reproduce the winning run a month from now?"

---

## Learning Objectives

- Understand the difference between Grid Search and Random Search
- Use MLflow **nested runs** (parent sweep → child trial)
- Log every hyperparameter combination as a tracked, comparable experiment
- Select the best run programmatically and promote it to the Model Registry

---

## Continuation from Exercise 3

Use the same project structure and dataset from previous exercises.
https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/1-QuickFoods-Model-Artifact.md#2-create-the-required-project-structure

The dataset `data/delivery_times.csv` and virtual environment from Exercise 1 are reused as-is.

---

## Prerequisites

- Python 3.9+
- Git
- VS Code (recommended)
- Exercises 1–3 completed

---

## Install Dependencies

Add to requirements.txt if not already present:

```
pandas
scikit-learn
joblib
mlflow
numpy
```

Install:

```
pip install -r requirements.txt
```

---

## MLflow Nested Runs — Quick Concept

In Exercises 2 and 3, each training run was a **flat, independent run**.

In hyperparameter tuning, we run many trials for the same purpose — finding the best configuration.
MLflow **nested runs** keep these organised:

```
Parent Run: HyperparamSweep-QuickFoods
├── Child Run: n_estimators=50,  max_depth=5   →  MAE: 5.10
├── Child Run: n_estimators=100, max_depth=5   →  MAE: 4.87
├── Child Run: n_estimators=100, max_depth=10  →  MAE: 4.53  ✅ best
└── Child Run: n_estimators=200, max_depth=10  →  MAE: 4.61
```

Each child run has its own parameters, metrics, and model artifact.
The parent run holds the summary — best run ID, best MAE.

---

## Step 1 — Create the Tuning Script

Create a new file: `src/train_hyperparameter_tuning.py`

```python
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from itertools import product as cartesian_product
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_PATH       = "data/delivery_times.csv"
MODEL_DIR       = "models"
EXPERIMENT_NAME = "quickfoods-delivery-time"
RANDOM_STATE    = 42
TEST_SIZE       = 0.2

FEATURES = ["distance_km", "items_count", "is_peak_hour", "traffic_level"]
TARGET   = "delivery_time_min"

# ── Hyperparameter Grids ───────────────────────────────────────────────────────

RF_PARAM_GRID = {
    "n_estimators":      [50, 100, 200],
    "max_depth":         [5, 10, None],
    "min_samples_split": [2, 5],
}

GB_PARAM_GRID = {
    "n_estimators":  [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth":     [3, 5],
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def evaluate(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_true, y_pred)
    return {"mae": float(mae), "mse": float(mse), "rmse": rmse, "r2": float(r2)}


# ── Child Run: one hyperparameter trial ───────────────────────────────────────

def run_trial(model_name, model, params, X_train, X_test, y_train, y_test):
    child_name = model_name + " | " + " ".join(f"{k}={v}" for k, v in params.items())

    with mlflow.start_run(run_name=child_name, nested=True):
        mlflow.set_tag("project",     "QuickFoods")
        mlflow.set_tag("sweep_child", "true")
        mlflow.log_param("model_name", model_name)

        for k, v in params.items():
            mlflow.log_param(k, v)

        # Train
        model.fit(X_train, y_train)
        preds   = model.predict(X_test)
        metrics = evaluate(y_test, preds)

        # Cross-validation MAE on the training fold
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=3,
            scoring="neg_mean_absolute_error"
        )
        cv_mae = float(-cv_scores.mean())

        mlflow.log_metric("mae",    metrics["mae"])
        mlflow.log_metric("mse",    metrics["mse"])
        mlflow.log_metric("rmse",   metrics["rmse"])
        mlflow.log_metric("r2",     metrics["r2"])
        mlflow.log_metric("cv_mae", cv_mae)

        # Save model artifact
        os.makedirs(MODEL_DIR, exist_ok=True)
        safe_name  = child_name.replace(" | ", "_").replace("=", "").replace(" ", "_")
        model_path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

        run_id = mlflow.active_run().info.run_id

        print(
            f"  [{model_name}] {params}"
            f" | MAE={metrics['mae']:.3f} | CV_MAE={cv_mae:.3f} | R2={metrics['r2']:.3f}"
        )

        return run_id, metrics["mae"]


# ── Grid Search: RandomForest ──────────────────────────────────────────────────

def grid_search_rf(X_train, X_test, y_train, y_test):
    print("\n=== Grid Search: RandomForestRegressor ===")
    results = []
    keys    = list(RF_PARAM_GRID.keys())
    values  = list(RF_PARAM_GRID.values())

    for combo in cartesian_product(*values):
        params = dict(zip(keys, combo))
        model  = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
        run_id, mae = run_trial(
            "RandomForest", model, params,
            X_train, X_test, y_train, y_test
        )
        results.append({"run_id": run_id, "mae": mae, "params": params, "model": "RandomForest"})

    return results


# ── Random Search: GradientBoosting ───────────────────────────────────────────

def random_search_gb(X_train, X_test, y_train, y_test, n_iter=6):
    print(f"\n=== Random Search: GradientBoostingRegressor  (n_iter={n_iter}) ===")
    rng     = np.random.RandomState(RANDOM_STATE)
    results = []

    for _ in range(n_iter):
        params = {k: rng.choice(v).item() for k, v in GB_PARAM_GRID.items()}
        model  = GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
        run_id, mae = run_trial(
            "GradientBoosting", model, params,
            X_train, X_test, y_train, y_test
        )
        results.append({"run_id": run_id, "mae": mae, "params": params, "model": "GradientBoosting"})

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== QuickFoods MLOps Lab 6: Hyperparameter Tuning ===")

    df = load_data(DATA_PATH)
    X  = df[FEATURES]
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="HyperparamSweep-QuickFoods") as parent:
        mlflow.set_tag("project",    "QuickFoods")
        mlflow.set_tag("sweep_type", "grid+random")
        mlflow.log_param("train_size",   len(X_train))
        mlflow.log_param("test_size",    len(X_test))
        mlflow.log_param("cv_folds",     3)
        mlflow.log_param("random_state", RANDOM_STATE)

        rf_results = grid_search_rf(X_train, X_test, y_train, y_test)
        gb_results = random_search_gb(X_train, X_test, y_train, y_test, n_iter=6)

        all_results = rf_results + gb_results
        best        = min(all_results, key=lambda r: r["mae"])

        mlflow.log_metric("best_mae",   best["mae"])
        mlflow.log_param("best_run_id", best["run_id"])
        mlflow.log_param("best_model",  best["model"])
        mlflow.log_param("best_params", str(best["params"]))

        print(f"\n{'='*60}")
        print(f"Best model : {best['model']}")
        print(f"Best params: {best['params']}")
        print(f"Best MAE   : {best['mae']:.3f} minutes")
        print(f"Run ID     : {best['run_id']}")
        print(f"Parent ID  : {parent.info.run_id}")
        print(f"{'='*60}")
        print("\nNext: mlflow ui  →  open http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
```

---

## Step 2 — Run the Sweep

```
python src/train_hyperparameter_tuning.py
```

Expected output (values will vary slightly):

```
=== QuickFoods MLOps Lab 6: Hyperparameter Tuning ===
Train: 16 rows  |  Test: 4 rows

=== Grid Search: RandomForestRegressor ===
  [RandomForest] {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 2}  | MAE=4.812 | CV_MAE=5.103 | R2=0.941
  [RandomForest] {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 5}  | MAE=4.901 | CV_MAE=5.218 | R2=0.938
  [RandomForest] {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2} | MAE=4.320 | CV_MAE=4.871 | R2=0.956
  ...

=== Random Search: GradientBoostingRegressor  (n_iter=6) ===
  [GradientBoosting] {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3} | MAE=4.105 | CV_MAE=4.790 | R2=0.961
  ...

============================================================
Best model : GradientBoosting
Best params: {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
Best MAE   : 4.105 minutes
Run ID     : a3f8c1...
Parent ID  : 7d2e09...
============================================================

Next: mlflow ui  →  open http://127.0.0.1:5000
```

> Note: With only 20 rows in `delivery_times.csv`, exact metric values will vary between runs.
> The ranking between configurations matters more than the absolute numbers.
> In a real project the same code runs on thousands of rows.

---

## Step 3 — Inspect Nested Runs in the MLflow UI

```
mlflow ui
```

Open: http://127.0.0.1:5000

**What to observe:**

1. Find the run named **HyperparamSweep-QuickFoods** in the `quickfoods-delivery-time` experiment
2. Click the **▶ expand arrow** — all child runs appear underneath
3. Sort child runs by `mae` ascending — the best floats to the top
4. Click any child run → inspect its exact parameters, 5 metrics, and saved model artifact
5. Click the **Chart** tab on the parent → select **Parallel Coordinates** → choose `n_estimators`, `max_depth`, `learning_rate` as axes and `mae` as colour

This reveals which parameters actually move the needle on MAE.

---

## Step 4 — Register the Best Model

Create a new file: `src/register_best_model.py`

```python
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "quickfoods-delivery-time"
REGISTERED_NAME = "quickfoods-delivery-predictor"
METRIC          = "mae"

def main():
    print("=== QuickFoods: Promote Best Tuned Model to Registry ===")

    client     = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found. Run train_hyperparameter_tuning.py first.")

    # Fetch all runs ordered by MAE — includes child runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{METRIC} > 0",
        order_by=[f"metrics.{METRIC} ASC"],
        max_results=100
    )

    # Only child runs have the model artifact we want
    candidates = [r for r in runs if r.data.tags.get("sweep_child") == "true"]

    if not candidates:
        raise ValueError("No child trial runs found. Check that sweep_child tag is set in train_hyperparameter_tuning.py.")

    best     = candidates[0]
    best_mae = best.data.metrics[METRIC]

    print(f"Best run ID : {best.info.run_id}")
    print(f"Best MAE    : {best_mae:.3f} minutes")
    print(f"Model       : {best.data.params.get('model_name')}")
    print(f"Params      : { {k: v for k, v in best.data.params.items() if k != 'model_name'} }")

    model_uri = f"runs:/{best.info.run_id}/sklearn-model"
    mv        = mlflow.register_model(model_uri=model_uri, name=REGISTERED_NAME)

    print(f"\n✅ Registered '{REGISTERED_NAME}'  version {mv.version}")
    print(f"   Status: {mv.status}")
    print(f"\nView in MLflow UI → Models tab → {REGISTERED_NAME}")


if __name__ == "__main__":
    main()
```

Run:

```
python src/register_best_model.py
```

Expected output:

```
=== QuickFoods: Promote Best Tuned Model to Registry ===
Best run ID : a3f8c1...
Best MAE    : 4.105 minutes
Model       : GradientBoosting
Params      : {'n_estimators': '100', 'learning_rate': '0.1', 'max_depth': '3'}

✅ Registered 'quickfoods-delivery-predictor'  version 1
   Status: READY

View in MLflow UI → Models tab → quickfoods-delivery-predictor
```

Open `http://127.0.0.1:5000` → **Models** tab → `quickfoods-delivery-predictor` version 1 links back to the exact child run that produced it.

---

## Grid Search vs Random Search — When to Use Which

| | Grid Search | Random Search |
|---|---|---|
| **How it works** | Every combination in the grid | Randomly sampled combinations |
| **Combinations tried** | `3 × 3 × 2 = 18` for RF grid | `n_iter = 6` regardless of grid size |
| **Best when** | Grid is small, every param matters | Grid is large, budget is limited |
| **Risk** | Slow if grid grows large | May miss the true optimum |

For the QuickFoods dataset (20 rows) both finish in seconds.
On a 1-million-row dataset, Random Search with `n_iter=20` is strongly preferred over an 18-combination Grid Search.

---

## What we Learned

- Grid Search exhaustively tries every combination — guaranteed to find the best within the defined grid
- Random Search samples the space — faster and often good enough when the grid is large
- MLflow **nested runs** keep sweeps organised: parent holds summary, children hold individual trial results
- Every trial is a reproducible, auditable record — any child run can be re-deployed independently
- The `sweep_child` tag makes programmatic filtering reliable when promoting the winner

---

## Key Questions and Answers

**Q: Why not just use `sklearn.model_selection.GridSearchCV` directly?**

A: `GridSearchCV` is faster for local evaluation but does not log each trial to MLflow.
In a real team, every combination must be tracked so you can audit, compare, and reproduce any result.
The manual nested-run approach gives full MLflow visibility at the cost of a few extra lines.

**Q: What does cv_mae tell us that test MAE does not?**

A: `test_mae` is measured on a single held-out split — it can be lucky or unlucky depending on which 4 rows land in the test set.
`cv_mae` averages performance across 3 different training/validation splits, giving a more stable estimate of generalisation.
If `cv_mae` is much higher than `test_mae`, the model may be overfitting to that particular test split.

**Q: If we run this script again with new data, will previous runs be overwritten?**

A: No. MLflow always appends new runs to the experiment. The full history is preserved.
This is the core value of experiment tracking — you can always go back and see what was tried and when.

**Q: What is the difference between a logged model and a registered model?**

A: A **logged model** is an artifact inside a run — it exists as long as that run exists.
A **registered model** has a name, a version number, and a lifecycle stage (None → Staging → Production → Archived).
Only registered models can be served or promoted through MLflow's deployment workflows.
In Exercise 7 we will load `quickfoods-delivery-predictor` directly from the registry to serve it via FastAPI.

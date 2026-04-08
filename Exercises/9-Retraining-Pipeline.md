# Exercise 09 — Retraining Pipeline

## Scenario: You are an MLOps Engineer at QuickFoods

QuickFoods has expanded to new delivery zones. The monitoring system (Exercise 7) has been raising drift alerts for two weeks:

* Average `distance_km` has shifted from 4.4 to 9.2
* Average `items_count` has shifted from 3.0 to 5.1
* Predicted delivery times are consistently higher than actual times reported by drivers

The Product Manager asks:

> "The model was trained on old data. We have new delivery records now. Can you retrain the model using the new data and prove it performs better before we put it in production?"

Your task:

1. Create a **new dataset** that includes recent delivery records
2. Retrain the model on the combined data
3. **Compare the old model vs the new model** on the new data
4. Register the new model only if it beats the current champion
5. Promote it using the alias workflow from Exercise 8

---

## Learning Objectives

By the end of this exercise you will be able to:

* Build a retraining script that loads new data and trains a fresh model
* Compare old vs new model performance on the same holdout set
* Make a data-driven promotion decision (only promote if metrics improve)
* Register and promote the retrained model in the MLflow registry
* Explain when and why retraining is necessary

---

## Prerequisites

* Exercises 1–8 completed
* A registered champion model in the MLflow registry
* MLflow tracking running locally

---

## Step 1 — Create the New Dataset

In production, new data comes from delivery completion logs. Here we simulate it.

Create `data/delivery_times_new.csv`:

```
distance_km,items_count,is_peak_hour,traffic_level,delivery_time_min
12.0,4,1,3,85
15.5,6,1,3,98
8.0,3,0,2,55
3.0,2,0,1,22
18.0,8,1,3,110
6.5,5,1,2,58
20.0,7,0,3,105
1.5,1,0,1,14
9.0,4,1,2,65
14.0,6,1,3,92
7.2,3,0,2,50
11.5,5,1,3,82
2.5,2,0,1,20
16.0,7,1,3,100
4.0,3,0,2,35
22.0,10,1,3,125
5.5,2,1,2,45
13.0,5,0,3,80
0.8,1,0,1,12
17.5,8,1,3,108
```

This dataset has larger distances and more items — reflecting the new delivery zones.

---

## Step 2 — Create the Retraining Script

Create `src/retrain.py`:

```python
import os
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# ── Config ─────────────────────────────────────────────────────────────────────

ORIGINAL_DATA   = "data/delivery_times.csv"
NEW_DATA        = "data/delivery_times_new.csv"
MODEL_DIR       = "models"
EXPERIMENT_NAME = "quickfoods-delivery-time"
REGISTERED_NAME = "quickfoods-delivery-predictor"
CHAMPION_ALIAS  = "champion"

FEATURES = ["distance_km", "items_count", "is_peak_hour", "traffic_level"]
TARGET   = "delivery_time_min"


def load_champion_model():
    """Load the current champion from the registry."""
    model_uri = f"models:/{REGISTERED_NAME}@{CHAMPION_ALIAS}"
    print(f"Loading current champion from: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def main():
    print("=== QuickFoods: Retraining Pipeline ===\n")

    # ── Load and combine data ─────────────────────────────────────────────
    df_old = pd.read_csv(ORIGINAL_DATA)
    df_new = pd.read_csv(NEW_DATA)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)

    print(f"Original data : {len(df_old)} rows")
    print(f"New data      : {len(df_new)} rows")
    print(f"Combined      : {len(df_combined)} rows")

    X = df_combined[FEATURES]
    y = df_combined[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Evaluate current champion on new test set ─────────────────────────
    champion_model = load_champion_model()
    champion_preds = champion_model.predict(X_test)
    champion_mae   = mean_absolute_error(y_test, champion_preds)
    champion_rmse  = float(np.sqrt(mean_squared_error(y_test, champion_preds)))
    champion_r2    = r2_score(y_test, champion_preds)

    print(f"\nCurrent Champion on new test set:")
    print(f"  MAE  : {champion_mae:.3f}")
    print(f"  RMSE : {champion_rmse:.3f}")
    print(f"  R²   : {champion_r2:.3f}")

    # ── Train new model on combined data ──────────────────────────────────
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Retrain-CombinedData"):
        mlflow.set_tag("project", "QuickFoods")
        mlflow.set_tag("purpose", "retraining")
        mlflow.set_tag("data", "original+new")

        new_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        )
        new_model.fit(X_train, y_train)

        new_preds = new_model.predict(X_test)
        new_mae   = mean_absolute_error(y_test, new_preds)
        new_rmse  = float(np.sqrt(mean_squared_error(y_test, new_preds)))
        new_r2    = r2_score(y_test, new_preds)

        mlflow.log_param("model_name", "GradientBoosting")
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("data_rows", len(df_combined))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        mlflow.log_metric("mae", new_mae)
        mlflow.log_metric("rmse", new_rmse)
        mlflow.log_metric("r2", new_r2)

        # Also log champion metrics for comparison
        mlflow.log_metric("champion_mae", champion_mae)
        mlflow.log_metric("champion_rmse", champion_rmse)
        mlflow.log_metric("champion_r2", champion_r2)
        mlflow.log_metric("mae_improvement", champion_mae - new_mae)

        mlflow.sklearn.log_model(new_model, artifact_path="sklearn-model")

        run_id = mlflow.active_run().info.run_id

        print(f"\nRetrained Model on new test set:")
        print(f"  MAE  : {new_mae:.3f}")
        print(f"  RMSE : {new_rmse:.3f}")
        print(f"  R²   : {new_r2:.3f}")

        # ── Compare and decide ────────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"  Champion MAE : {champion_mae:.3f}")
        print(f"  Retrained MAE: {new_mae:.3f}")
        print(f"  Improvement  : {champion_mae - new_mae:.3f} minutes")
        print(f"{'='*50}")

        if new_mae < champion_mae:
            print("\n✅ Retrained model is BETTER. Registering and promoting...")

            model_uri = f"runs:/{run_id}/sklearn-model"
            mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_NAME)

            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.set_registered_model_alias(REGISTERED_NAME, CHAMPION_ALIAS, mv.version)

            print(f"   Registered version {mv.version}")
            print(f"   Promoted to '{CHAMPION_ALIAS}' alias")
            print(f"\n   Restart the FastAPI service to serve the new model.")
        else:
            print("\n❌ Retrained model is NOT better. Keeping current champion.")
            print("   Consider: more data, different features, or different algorithm.")


if __name__ == "__main__":
    main()
```

---

## Step 3 — Run the Retraining Pipeline

```
python src/retrain.py
```

Expected output:

```
=== QuickFoods: Retraining Pipeline ===

Original data : 20 rows
New data      : 20 rows
Combined      : 40 rows
Loading current champion from: models:/quickfoods-delivery-predictor@champion

Current Champion on new test set:
  MAE  : 12.XXX
  RMSE : 15.XXX
  R²   : 0.7XX

Retrained Model on new test set:
  MAE  : 4.XXX
  RMSE : 5.XXX
  R²   : 0.9XX

==================================================
  Champion MAE : 12.XXX
  Retrained MAE: 4.XXX
  Improvement  : 8.XXX minutes
==================================================

✅ Retrained model is BETTER. Registering and promoting...
   Registered version 3
   Promoted to 'champion' alias

   Restart the FastAPI service to serve the new model.
```

The champion model was trained on 20 rows of old data and extrapolates poorly to new delivery zones. The retrained model on 40 rows handles both distributions well.

---

## Step 4 — Verify in MLflow UI

```
mlflow ui
```

Open `http://127.0.0.1:5000`

Check:

1. **Experiments tab** — find the `Retrain-CombinedData` run. It shows both `mae` and `champion_mae` side by side.
2. **Models tab** — `quickfoods-delivery-predictor` now has version 3 with the `champion` alias.

---

## Step 5 — Restart the API and Test

```
uvicorn src.api:app --reload
```

```
curl http://127.0.0.1:8000/health
```

The `/health` response confirms the new champion is loaded. Test a prediction with a far-distance order:

```
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"distance_km": 18.0, "items_count": 7, "is_peak_hour": 1, "traffic_level": 3}'
```

The retrained model should give a more realistic prediction for this far-distance, large-order scenario.

---

## What We Learned

* Retraining is triggered by monitoring alerts (Exercise 7), not by a fixed schedule
* Always compare the new model against the current champion on the **same test set**
* Only promote if metrics improve — this prevents accidental regressions
* The retraining script is a pipeline: load data → train → evaluate → compare → decide → register → promote
* Combined with versioning (Exercise 8), every model ever deployed is traceable

---

## Key Questions and Answers

**Q: Should we retrain on new data only or combined old + new data?**

A: Usually combined. Training on new data only risks losing knowledge of the original patterns. If the old patterns are truly obsolete (a complete business pivot), then new-only may be appropriate. For QuickFoods, old delivery zones still exist — so combined is correct.

**Q: What if the retrained model is better on new data but worse on old data?**

A: This is a real tradeoff. One approach is to evaluate on both subsets separately and ensure neither degrades beyond a threshold. Another approach is to weight recent data more heavily during training.

**Q: How often should retraining happen?**

A: When monitoring says it is needed — not on a fixed calendar. If drift alerts fire weekly, retrain weekly. If the model is stable for six months, there is no reason to retrain. Unnecessary retraining wastes compute and introduces risk.

**Q: Is this script production-ready?**

A: It demonstrates the right workflow. In production you would add: automated data validation, holdout sets stratified by time, A/B testing before full promotion, and integration with a scheduler (Airflow, Prefect, cron).

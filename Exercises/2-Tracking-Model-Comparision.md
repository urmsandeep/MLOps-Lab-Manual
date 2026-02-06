# MLOps Lab 2  Tracking and Model Comparison

## Scenario: You are an MLOps Engineer at “QuickFoods”
The QuickFoods data science team starts experimenting:
- Different train/test splits
- Different model parameters
- Multiple runs per day

CTO asks:
“Which model is currently the best and why?”

Your task is : introduce experiment tracking and model comparison

## Learning Objectives
- Track ML experiments systematically
- Log parameters, metrics, and models
- Compare runs visually
- Learn about MLflow

## Use Python Virtual Environment 
python -m venv myenv
source venv/bin/activate

## Install Dependencies
Create requirements.txt
```
pandas
scikit-learn
joblib
mlfow
```
Install
```
pip install -r requirements.txt
```

## Update Training Script with Tracking
Refer to the code from Exercise-1: https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/1-QuickFoods-Model-Estimate.md#6-create-training-script

```
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/delivery_times.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "delivery_time_model.pkl")

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)

def train_model(df: pd.DataFrame) -> dict:
    X = df[["distance_km", "items_count", "is_peak_hour", "traffic_level"]]
    y = df["delivery_time_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    return {
        "model": model,
        "mae": mae,
        "mse": mse,
        "test_size": len(X_test),
    }

def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def main():
    print("=== QuickFoods MLOps Lab 2: Experiment Tracking with MLflow ===")

    mlflow.set_experiment("quickfoods-delivery-time")

    with mlflow.start_run():
        df = load_data(DATA_PATH)
        result = train_model(df)

        # Params
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Metrics
        mlflow.log_metric("mae", result["mae"])
        mlflow.log_metric("mse", result["mse"])

        # Save local model artifact (file)
        save_model(result["model"])
        mlflow.log_artifact(MODEL_PATH)

        # Save MLflow model artifact (structured)
        mlflow.sklearn.log_model(
            result["model"],
            artifact_path="model"
        )

        print(f"Test samples: {result['test_size']}")
        print(f"MAE (minutes): {result['mae']:.2f}")
        print(f"MSE: {result['mse']:.2f}")
        print(f"Local model saved to: {MODEL_PATH}")
        print("Logged run to MLflow (params, metrics, model).")

if __name__ == "__main__":
    main()
```

## Run Training Again
Run it 2–3 times (optionally change test size).
```
python3 src/train.py
```

## Launch MLflow UI
```
mlflow ui
```
Open browser: http://localhost:5000

Observe:
- Multiple experiment runs
- Parameters and metrics logged
- Models stored as artifacts

## What we Learned
- How professional teams track experiments
- Why reproducibility alone is not enough
- How to answer: “Which model is in production?”

# Question and Answers
Why is experiment tracking critical in real ML teams?
What happens if metrics are not logged?
Why should models be treated as versioned artifacts?




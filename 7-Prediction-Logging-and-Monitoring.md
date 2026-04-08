# Exercise 07 — Prediction Logging and Monitoring

## Scenario: You are an MLOps Engineer at QuickFoods

QuickFoods now has a deployed model served via FastAPI (Exercise 6).
Hundreds of prediction requests arrive daily from the customer app, delivery dashboard, and operations portal.

The Product Manager asks:

> "The model has been live for two weeks. How is it performing? Are predictions still accurate? Has anything changed in the incoming data?"

You realise that: **the API returns predictions but does not record them anywhere.**

If the model silently degrades — due to a new delivery zone, a seasonal shift, or a change in order patterns — nobody will know until customers complain.

Your task:

1. Add **prediction logging** to the FastAPI service — record every request and response
2. Build a **monitoring script** that reads the log and computes live metrics
3. Implement **basic input drift detection** by comparing live feature distributions to training data
4. Set **alert thresholds** so the team knows when to investigate

 

## Learning Objectives

By the end of this exercise we will be able to:

* Log every prediction (input features + output + timestamp) to a structured file
* Compute rolling accuracy metrics from logged predictions
* Detect input distribution drift using simple statistical checks
* Define alert thresholds based on business tolerance
* Explain why monitoring is essential in production ML

 

## Prerequisites

* Exercises 1–6 completed
* FastAPI service from Exercise 6 running
* A trained model file at `models/delivery_time_model.pkl`

 

## Continuation from Exercise 6

Use the same project structure. We will modify `src/api.py` and add new files.

Updated project structure:

```
quickfoods-mlops-lab/
├── data/
│   ├── delivery_times.csv
│   └── delivery_times_new.csv        ← new (simulated production data)
├── logs/
│   └── predictions.jsonl             ← created automatically
├── models/
│   └── delivery_time_model.pkl
├── src/
│   ├── api.py                        ← modified
│   ├── monitor.py                    ← new
│   └── simulate_traffic.py           ← new
├── requirements.txt
└── README.md
```


## Step 1 — Add Prediction Logging to the API

Update `requirements.txt`:

```
pandas
scikit-learn
joblib
fastapi
uvicorn
numpy
```

Install:

```
pip install -r requirements.txt
```

Modify `src/api.py` — add logging after the prediction is computed.

Replace the existing `/predict` endpoint with:

```python
import os
import json
import joblib
import pandas as pd
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = "models/delivery_time_model.pkl"
LOG_DIR    = "logs"
LOG_PATH   = os.path.join(LOG_DIR, "predictions.jsonl")

app = FastAPI(
    title="QuickFoods Delivery Time Prediction API",
    description="API with prediction logging for monitoring",
    version="2.0.0"
)

# Load model once at startup
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
os.makedirs(LOG_DIR, exist_ok=True)


class DeliveryRequest(BaseModel):
    distance_km: float = Field(..., gt=0)
    items_count: int = Field(..., gt=0)
    is_peak_hour: int = Field(..., ge=0, le=1)
    traffic_level: int = Field(..., ge=1, le=3)


class PredictionResponse(BaseModel):
    delivery_time_min: float


def log_prediction(request_data: dict, prediction: float):
    """Append one JSON line per prediction to the log file."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": request_data,
        "prediction": prediction,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health_check():
    log_count = 0
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            log_count = sum(1 for _ in f)
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "predictions_logged": log_count,
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

        # Log every prediction
        log_prediction(input_dict, prediction)

        return PredictionResponse(delivery_time_min=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
```

 

## Step 2 — Simulate Production Traffic

Real production traffic arrives over time. We simulate this with a script that sends requests to the running API.

Create `src/simulate_traffic.py`:

```python
import requests
import random
import time

API_URL = "http://127.0.0.1:8000/predict"

def generate_normal_request():
    """Requests similar to training data distribution."""
    return {
        "distance_km": round(random.uniform(0.5, 10.0), 1),
        "items_count": random.randint(1, 6),
        "is_peak_hour": random.choice([0, 1]),
        "traffic_level": random.choice([1, 2, 3]),
    }

def generate_drifted_request():
    """Requests outside training distribution — simulates drift."""
    return {
        "distance_km": round(random.uniform(15.0, 30.0), 1),   # much farther
        "items_count": random.randint(8, 15),                    # much larger orders
        "is_peak_hour": 1,                                       # always peak
        "traffic_level": 3,                                      # always high traffic
    }

def main():
    print("=== Simulating QuickFoods API Traffic ===")
    print("Sending 30 normal requests, then 20 drifted requests...\n")

    # Phase 1: Normal traffic
    for i in range(30):
        payload = generate_normal_request()
        resp = requests.post(API_URL, json=payload)
        data = resp.json()
        print(f"[Normal  {i+1:02d}] dist={payload['distance_km']:5.1f} items={payload['items_count']} → {data['delivery_time_min']:.1f} min")
        time.sleep(0.1)

    print("\n--- Drift begins ---\n")

    # Phase 2: Drifted traffic
    for i in range(20):
        payload = generate_drifted_request()
        resp = requests.post(API_URL, json=payload)
        data = resp.json()
        print(f"[Drifted {i+1:02d}] dist={payload['distance_km']:5.1f} items={payload['items_count']} → {data['delivery_time_min']:.1f} min")
        time.sleep(0.1)

    print(f"\nDone. 50 predictions logged to logs/predictions.jsonl")


if __name__ == "__main__":
    main()
```

Run the API first (in one terminal):

```
uvicorn src.api:app --reload
```

Then run the simulator (in another terminal):

```
python src/simulate_traffic.py
```

 

## Step 3 — Build the Monitoring Script

Create `src/monitor.py`:

```python
import json
import os
import pandas as pd
import numpy as np

LOG_PATH      = "logs/predictions.jsonl"
TRAINING_DATA = "data/delivery_times.csv"

# Alert thresholds (business-defined)
ALERT_DISTANCE_MEAN_SHIFT = 3.0    # if mean distance shifts by more than 3 km
ALERT_ITEMS_MEAN_SHIFT    = 2.0    # if mean items shifts by more than 2
ALERT_PREDICTION_MEAN     = 60.0   # if avg prediction exceeds 60 min, something is off


def load_prediction_log(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No prediction log at: {path}")

    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    rows = []
    for r in records:
        row = {
            "timestamp": r["timestamp"],
            "prediction": r["prediction"],
            **r["input"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_prediction_stats(df: pd.DataFrame) -> dict:
    return {
        "total_predictions": len(df),
        "mean_prediction": round(df["prediction"].mean(), 2),
        "max_prediction": round(df["prediction"].max(), 2),
        "min_prediction": round(df["prediction"].min(), 2),
        "std_prediction": round(df["prediction"].std(), 2),
    }


def check_input_drift(live_df: pd.DataFrame, train_df: pd.DataFrame) -> list:
    alerts = []

    # Compare mean distance
    train_mean_dist = train_df["distance_km"].mean()
    live_mean_dist  = live_df["distance_km"].mean()
    dist_shift      = abs(live_mean_dist - train_mean_dist)

    if dist_shift > ALERT_DISTANCE_MEAN_SHIFT:
        alerts.append(
            f"DRIFT: distance_km mean shifted by {dist_shift:.1f} km "
            f"(train={train_mean_dist:.1f}, live={live_mean_dist:.1f})"
        )

    # Compare mean items
    train_mean_items = train_df["items_count"].mean()
    live_mean_items  = live_df["items_count"].mean()
    items_shift      = abs(live_mean_items - train_mean_items)

    if items_shift > ALERT_ITEMS_MEAN_SHIFT:
        alerts.append(
            f"DRIFT: items_count mean shifted by {items_shift:.1f} "
            f"(train={train_mean_items:.1f}, live={live_mean_items:.1f})"
        )

    # Check prediction range
    mean_pred = live_df["prediction"].mean()
    if mean_pred > ALERT_PREDICTION_MEAN:
        alerts.append(
            f"WARNING: avg prediction is {mean_pred:.1f} min — exceeds threshold of {ALERT_PREDICTION_MEAN} min"
        )

    return alerts


def main():
    print("=== QuickFoods Model Monitoring Report ===\n")

    live_df  = load_prediction_log(LOG_PATH)
    train_df = pd.read_csv(TRAINING_DATA)

    # Prediction statistics
    stats = compute_prediction_stats(live_df)
    print("Prediction Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Feature distribution summary
    print("\nLive Feature Means:")
    for col in ["distance_km", "items_count", "is_peak_hour", "traffic_level"]:
        print(f"  {col}: {live_df[col].mean():.2f}  (train: {train_df[col].mean():.2f})")

    # Drift alerts
    print("\nDrift Check:")
    alerts = check_input_drift(live_df, train_df)

    if alerts:
        for a in alerts:
            print(f"  ⚠️  {a}")
        print("\n→ Action: Investigate data source. Consider retraining (Exercise 09).")
    else:
        print("  ✅ No drift detected. Model inputs look consistent with training data.")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

 

## Step 4 — Run the Monitor

```
python src/monitor.py
```

Expected output (after running the simulator):

```
=== QuickFoods Model Monitoring Report ===

Prediction Statistics:
  total_predictions: 50
  mean_prediction: 58.42
  max_prediction: 112.30
  min_prediction: 15.20
  std_prediction: 28.71

Live Feature Means:
  distance_km: 11.23  (train: 4.41)
  items_count: 5.84  (train: 2.95)
  is_peak_hour: 0.72  (train: 0.50)
  traffic_level: 2.48  (train: 2.20)

Drift Check:
  ⚠️  DRIFT: distance_km mean shifted by 6.8 km (train=4.4, live=11.2)
  ⚠️  DRIFT: items_count mean shifted by 2.9 (train=3.0, live=5.8)

→ Action: Investigate data source. Consider retraining (Exercise 09).

Done.
```

The drifted requests from the simulator pushed `distance_km` and `items_count` well beyond training distribution — the monitor catches this.

 

## What We Learned

* A deployed model without logging is a black box — we cannot measure production performance
* JSONL (one JSON object per line) is a simple, appendable log format
* Comparing live feature means to training feature means is the simplest form of drift detection
* Alert thresholds should come from business requirements (e.g. "predictions above 60 minutes need review")
* Monitoring is what connects deployment to retraining — it tells we **when** the model needs updating

 

## Key Questions and Answers

**Q: Why log predictions to a file instead of a database?**

A: For this lab, a file keeps things simple. In production, we would log to a database, a message queue (Kafka), or a monitoring platform (Evidently, Prometheus). The principle is the same — every prediction must be recorded with its inputs and timestamp.

**Q: Why compare means instead of using a statistical test?**

A: Mean comparison is the simplest approach and is often enough to catch obvious distribution shifts. In production we might use the Kolmogorov-Smirnov test, Population Stability Index (PSI), or tools like Evidently AI. We start simple to build the intuition first.

**Q: What if we do not have ground truth labels in production?**

A: This is common. we can still monitor input distributions and prediction distributions. When ground truth (actual delivery time) arrives later — for example from delivery completion logs — we can compute actual MAE and compare it to training MAE. This is called **delayed evaluation**.

**Q: When should drift trigger retraining?**

A: When monitored metrics cross wer alert thresholds consistently (not just one spike). In Exercise 09 we will build a retraining pipeline that uses new data to update the model and compares performance before and after.

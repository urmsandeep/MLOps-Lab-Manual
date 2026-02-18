# Exercise 4 — Docker for Model Packaging 

Containerize the ML model app and run it locally with test 

## Scenario: You are an MLOps Engineer at QuickFoods

QuickFoods has a delivery-time prediction model. 
The data science team can train models locally, but Product wants a **portable package** that runs the same way on any deployment (cloud or on-prem).

Your task:
1. Create a **small “model app”** (CLI-based) that loads a trained model and predicts delivery time
2. Package it using **Docker**
3. Run it locally and verify **test input/output**
4. Tag the Docker image with a version

## Learning outcomes
By the end of this exercise you will be able to:
- Write a minimal Dockerfile for an ML project
- Build and run Docker containers with tagged images
- Pass input to a container and get prediction output
- Mount volumes (optional) to persist artifacts

## Prerequisites
- Docker Desktop installed (Windows/macOS) OR Docker Engine (Linux)
- Project files from Exercises 1–3 (dataset + training script)
- A trained model file available locally

## Expected project structure

Use  existing repo structure from previous exercises:
https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/1-QuickFoods-Model-Artifact.md#2-create-the-required-project-structure

**Exercise 3** script saves models with different names (e.g. RandomForest.pkl), pick one model and rename/copy it to:
models/delivery_time_model.pkl
For Example:
cp models/RandomForest.pkl models/delivery_time_model.pkl

## Step-1: Create the “model app” 
Create src/predict_cli.py

```
import argparse
import json
import os
import joblib
import pandas as pd

MODEL_PATH = os.environ.get("MODEL_PATH", "models/delivery_time_model.pkl")

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at: {path}. Train a model first and save it to {path}."
        )
    return joblib.load(path)

def predict_one(model, distance_km: float, items_count: int, is_peak_hour: int, traffic_level: int) -> float:
    X = pd.DataFrame([{
        "distance_km": distance_km,
        "items_count": items_count,
        "is_peak_hour": is_peak_hour,
        "traffic_level": traffic_level
    }])
    pred = model.predict(X)[0]
    return float(pred)

def main():
    parser = argparse.ArgumentParser(description="FooFoods Delivery Time Predictor (CLI)")
    parser.add_argument("--distance_km", type=float, required=True)
    parser.add_argument("--items_count", type=int, required=True)
    parser.add_argument("--is_peak_hour", type=int, choices=[0, 1], required=True)
    parser.add_argument("--traffic_level", type=int, choices=[1, 2, 3], required=True)

    args = parser.parse_args()

    model = load_model(MODEL_PATH)
    y = predict_one(model, args.distance_km, args.items_count, args.is_peak_hour, args.traffic_level)

    out = {
        "input": {
            "distance_km": args.distance_km,
            "items_count": args.items_count,
            "is_peak_hour": args.is_peak_hour,
            "traffic_level": args.traffic_level
        },
        "prediction": {
            "delivery_time_min": round(y, 2)
        }
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

Quick local test on the system:
```
python src/predict_cli.py --distance_km 4.2 --items_count 3 --is_peak_hour 1 --traffic_level 2
```

Ouput
```
{
  "input": { "...": "..." },
  "prediction": { "delivery_time_min": 45.12 }
}
```

## Step 2 — Create a Dockerfile

Create file name: Dockerfile

```
# Small base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory inside container
WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model artifact
COPY src/ /app/src/
COPY models/ /app/models/

# Default command: run the CLI predictor
ENTRYPOINT ["python", "src/predict_cli.py"]
```

## Step 3 — Build the Docker image and tag it
```
docker build -t quickfoods-delivery-model:0.1 .
docker images | grep quickfoods
```

## Step 4 — Run container with test input/output

```
docker run --rm quickfoods-delivery-model:0.1 \
  --distance_km 4.2 --items_count 3 --is_peak_hour 1 --traffic_level 2
```
Expected: JSON output printed from inside the container.

Run one more test case:
```
docker run --rm foofoods-delivery-model:0.1 \
  --distance_km 1.0 --items_count 1 --is_peak_hour 0 --traffic_level 1
```

Step 5 — Demonstrate Container lifecycle 
```
docker ps                 # running containers
docker ps -a              # all containers (including stopped)
docker images             # list images
docker rmi foofoods-delivery-model:0.1   # remove image
```


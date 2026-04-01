# Exercise 06 — FastAPI Model Serving

## Objective

In the previous exercises, 
1. we trained a machine learning model
2. compared different runs
3. selected the best model
4. packaged the application using Docker.

Now the next step is to make the model usable by other systems.

In this exercise, we will deploy the trained model as a **REST API** using **FastAPI**.

we will create:

-  `/health` endpoint to check whether the service is running
-  `/predict` endpoint to send input data and receive model predictions

This is one of the most important steps in MLOps because a trained model is useful only when it can be consumed by applications or other teams.


# Real-World Scenario

You are working as an **MLOps Engineer at QuickFoods**.

Your team has built a machine learning model to predict **food delivery time** based on order and route information.

The business team now wants to integrate this model into:

- the customer app
- the delivery partner dashboard
- the support operations portal

They cannot use a Python script directly.

So you task is to expose the model as an API service that any application can call.

# Where This Fits in the MLOps Learning Map

This exercise maps to:

**Deployment (Serving)**

Learning flow:

```
Raw Data → Data Pipeline → Training → Evaluation → Versioning → Deployment → Monitoring → Retraining
```

## Concepts to Understand Before Starting
1. What is model serving?
Model serving means making a trained model available to other systems so they can send input and receive predictions.
Here a frontend (e.g. ChatApp) can all the model via and API
```
POST /predict

{
  "prediction": 42.3
}
```

## Problem Statement

Build a FastAPI service for the QuickFoods delivery-time prediction model.

The API must:
load the trained model from disk
expose a GET /health endpoint
expose a POST /predict endpoint
accept input features in JSON format
return prediction output in JSON format

## Input Features

The model predicts delivery time using these features:
distance_km
items_count
is_peak_hour
traffic_level

Target:
delivery_time_min

## Prerequisites
Python 3.9 or above
the model file created in earlier exercises
a virtual environment
requirements.txt

## Expected model file:
models/delivery_time_model.pkl
If it does not exist, first train and save a model. Refer to previous exercise

## Project Structure
```
MLOps-Lab-Manual/
├── data/
├── logs/
├── models/
│   └── delivery_time_model.pkl
├── src/
│   ├── train.py
│   ├── train_with_mlflow.py
│   ├── monitor.py
│   └── api.py
├── requirements.txt
└── README.md
```

## Step 1 — Install Required Packages

Update requirements.txt:
```
pandas
scikit-learn
joblib
fastapi
uvicorn
```

Install dependencies:
```
pip install -r requirements.txt
```

## Step 2 — Create the FastAPI Application
Create a new file:
```
src/api.py
```

Paste this code:
```
import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = "models/delivery_time_model.pkl"

app = FastAPI(
    title="QuickFoods Delivery Time Prediction API",
    description="API for predicting food delivery time using a trained ML model",
    version="1.0.0"
)


# -----------------------------
# Load the model at startup
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found at: {MODEL_PATH}. "
        "Please train the model and save it before starting the API."
    )

model = joblib.load(MODEL_PATH)


# -----------------------------
# Request schema
# -----------------------------
class DeliveryRequest(BaseModel):
    distance_km: float = Field(..., gt=0, description="Distance in kilometers")
    items_count: int = Field(..., gt=0, description="Number of items in the order")
    is_peak_hour: int = Field(..., ge=0, le=1, description="0 = no, 1 = yes")
    traffic_level: int = Field(..., ge=1, le=3, description="1 = low, 2 = medium, 3 = high")


# -----------------------------
# Response schema
# -----------------------------
class PredictionResponse(BaseModel):
    delivery_time_min: float


# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH
    }


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: DeliveryRequest):
    try:
        input_df = pd.DataFrame([{
            "distance_km": request.distance_km,
            "items_count": request.items_count,
            "is_peak_hour": request.is_peak_hour,
            "traffic_level": request.traffic_level
        }])

        prediction = model.predict(input_df)[0]

        return PredictionResponse(
            delivery_time_min=round(float(prediction), 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

``` 

## Step 3 — Understanding the Code

**A. FastAPI(...)**
This creates the API application.
app = FastAPI(...)

**B. Loading model at startup**
model = joblib.load(MODEL_PATH)

Why do we load the model once?
Because loading the model for every request would be very slow.

Correct approach:
load once at startup
reuse for each prediction request

**C. DeliveryRequest**
This defines the input schema.
Example valid request:

```
{
  "distance_km": 4.5,
  "items_count": 3,
  "is_peak_hour": 1,
  "traffic_level": 2
}
```

Pydantic checks:

```
distance must be greater than 0
items must be greater than 0
peak hour must be 0 or 1
traffic level must be 1, 2, or 3
```

**D. /health**
This endpoint tells us whether the service is alive.
Example response:
```
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/delivery_time_model.pkl"
}
```

**E. /predict**

This endpoint accepts input and returns the predicted delivery time.

## Step 4 — Run the API

From the project root, run:

```
uvicorn src.api:app --reload
```

Expected output:

```
Uvicorn running on http://127.0.0.1:8000
```

## Step 5 — Test the API in Browser

Open this in browser:
http://127.0.0.1:8000/health

We should see JSON output.

## Step 6 — Use Swagger UI

FastAPI automatically creates interactive API documentation.

Open:
http://127.0.0.1:8000/docs

This page allows us to:

test /health
test /predict
send JSON input directly from browser

## Step 7 — Test /predict
Sample Request 1

Use this JSON:

```
{
  "distance_km": 4.2,
  "items_count": 3,
  "is_peak_hour": 1,
  "traffic_level": 2
}
```
Expected response format:
```
{
  "delivery_time_min": 45.23
}
```
The exact number may differ depending on the trained model.
```
Sample Request 2
{
  "distance_km": 1.2,
  "items_count": 1,
  "is_peak_hour": 0,
  "traffic_level": 1
}
```
Possible output:
```
{
  "delivery_time_min": 17.84
}
```

## Step 8 — Test Using curl

We can also test from terminal using command line URL (curl)

Health check
```
curl http://127.0.0.1:8000/health
```
Prediction
```
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "distance_km": 4.2,
  "items_count": 3,
  "is_peak_hour": 1,
  "traffic_level": 2
}'
```

## Step 9 — Try Invalid Input

We can also observe validation behavior.

Example invalid input:
```
{
  "distance_km": -2,
  "items_count": 0,
  "is_peak_hour": 5,
  "traffic_level": 10
}
```
FastAPI should reject this automatically and return a validation error.

This is important because APIs must be robust against bad input.

## Why This Exercise Matters
This is the how a model becomes usable by real systems.

Before this exercise:
the model was just a file on disk

After this exercise:
the model becomes a service that applications can call

This is the bridge between:
machine learning
software engineering
production deployment

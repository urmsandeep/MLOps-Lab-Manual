# MLOps Lab 1  Reproducible ML Project Setup 

## Scenario: You are an MLOps Engineer at “QuickFoods”

**QuickFoods** is a food-delivery company. The business team wants a simple ML model to estimate **delivery time (in minutes)** based on order and distance information.

You have been hired as an **MLOps**. Your first task is not to build a perfect model. Your task is to build a **reproducible ML project** that:

- Runs on developer laptop or desktop
- Has a clean project structure
- Uses a virtual environment
- Can train a baseline ML model from a script
- Saves the trained model artifact in a standard location
- Is tracked with Git (clean commits, proper ignore files)

---

## Learning Outcome

A  MLOps project that trains a baseline regression model:

**Inputs (features):**
- distance_km
- items_count
- is_peak_hour (0 or 1)
- traffic_level (1 to 3)

**Output (target):**
- delivery_time_min

---

## Tools Required (Install before lab)

### Mandatory
- **Python 3.9+**
- **Git**
- A code editor (VS Code recommended)

## 1. Create Project folder
```
mkdir quickfoods-mlops-lab1
cd quickfoods-mlops-lab1
```
## 2. Create the required project structure
```
quickfoods-mlops-lab1/
├── data/
│   └── delivery_times.csv
├── models/
├── src/
│   ├── __init__.py
│   └── train.py
├── .gitignore
├── requirements.txt
└── README.md

mkdir -p data models src
touch src/__init__.py
```

## 3. Add dataset 
Create this file: data/delivery_times.csv

and copy/paste:
```
distance_km,items_count,is_peak_hour,traffic_level,delivery_time_min
1.2,1,0,1,18
3.5,2,0,2,28
5.0,3,1,3,52
2.0,4,1,2,35
7.5,2,0,3,55
4.2,5,1,2,48
0.8,1,0,1,15
6.1,3,0,3,50
3.0,2,1,2,34
8.0,6,1,3,70
2.8,1,0,2,26
1.5,2,1,2,25
9.2,4,0,3,68
10.0,6,1,3,78
4.8,3,0,2,38
5.5,2,1,3,56
2.2,3,0,2,31
7.0,5,1,3,72
3.8,4,0,2,40
6.5,1,0,3,47
```

## 4. Create a Python virtual environment

```
python -m venv venv
source venv/bin/activate

```

## 5. Install Requirements
Create requirements.txt:

```
pandas
scikit-learn
joblib
```
Install:
```
pip install -r requirements.txt
```

## 6. Create training script
```
import os
import joblib
import pandas as pd

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
    print("=== QuickFoods MLOps Lab 1: Baseline Training ===")

    df = load_data(DATA_PATH)
    result = train_model(df)

    print(f"Test samples: {result['test_size']}")
    print(f"MAE (minutes): {result['mae']:.2f}")
    print(f"MSE: {result['mse']:.2f}")

    save_model(result["model"])
    print(f"Model saved to: {MODEL_PATH}")
    print("Done. Reproducible ML artifact created.")

if __name__ == "__main__":
    main()

```

Run

```
python src/train.py
```
Output
```
=== QuickFoods MLOps Lab 1: Training Baseline Model ===
Test samples: 4
MAE (minutes): 4.XX
MSE: XX.XX
Model saved to: models/delivery_time_model.pkl
Done. You have produced a reproducible ML artifact.
```

## What we learnt
- Built a clean ML repository
- Created a repeatable training pipeline
- Produced a versionable ML artifact

## Key Questions and Answers
Q: What is a Model Artifact?
A: It's output of a training process can be loaded and used for interference without retraining

Q: Why is the training done using a Python script instead of a notebook?
Use of Scripts or code make it: Automatable and CI/CD friendly
Notebooks are good for exploration, but scripts are better for production pipeline




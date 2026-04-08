# MLOps Journey — End-to-End Learning Map

This lab follows a real-world MLOps lifecycle. 

Each exercise builds on the previous one to demonstrate how ML systems are developed and operated in production.

## Learning Flow and Exercise Mapping

| Exercise | What You Learn | MLOps Stage | GitHub Link |
|----------|---------------|------------|-------------|
| Exercise 01 | Project setup, dataset, baseline model, model artifact creation | Raw Data, Training | https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/1-QuickFoods-Model-Artifact.md |
| Exercise 02 | Experiment tracking basics, comparing runs using MLflow | Evaluation (Intro) | https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/2-Tracking-Model-Comparision.md |
| Exercise 03 | Multi-metric evaluation (MAE, RMSE, R²), model selection using MLflow | Evaluation, Versioning | https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/3-Multi-Metric-MLflow-Model-Selection.md |
| Exercise 04 | Packaging ML model using Docker, ensuring reproducibility | Deployment (Packaging) | https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/4-Docker-for-Model-Packaging.md |
| Exercise 05 | Model optimization using hyperparameter tuning | Training (Advanced) | https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/5-Hyperparameter-Tuning.md |
| Exercise 06 | Deploying model using FastAPI (`/predict`, `/health`) | Deployment (Serving) | https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/6-FastAPI-Model-Serving.md |
| Exercise 07 | Logging predictions, monitoring inputs/outputs, basic drift detection | Monitoring | https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/7-Prediction-Logging-and-Monitoring.md |
| Exercise 08 | Model versioning and promotion (MLflow Model Registry) | Versioning (Advanced) | https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/8-Model-Versioning-and-Promotion.md |
| Exercise 09 | Retraining pipeline using new data and performance comparison | Retraining | https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/9-Retraining-Pipeline.md |
| Exercise 10 | End-to-end ML pipeline (train → track → deploy → monitor → retrain) | Full Lifecycle | https://github.com/urmsandeep/MLOps-Lab-Manual/blob/main/Exercises/Final-10-End-to-End-ML-Pipeline.md |

## What You Will Learn in this course

By completing this lab, you will learn to:

- Build a complete ML system from scratch
- Track and compare experiments using MLflow
- Select the best model using multiple evaluation metrics
- Package models using Docker
- Deploy models as APIs using FastAPI
- Monitor predictions and detect issues in production
- Retrain models based on real-world data
- Explain an end-to-end ML system in interviews

## Important Thought Process

MLOps is not:

> “Train a model once and finish”

MLOps is:

> “Continuously improve models using data, feedback, and monitoring”


## End Goal - Build Skills

You should be able to answer:

- Which model is currently deployed?
- How accurate is it in production?
- How do you detect model issues?
- When should you retrain the model?
- What changed between model versions?

This is what real-world ML and MLOps engineers do.

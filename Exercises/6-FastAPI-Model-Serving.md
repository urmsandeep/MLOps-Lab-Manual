# Exercise 06 — FastAPI Model Serving

## Objective

In the previous exercises, 
1. we trained a machine learning model
2. compared different runs
3. selected the best model
4. packaged the application using Docker.

Now the next step is to make the model usable by other systems.

In this exercise, we will deploy the trained model as a **REST API** using **FastAPI**.

You will create:

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

So your task is to expose the model as an API service that any application can call.

# Where This Fits in the MLOps Journey

This exercise maps to:

**Deployment (Serving)**

Learning flow:

```text
Raw Data → Data Pipeline → Training → Evaluation → Versioning → Deployment → Monitoring → Retraining

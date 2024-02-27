---
title: snippet example
---
# Introduction

This document will walk you through the implementation of a new feature in our FastAPI application. The feature is a new endpoint that predicts whether a passenger would have survived the Titanic disaster based on certain input parameters.

We will cover:

1. Why we chose FastAPI for this feature.


1. How we load the machine learning model.


1. How we define the new endpoint.


1. How we use the machine learning model to make predictions.

# Choosing FastAPI

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints. It's easy to use and allows for quick development of robust APIs. This is why we chose it for this feature.

# Loading the machine learning model

<SwmSnippet path="/app/main.py" line="4">

---

In the first line of the code snippet, we load a machine learning model that was previously trained and saved to a file named 'output.joblib'. This model will be used to make predictions based on the input parameters provided by the user.

```python
app = FastAPI()

model = joblib.load('output.joblib')

@app.get('/')
def get_root():

	return {'message': 'Welcome to the Titanic prediction API'}

@app.get('/predict')
def predict(Pclass, male, Age, Siblings_Spouses, Parents_Children, Fare):
	y_pred = model.predict([[int(Pclass), int(male), int(Age), int(Siblings_Spouses), int(Parents_Children), int(Fare)]])[0]

	return "survived" if int(y_pred)==1 else "not survived"
```

---

</SwmSnippet>

# Defining the new endpoint

<SwmSnippet path="/app/main.py" line="4">

---

We define a new endpoint '/predict' that takes in six parameters: Pclass, male, Age, Siblings_Spouses, Parents_Children, and Fare. These parameters represent the passenger's class, gender, age, number of siblings/spouses aboard, number of parents/children aboard, and fare respectively. The endpoint is defined using the @app.get decorator, which means it will respond to HTTP GET requests.

```python
app = FastAPI()

model = joblib.load('output.joblib')

@app.get('/')
def get_root():

	return {'message': 'Welcome to the Titanic prediction API'}

@app.get('/predict')
def predict(Pclass, male, Age, Siblings_Spouses, Parents_Children, Fare):
	y_pred = model.predict([[int(Pclass), int(male), int(Age), int(Siblings_Spouses), int(Parents_Children), int(Fare)]])[0]

	return "survived" if int(y_pred)==1 else "not survived"
```

---

</SwmSnippet>

# Making predictions

<SwmSnippet path="/app/main.py" line="4">

---

Inside the predict function, we use the loaded model to make a prediction. The model's predict method takes in a 2D array where each inner array represents a passenger's details. The prediction is then returned as a string: "survived" if the model predicts 1, and "not survived" otherwise.

```python
app = FastAPI()

model = joblib.load('output.joblib')

@app.get('/')
def get_root():

	return {'message': 'Welcome to the Titanic prediction API'}

@app.get('/predict')
def predict(Pclass, male, Age, Siblings_Spouses, Parents_Children, Fare):
	y_pred = model.predict([[int(Pclass), int(male), int(Age), int(Siblings_Spouses), int(Parents_Children), int(Fare)]])[0]

	return "survived" if int(y_pred)==1 else "not survived"
```

---

</SwmSnippet>

This implementation allows us to quickly and easily make predictions using our machine learning model, and it provides a simple and intuitive API for users to interact with.

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBVGl0YW5pY19NYWNoaW5lX0xlYXJuaW5nX2Zyb21fRGlzYXN0ZXIlM0ElM0FBbWlyYW5Hb3phbGlzaHZpbGk=" repo-name="Titanic_Machine_Learning_from_Disaster"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>

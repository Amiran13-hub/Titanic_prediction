---
title: snippet train_titanic_data.py
---
# Introduction

This document will walk you through the implementation of the Titanic survival prediction feature. This feature uses machine learning to predict whether a passenger on the Titanic would have survived based on various factors such as their age, fare, and class.

We will cover:

1. How the data is imported and prepared for the model.


1. The creation and training of the prediction model.


1. The evaluation of the model's performance.


1. The use of K-fold cross-validation to improve the model's accuracy.

# Importing and preparing the data

<SwmSnippet path="/app/train_titanic_data.py" line="1">

---

The first step in our implementation is to import the necessary libraries and load the data from a CSV file. We use pandas to load the data into a DataFrame, which allows us to easily manipulate and analyze the data. We then add a new boolean column to the DataFrame to indicate whether the passenger is male. This is done because the 'Sex' column in the original data is a string, and our model can only handle numerical data. We also extract the features and target from the DataFrame and store them in separate variables.

```python
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# import data frame to variable
df = pd.read_csv('data/titanic.csv')
print(df.head())
print(df.describe())

# plot data with or without diff colors
def plot_data():
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])
    plt.show()

# add new boolean column into data
df['male'] = df['Sex'] == 'male'

#  evaluate data (take values from data columns)
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
# y = target data
y = df['Survived'].values

# add prediction model
model = LogisticRegression()

# train-test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=27)

def datasplit_pred():

    model.fit(X_train, y_train)
    print("'split' model.score:", model.score(X_test, y_test))

    y_pred = model.predict(X_test)
    print("'split' accuracy:", accuracy_score(y_test, y_pred))
    print("'split' precision:", precision_score(y_test, y_pred))
    print("'split' recall:", recall_score(y_test, y_pred))
    print("'split' f1_score:", f1_score(y_test, y_pred))
    return

def Kfold_model():
    X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
    y = df['Survived'].values
    scores = []
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticRegression()
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    print("K_fold scores:", scores)
    print("K_fold mean score:", np.mean(scores))
    return

all = "plot_data()", "datasplit_pred()", "Kfold_model()"
print("Functions", all)

output = datasplit_pred()
# print(output)

dump(model, 'output.joblib')



```

---

</SwmSnippet>

# Creating and training the model

<SwmSnippet path="/app/train_titanic_data.py" line="1">

---

Next, we create a Logistic Regression model using scikit-learn. This is a simple yet powerful model that is often used for binary classification problems like ours. We then split the data into a training set and a test set. The model is trained on the training set, and its performance is evaluated on the test set. This is done to ensure that our model can generalize well to unseen data.

```python
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# import data frame to variable
df = pd.read_csv('data/titanic.csv')
print(df.head())
print(df.describe())

# plot data with or without diff colors
def plot_data():
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])
    plt.show()

# add new boolean column into data
df['male'] = df['Sex'] == 'male'

#  evaluate data (take values from data columns)
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
# y = target data
y = df['Survived'].values

# add prediction model
model = LogisticRegression()

# train-test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=27)

def datasplit_pred():

    model.fit(X_train, y_train)
    print("'split' model.score:", model.score(X_test, y_test))

    y_pred = model.predict(X_test)
    print("'split' accuracy:", accuracy_score(y_test, y_pred))
    print("'split' precision:", precision_score(y_test, y_pred))
    print("'split' recall:", recall_score(y_test, y_pred))
    print("'split' f1_score:", f1_score(y_test, y_pred))
    return

def Kfold_model():
    X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
    y = df['Survived'].values
    scores = []
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticRegression()
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    print("K_fold scores:", scores)
    print("K_fold mean score:", np.mean(scores))
    return

all = "plot_data()", "datasplit_pred()", "Kfold_model()"
print("Functions", all)

output = datasplit_pred()
# print(output)

dump(model, 'output.joblib')



```

---

</SwmSnippet>

# Evaluating the model's performance

<SwmSnippet path="/app/train_titanic_data.py" line="1">

---

After training the model, we evaluate its performance by calculating various metrics such as accuracy, precision, recall, and F1 score. These metrics give us a comprehensive view of the model's performance. Accuracy tells us the proportion of correct predictions, precision tells us the proportion of positive predictions that are actually positive, recall tells us the proportion of actual positives that were predicted positive, and the F1 score is the harmonic mean of precision and recall.

```python
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# import data frame to variable
df = pd.read_csv('data/titanic.csv')
print(df.head())
print(df.describe())

# plot data with or without diff colors
def plot_data():
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])
    plt.show()

# add new boolean column into data
df['male'] = df['Sex'] == 'male'

#  evaluate data (take values from data columns)
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
# y = target data
y = df['Survived'].values

# add prediction model
model = LogisticRegression()

# train-test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=27)

def datasplit_pred():

    model.fit(X_train, y_train)
    print("'split' model.score:", model.score(X_test, y_test))

    y_pred = model.predict(X_test)
    print("'split' accuracy:", accuracy_score(y_test, y_pred))
    print("'split' precision:", precision_score(y_test, y_pred))
    print("'split' recall:", recall_score(y_test, y_pred))
    print("'split' f1_score:", f1_score(y_test, y_pred))
    return

def Kfold_model():
    X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
    y = df['Survived'].values
    scores = []
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticRegression()
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    print("K_fold scores:", scores)
    print("K_fold mean score:", np.mean(scores))
    return

all = "plot_data()", "datasplit_pred()", "Kfold_model()"
print("Functions", all)

output = datasplit_pred()
# print(output)

dump(model, 'output.joblib')



```

---

</SwmSnippet>

# Improving the model's accuracy with K-fold cross-validation

<SwmSnippet path="/app/train_titanic_data.py" line="1">

---

Finally, we use K-fold cross-validation to improve the model's accuracy. This technique involves splitting the data into K subsets and training the model K times, each time using a different subset as the test set and the remaining subsets as the training set. The model's performance is then averaged over the K iterations. This technique helps to ensure that our model's performance is not dependent on the particular way we split the data.

```python
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# import data frame to variable
df = pd.read_csv('data/titanic.csv')
print(df.head())
print(df.describe())

# plot data with or without diff colors
def plot_data():
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])
    plt.show()

# add new boolean column into data
df['male'] = df['Sex'] == 'male'

#  evaluate data (take values from data columns)
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
# y = target data
y = df['Survived'].values

# add prediction model
model = LogisticRegression()

# train-test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=27)

def datasplit_pred():

    model.fit(X_train, y_train)
    print("'split' model.score:", model.score(X_test, y_test))

    y_pred = model.predict(X_test)
    print("'split' accuracy:", accuracy_score(y_test, y_pred))
    print("'split' precision:", precision_score(y_test, y_pred))
    print("'split' recall:", recall_score(y_test, y_pred))
    print("'split' f1_score:", f1_score(y_test, y_pred))
    return

def Kfold_model():
    X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
    y = df['Survived'].values
    scores = []
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticRegression()
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    print("K_fold scores:", scores)
    print("K_fold mean score:", np.mean(scores))
    return

all = "plot_data()", "datasplit_pred()", "Kfold_model()"
print("Functions", all)

output = datasplit_pred()
# print(output)

dump(model, 'output.joblib')



```

---

</SwmSnippet>

# Conclusion

In conclusion, this code change implements a machine learning model to predict Titanic survival. The model is trained on a dataset of Titanic passengers, and its performance is evaluated using various metrics. The use of K-fold cross-validation helps to ensure that the model's performance is robust and not dependent on the particular way the data is split.

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBVGl0YW5pY19NYWNoaW5lX0xlYXJuaW5nX2Zyb21fRGlzYXN0ZXIlM0ElM0FBbWlyYW5Hb3phbGlzaHZpbGk=" repo-name="Titanic_Machine_Learning_from_Disaster"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>

import pandas as pd
import numpy as np
# import data frame to variable
df = pd.read_csv('data/titanic.csv')
print(df.head())
print(df.describe())

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# plot data with or without diff colors
def plot_data():
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])
    plt.show()
# plot_data()

# logistic regresion
from sklearn.linear_model import LogisticRegression

# add new boolean column into data
df['male'] = df['Sex'] == 'male'

#  evaluate data (take values from data columns)
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
# y = target data
y = df['Survived'].values

# add prediction model
model = LogisticRegression()
# input data to prediction model
def nosplit_pred():
    model.fit(X,y)
    pred = model.predict(X)
    # print(pred)
    y_pred = model.predict(X)
    print("'nosplit' sum of True predictions:", (y == y_pred).sum())  # how many true prediction
    print("'nosplit' accuracy:", ((y == y_pred).sum() / y.shape[0]))  # accuaracy in percents
    print("'nosplit' model score:", model.score(X, y))  # accuracy with score mathod
    print('model_coef:',model.coef_, 'model_intercpt:', model.intercept_)
# nosplit_pred()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# train-test data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=27)

def datasplit_pred():
    print("'split' whole dataset:", X.shape, y.shape)
    print("'split' training set:", X_train.shape, y_train.shape)
    print("'split' test set:", X_test.shape, y_test.shape)

    model.fit(X_train, y_train)
    print("'split' model.score:", model.score(X_test, y_test))

    y_pred = model.predict(X_test)
    print("'split' accuracy:", accuracy_score(y_test, y_pred))
    print("'split' precision:", precision_score(y_test, y_pred))
    print("'split' recall:", recall_score(y_test, y_pred))
    print("'split' f1_score:", f1_score(y_test, y_pred))
# datasplit_pred()

from sklearn.model_selection import KFold

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
# Kfold_model()

all = "plot_data()", "nosplit_pred()", "datasplit_pred()", "Kfold_model()"
print("Functions", all)

output = datasplit_pred()
# print(output)
from joblib import dump
dump(model, 'output.joblib')



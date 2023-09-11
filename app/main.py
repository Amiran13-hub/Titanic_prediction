import joblib
from fastapi import FastAPI

app = FastAPI()

model = joblib.load('output.joblib')

@app.get('/')
def get_root():

	return {'message': 'Welcome to the Titanic prediction API'}

@app.get('/predict')
def predict(Pclass, male, Age, Siblings_Spouses, Parents_Children, Fare):
	y_pred = model.predict([[int(Pclass), int(male), int(Age), int(Siblings_Spouses), int(Parents_Children), int(Fare)]])[0]

	return "survived" if int(y_pred)==1 else "not survived"
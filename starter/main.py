# autopep8 --in-place --aggressive --aggressive
# Put the code for your API here.
import os
import joblib
import pandas as pd
from fastapi import FastAPI, Query
from typing import Union
from pydantic import BaseModel
from starter.ml.model import inference, Model
from starter.ml.data import process_data

# Declare the data object with its components and their type.

# provide input structure
class Features(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float = Query(..., alias="education-num")
    marital_status: str = Query(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Query(..., alias="capital-gain")
    capital_loss: float = Query(..., alias="capital-loss")
    hours_per_week: float = Query(..., alias="hours-per-week")
    native_country: str = Query(..., alias="native-country")

# load model
model = Model(preprocessor = process_data) 
model.load_weights(os.path.join('model', 'model.pkl'))

# provide input sample
input_sample = {"age": "39",
                "workclass": "State-gov",
                "fnlgt": "77516",
                "education": "Bachelors",
                "education-num": "13",
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": "2174",
                "capital-loss": "0",
                "hours-per-week": "40",
                "native-country": "United-States"}


# Instantiate the app.
app = FastAPI()

# Define a GET on the root giving a welcome message.
@app.get("/")
async def welcome_message():
    return {"welcome message": "Hi, Welcome!!"}


# Define a POST that does model inference.
@app.post("/predict")
async def get_prediction(features: Features):
    features = pd.DataFrame(pd.DataFrame(
        {k: [v] for k, v in features.dict(by_alias=True).items()}))
    preds = inference(model, features)[0]
    return {"result": preds}

# autopep8 --in-place --aggressive --aggressive
# Put the code for your API here.
import os
import joblib
import pandas as pd
from fastapi import FastAPI, Query
from typing import Union
from pydantic import BaseModel
from .starter.ml.model import inference, Model
from .starter.ml.data import process_data

# Declare the data object with its components and their type.

# for heroku environment
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc") #rm -r .dvc .apt/usr/lib/dvc
    
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
model.load_weights(os.path.join('starter', 'model', 'model.pkl'))


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

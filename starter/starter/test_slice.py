import os
import joblib
import pandas as pd
import pytest
from ml.model import train_model, inference, compute_model_metrics



@pytest.fixture
def testdata():
    df = pd.read_csv(os.path.join('./', 'starter' ,'data', 'test_census.csv'))
    return df

@pytest.fixture
def traindata():
    df = pd.read_csv(os.path.join('./', 'starter','data', 'train_census.csv'))
    return df



def test_train_model(traindata):
    
    model = train_model(traindata)
    assert(1, "train_model passed.")
    
def test_inference(traindata, testdata):
    model = train_model(traindata)
    joblib.dump(model, 'model.pkl')
    model = joblib.load('model.pkl')
    preds = inference(model, testdata)
    assert(1, "Loaded model can used for inference. Predicted samples for this test are {preds[:10]}...")      
    
def test_compute_model_metrics(traindata, testdata):
    model = train_model(traindata)
    y_test, preds = model.predict(testdata) 
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert(1, "compute_model_metrics passed.")    

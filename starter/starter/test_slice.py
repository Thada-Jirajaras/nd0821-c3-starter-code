import os
import joblib
import pandas as pd
import numpy as np
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
    for att in ['model', 'encoder', 'lb']:
        assert hasattr(model, att), f'''train_model failed. The attribute "{att}" not found'''
    assert (model.lb.classes_ == np.array(['<=50K', '>50K'])).all(), "model.lb.classes_ != ['<=50K', '>50K']"
    
def test_inference(traindata, testdata):
    # train model
    model = train_model(traindata)
    joblib.dump(model, 'model.pkl')
    model = joblib.load('model.pkl')
    
    # test inference case 1
    testdata = testdata.drop(columns = ['salary'])
    preds = model.predict(testdata)
    
    # test inference case 2  
    preds = inference(model, testdata)
    
    
    
    assert set(preds) == set(['<=50K', '>50K']), f"Predict result set {set(preds)} not equal label set {set(['<=50K', '>50K'])}"    
    
def test_compute_model_metrics(traindata, testdata):
    model = train_model(traindata)
    y_test, preds = model.predict(testdata) 
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert fbeta > 0.5, f"Performance is too low. fbeta = {fbeta} <= 0.5 "  

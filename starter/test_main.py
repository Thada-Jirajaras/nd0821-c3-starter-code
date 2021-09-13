from fastapi.testclient import TestClient

# Import our app from main.py.
import os
import pytest
import pandas as pd
from .main import app

# Instantiate the testing client with our app.
client = TestClient(app)

@pytest.fixture
def positive_samples():
    df = pd.read_csv(os.path.join(
        'starter',
        'data',
        'test_prediction.csv')
    )
    samples = df[df['prediction']=='>50K'].sample(n=10,random_state=9).drop(columns = ['salary', 'prediction']).to_dict('record')
    return samples

@pytest.fixture
def negative_samples():
    df = pd.read_csv(os.path.join(
        'starter',
        'data',
        'test_prediction.csv')
    )
    samples = df[df['prediction']=='<=50K'].sample(n=10,random_state=9).drop(columns = ['salary', 'prediction']).to_dict('record')
    return samples


# Write tests using the same syntax as with the requests module.
def test_rootaddress():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"welcome message":"Hi, Welcome!!"}
    
def test_get_prediction_negative(negative_samples):
    
    for input_sample in negative_samples:
        r = client.post(url = "/predict", json = input_sample)
        assert r.status_code == 200  
        assert r.json() == {"result": "<=50K"}
    
def test_get_prediction_positive(positive_samples):
    for input_sample in positive_samples:
        r = client.post(url = "/predict", json = input_sample)
        assert r.status_code == 200  
        assert r.json() == {"result": ">50K"}   

from fastapi.testclient import TestClient

# Import our app from main.py.
from .main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_rootaddress():
    r = client.get("/")
    assert r.status_code == 200
    
def test_get_prediction1():
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
    
    r = client.post(url = "/predict", json = input_sample)
    assert r.status_code == 200     
    
def test_get_prediction2():
    # insert some 1 strange feature (native-country) and see if model still be able to get prediction 
    input_sample = {"age": 39,
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
                "native-country": "Strange-Country"}    
    
    
    r = client.post(url = "/predict", json = input_sample)
    assert r.status_code == 200, r.status_code       

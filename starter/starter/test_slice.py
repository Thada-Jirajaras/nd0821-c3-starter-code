import os
import joblib
import pandas as pd
import pytest
from ml.model import compute_model_metrics


@pytest.fixture
def model():
    trained_model = joblib.load(os.path.join('..', 'model', 'model.pkl'))
    return trained_model


@pytest.fixture
def testdata():
    df = pd.read_csv(os.path.join('..', 'data', 'test_census.csv'))
    return df


@pytest.fixture
def cat_features():
    features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return features

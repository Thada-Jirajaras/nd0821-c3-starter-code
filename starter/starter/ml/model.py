"""
library for model training and inference
"""
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
import xgboost as xgb
from .data import process_data


# define a model class
class Model:
    def __init__(self, model=xgb.XGBClassifier()):
        self.model = model
        self.cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

    def fit(self, train, label="salary"):

        # train the model
        X_train, y_train, encoder, lb = process_data(
            train, categorical_features=self.cat_features, label=label, training=True)
        self.model = xgb.XGBClassifier()
        self.model.fit(X_train, y_train)

        # keep components
        self.encoder = encoder
        self.lb = lb
        self.label = label

    def predict(self, test, label=None):

        if self.label in test.columns:
            label = self.label
        X_test, y_test, encoder, lb = process_data(
            test, categorical_features=self.cat_features, label=label,
            training=False, encoder=self.encoder, lb=self.lb
        )

        preds = self.model.predict(X_test)
        return preds, y_test

# Optional: implement hyperparameter tuning.


def train_model(train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = Model()
    model.fit(train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds, _ = model.predict(X)
    preds = model.lb.inverse_transform(preds)
    return preds

#


def performance_on_slices(model, testdata, cat_features):
    """ Test performance on slices"""

    performance_on_slices = []
    subdata = testdata
    y_test, preds = model.predict(subdata)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    performance_on_slices.append(
        {"Feature": "All", "Value": "All", 'N': len(subdata), "fbeta": fbeta, "precision": precision, "recall": recall})
    for cat_feat in cat_features:
        for cat_value in testdata[cat_feat].unique():
            subdata = testdata[testdata[cat_feat] == cat_value]
            y_test, preds = model.predict(subdata)
            precision, recall, fbeta = compute_model_metrics(y_test, preds)
            performance_on_slices.append(
                {"Feature": cat_feat, "Value": cat_value, 'N': len(subdata), "fbeta": fbeta, "precision": precision, "recall": recall})
    performance_on_slices = pd.DataFrame(performance_on_slices)
    return performance_on_slices

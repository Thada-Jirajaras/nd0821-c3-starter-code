"""
Train a model to predict whether income exceeds $50K/yr based on census data.
"""
# general imports
import os
import joblib
import pandas as pd

# Script to train machine learning model.
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, performance_on_slices, Model

# Add code to load in the data.
data = pd.read_csv(os.path.join('..', 'data', 'cleaned_census.csv'))

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)
train.to_csv(os.path.join('..', 'data', 'train_census.csv'), index=False)
test.to_csv(os.path.join('..', 'data', 'test_census.csv'), index=False)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model pipeline.
model = train_model(train)
model.save_weights(os.path.join('..', 'model', 'model.pkl'))


# Evaluation
model = Model(preprocessor = process_data)
model.load_weights(os.path.join('..', 'model', 'model.pkl'))
y_test, preds = model.predict(test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"fbeta = {fbeta}")
performance_by_group = performance_on_slices(model, test, cat_features)
performance_by_group.to_csv(
    os.path.join(
        '..',
        'data',
        'performance_by_group.csv'),
    index=False)

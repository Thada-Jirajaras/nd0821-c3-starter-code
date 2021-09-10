# Model Card

## Model Details

Thada created the model. The model pipline consists of 

1. a OnehotEncoder for the catagorical features
2. a label binarizer for the labels (the positive class '>50K' is encoded to 1 and '<=50K' is encoded as 0)
3. and an XGBoost model (XGBoost=1.4.0) with default parameters.


## Intended Use

Model is trained to predict whether income exceeds $50K/yr based on census data from various countries. 

## Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The original data set has 32,561 rows, and a 80-20 split was used to break this into a train and test set. To use the data for training a label binarizer was used on the labels.

## Metrics

fbeta is used in this project. Overall performance is fbeta=0.71 Perfomance of the test data by groups can be found at ./data/performance_by_group.csv

## Bias
The majority of the data is from race=White (N=27,816) so the perfomance of other races may drops (please see more bias detail at ./data/performance_by_group.csv)

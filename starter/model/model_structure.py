class Model:
    def __init__(self, model, preprocessor):
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
        self.model = model
        self.preprocessor = preprocessor

    def fit(self, train, label="salary"):

        # train the model
        X_train, y_train, encoder, lb = self.preprocessor(
            train, categorical_features=self.cat_features, label=label, training=True)
        self.model.fit(X_train, y_train)

        # keep components
        self.encoder = encoder
        self.lb = lb
        self.label = label

    def predict(self, test, label=None):

        if self.label in test.columns:
            label = self.label

        X_test, y_test, encoder, lb = self.preprocessor(
            test, categorical_features=self.cat_features, label=label,
            training=False, encoder=self.encoder, lb=self.lb
        )

        preds = self.model.predict(X_test)
        return preds, y_test
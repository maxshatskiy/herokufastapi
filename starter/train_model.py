# Script to train machine learning model.
from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model
from ml.model import compute_model_metrics
from ml.model import inference
from ml.model import compute_model_metrics_on_cat_feature
import joblib

data = pd.read_csv("./data/census_сleaned.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
# Train and save a model.
model = train_model(X_train, y_train)
preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

metrics_on_slices = compute_model_metrics_on_cat_feature(preds, y_test, test, cat_feature=cat_features[0])
with open("slice_output.txt", 'w') as f:
    for key, val in metrics_on_slices.items():
        f.write('%s:%s\n' % (key, val))
f.close()

joblib.dump(model,'./model/logistic_regression_model.joblib')
joblib.dump(encoder,'./model/encoder.joblib')
from sklearn.model_selection import train_test_split
from sklearn.linear_model._logistic import LogisticRegression
import pandas as pd
from numpy import ndarray
from starter.ml.data import process_data
from starter.ml.model import train_model
from starter.ml.model import compute_model_metrics
from starter.ml.model import inference
import pytest

@pytest.fixture(scope='module')
def prepare_test():
    data = pd.read_csv("../data/census_сleaned.csv")
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
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary", training=False,
                                        encoder=encoder, lb=lb)

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    result = {'X_train':X_train, 'y_train':y_train, 'X_test': X_test, 'y_test':y_test,'model':model,'preds':preds,'precision':precision,'recall':recall,'fbeta':fbeta}

    return result

def test_train_model(prepare_test):
    result = prepare_test
    assert isinstance(result['model'], LogisticRegression), "wrong type of the model returned"

def test_inference(prepare_test):
    result = prepare_test
    assert isinstance(result['preds'], ndarray), "type of the prodictions is wrong"

def test_compute_model_metrics(prepare_test):
    result = prepare_test
    assert isinstance(result['precision'], float), "type of the precision is wrong"
    assert isinstance(result['recall'], float), "type of the recall is wrong"
    assert isinstance(result['fbeta'], float), "type of fbeta is wrong"




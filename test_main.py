from fastapi.testclient import TestClient
import json
from main import app

client = TestClient(app)

def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {'Message': 'Hello this is a census application'}

def test_post_prediction_0():

    data = {"age": 39, "workclass": "State-gov", "fnlgt": 77516, "education": "Bachelors", "education-num": 13,
               "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family",
               "race": "White", "sex": "Male", "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40,
               "native-country": "United-States"}
    response = client.post("/inference", json=data)

    assert response.status_code == 200
    assert response.json() == {"prediction": 0}

def test_post_prediction_1():

    data = {"age": 39, "workclass": "State-gov", "fnlgt": 77516, "education": "Bachelors", "education-num": 13,
               "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family",
               "race": "White", "sex": "Male", "capital-gain": 100000, "capital-loss": 0, "hours-per-week": 40,
               "native-country": "United-States"}
    response = client.post("/inference", json=data)

    assert response.status_code == 200
    assert response.json() == {"prediction": 1}
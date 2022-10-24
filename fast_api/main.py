from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from model_train.ml.data import process_data
from model_train.ml.model import inference
import pandas as pd
import uvicorn
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
     os.system("dvc config core.no_scm true")
     if os.system("dvc pull") != 0:
         exit("dvc pull failed")
     os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

class Item(BaseModel):
    name:str
    price:float
    is_offer: Union[bool, None] = None

class Person(BaseModel):
    age: int = Field(None, example=39)
    workclass: str = Field(None, example="State-gov")
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example="Bachelors")
    education_num : int = Field(None, alias="education-num", example=13)
    marital_status: str = Field(None, alias="marital-status", example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Male')
    capital_gain: float = Field(None, alias='capital-gain', example=2174)
    capital_loss: float = Field(None, alias='capital-loss', example=0)
    hours_per_week: float = Field(None, alias='hours-per-week', example=40)
    native_country: str = Field(None, alias='native-country', example='United-States')

    class Config:
        schema_extra = {
            "example": {"age": 39,
                        "workclass": "State-gov",
                        "fnlgt": 77516,
                        "education": "Bachelors",
                        "education-num": 13,
                        "marital-status": "Never-married",
                        "occupation": "Adm-clerical",
                        "relationship": "Not-in-family",
                        "race": "White",
                        "sex": "Male",
                        "capital-gain": 2174,
                        "capital-loss": 0,
                        "hours-per-week": 40,
                        "native-country": "United-States"}
            }

model = joblib.load('./model/logistic_regression_model.joblib')
encoder = joblib.load('./model/encoder.joblib')

@app.get("/")
def read_root():
    return {'Message': 'Hello this is a census application'}

@app.post("/inference")
def model_inference(data:Person):

    data = data.dict(by_alias=True)
    df = pd.DataFrame(data, index=[0])
    age = data['age']
    workclass = data['workclass']
    fnlgt = data['fnlgt']
    education = data['education']
    education_num = data['education-num']
    marital_status = data['marital-status']
    occupation = data['occupation']
    relationship = data['relationship']
    race = data['race']
    sex = data['sex']
    capital_gain = data['capital-gain']
    capital_loss = data['capital-loss']
    hours_per_week = data['hours-per-week']
    native_country = data['native-country']

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

    X_test_encoded, _, _, _ = process_data(df, categorical_features=cat_features, label=None, training=False,
                                        encoder=encoder)

    preds = inference(model, X_test_encoded)

    return {"prediction": preds.tolist()[0]}

if __name__=="__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)



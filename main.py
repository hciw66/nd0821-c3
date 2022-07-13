# Put the code for your API here.
import pandas as pd
from typing import Union, Optional, List

from fastapi import FastAPI, Path, Body
from pydantic import BaseConfig, BaseModel, Field
from pydantic.validators import str_validator

from pathlib import Path
import sys
import os

starter_path = Path(__file__).parent
sys.path.append(os.path.join(starter_path, 'starter'))

from ml.data import *
import pickle
import numpy as np

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
    
# get encoder, lb
encoder = pickle.load(open(os.path.join(starter_path, 'model/OneHotEncoder.pkl'),'rb'))
lb = pickle.load(open(os.path.join(starter_path,'model/LabelEncoder.pkl'),'rb'))
# get model
model = pickle.load(open(os.path.join(starter_path,'model/RandomForest_model.pkl'),'rb'))


class MyStr(str):
    @classmethod
    def __get_validators__(cls):
        yield str_validator
        yield cls._strip_whitespace
        # Yield more custom validators if needed

    @classmethod
    def _strip_whitespace(cls, value: str, config: BaseConfig) -> str:
        if config.anystr_strip_whitespace:
            return value.strip()
        return value

    
example = {"age": 39,
           " workclass": ' State-gov',
           " fnlgt": 77516,
           " education": ' Bachelors',
           " education-num": 13,
           " marital-status": ' Never-married',
           " occupation": ' Adm-clerical',
           " relationship": ' Not-in-family',
           " race": ' White',
           " sex": ' Male',
           " capital-gain": 2174,
           " capital-loss": 0,
           " hours-per-week": 40,
           " native-country": ' United-States'
          }


class User(BaseModel):
    age: int              
    workclass: MyStr = Field(alias=' workclass')    
    fnlgt: int = Field(alias=' fnlgt')         
    education: MyStr = Field(alias=' education')        
    education_num: int = Field(alias=' education-num')     
    marital_status: MyStr = Field(alias=' marital-status') 
    occupation: MyStr = Field(alias=' occupation')       
    relationship: MyStr  = Field(alias=' relationship')    
    race: MyStr = Field(alias=' race')             
    sex: MyStr = Field(alias=' sex')             
    capital_gain: int = Field(alias=' capital-gain')      
    capital_loss: int = Field(alias=' capital-loss')     
    hours_per_week: int = Field(40, alias=' hours-per-week')    
    native_country: MyStr = Field(alias=' native-country')   
    salary: Optional[MyStr] = Field(alias=' salary')   
        
    class Config:
        anystr_strip_whitespace = True
        schema_extra = {
            "example": example
        }
        
        
class UserList(BaseModel):
    each_item: List[User] 
        
        
cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country"
]
  

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to nd0821-c3 project!"}


@app.post("/predict_user_batch")
async def predict(user_list: UserList):
    data = pd.DataFrame([t.__dict__ for t in user_list.each_item])
    
    if 'salary' in data.columns:
        y = data['salary']
        X = data.drop(['salary'], axis=1)
    else:
        y = np.array([])
        X = data       

    X_categorical = X[cat_features].values
    X_continuous = X.drop(*[cat_features], axis=1)
    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)
       
    preds = model.predict(X).tolist()
    out_list = list()
    for idx in range(len(preds)):
        if preds[idx] == 0:
            out_list.append('<=50K')
        else: 
            out_list.append('>50K')
    return {"salary prediction": out_list}    
        

@app.post("/predict_user")
async def predict(user: User = Body(example)):
    user_dict = user.dict()
    
    if 'salary' in user_dict.keys():
        y = user_dict.pop('salary')
    
    X_categorical = np.array([user_dict[col] for col in cat_features])
    X_continuous = np.array([user_dict[col] for col in user_dict.keys() if col not in cat_features]).reshape(1,-1)
    X_categorical = encoder.transform(X_categorical.reshape(1,-1))
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    print(X.shape)
    #preds = model.predict(X).tolist()
    pred = model.predict(X).tolist()
    out_list = list()
    for idx in range(len(pred)):
        if pred[idx] == 0:
            out_list.append('<=50K')
        else: 
            out_list.append('>50K')
    return {"salary prediction": out_list}    
    

    

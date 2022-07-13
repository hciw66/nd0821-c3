#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from pydantic import BaseConfig, BaseModel, Field
from pydantic.validators import str_validator
from typing import Union, Optional, List
from fastapi.encoders import jsonable_encoder

import requests

url = 'https://census-rfmodel.herokuapp.com/predict_user'
    
user = {"age": 39,
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

x = requests.post(url, json = user)

print(x.text)

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
        
file_path = './data/census_100.csv'
dict_from_csv = pd.read_csv(file_path).to_dict(orient='records')
user_list = UserList(each_item= dict_from_csv)

url = 'https://census-rfmodel.herokuapp.com/predict_user_batch'
from fastapi.encoders import jsonable_encoder
x = requests.post(url, json = jsonable_encoder(user_list))

print(x.text)


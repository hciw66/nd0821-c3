from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import pandas as pd
import os
import sys
from pathlib import Path
import json
from pydantic import BaseConfig, BaseModel, Field
from pydantic.validators import str_validator

from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to nd0821-c3 project!"}

"""    
def test_predict_file():
    response = client.post("/predict_file//home/huichuan/census_100.csv")
    assert response.status_code == 200
    assert len(response.json()['salary prediction']) == 100
"""
    
def test_predict_user_pos():
    response = client.post("/predict_user",
                              json = {"age": 39,
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
                                     " native-country": ' United-States',
                                     " salary": ' <=50K'} , )
    
    assert response.status_code == 200
    assert response.json() == {"salary prediction": ['<=50K']}    
                     
    
def test_predict_user_neg():
    response = client.post("/predict_user",
                              json = {"age": 39,
                                     " workclass": ' State-gov',
                                     " fnlgt": 77516,
                                     " education": ' Doctorate',
                                     " education-num": 18,
                                     " marital-status": ' Never-married',
                                     " occupation": ' Adm-clerical',
                                     " relationship": ' Not-in-family',
                                     " race": ' White',
                                     " sex": ' Male',
                                     " capital-gain": 12174,
                                     " capital-loss": 0,
                                     " hours-per-week": 40,
                                     " native-country": ' United-States',
                                     " salary": ' <=50K'} , )
    
    assert response.status_code == 200
    assert response.json() == {"salary prediction": ['>50K']}    
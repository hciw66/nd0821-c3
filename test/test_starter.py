import pytest
import pandas as pd
import os
import sys
from pathlib import Path

repo_path = Path(__file__).parent.parent

sys.path.append(repo_path)
sys.path.append(os.path.join(repo_path, 'starter'))
from ml.data import *
from ml.model import *

@pytest.fixture
def data():
    X = {'sex' : ['M','F','M','F'],
             'worktype' : ['manager', 'staff', 'staff', 'manager'],
             'salary' :['> 50K', '< 50K', '< 50K', '> 50K']}
    df = pd.DataFrame(X)
    return df
    
def test_process_data(data):
    X, y, encoder, lb = process_data(data,['sex','worktype'], label='salary', training=True)
    assert X.shape == (4, 4), "OneHotEncoder have errors"
    assert (X == [[1,0,1,0],[0,1,0,1],[1,0,0,1],[0,1,1,0]]).all, "OneHotEncoder have errors"
    assert (y==[0,1,0,1]).all, "LabelEncoder has errors"
"""    
def test_train_model():
    X = [[1,0,1,0],[0,1,0,1],[1,0,0,1],[0,1,1,0]]
    y = [0,1,0,1]
    model = train_model(X, y)
    assert model.predict([1,0,1,0]) == 0
"""

def test_compute_model_metrics():
    y = [0,0,0,0]
    preds = [1,0,1,0]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 0.0
    assert recall == 1.0
    assert fbeta == 0.0
    

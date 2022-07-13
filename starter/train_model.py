# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import dvc
import dvc.api
from ml.data import *
from ml.model import *
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path
import logging

# get repo
repo_path = Path(__file__).parent.parent

logging.basicConfig(
    filename= repo_path / 'logs/census.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def dataslice_model_metrics(X, categorical_features, label, encoder, lb, feature, model):
    """This function return a list of metrics for data slices
       input:
           model : model to test
           X : np.array
               Processed data.
           y : np.array
               Processed labels for X
           feature : a categorical feature in dataset X 
       output: a list of tuple (feature_value, precision, recall, fbeta)
    """
    score_list = list()
    for value in X[feature].unique():  
        tmp_X, tmp_y, encoder, lb = process_data(X[X[feature] == value], categorical_features= categorical_features,\
                                                 label=label, training=False, encoder=encoder, lb=lb)
        tmp_pred = model.predict(tmp_X)
        precision, recall, fbeta = compute_model_metrics(tmp_y, tmp_pred)
        score_list.append((value, precision, recall, fbeta))
    return score_list

def clean_data(data):
    """ This function take in a dataframe and return a dataframe 
         1. by removing whitespace in the originalcategorical data and column name
         2. remove duplicate
    """
    #1. by removing whitespace in the originalcategorical data and column name
    data.columns = [col.strip() for col in data.columns]
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = [x.strip() for x in data[col]]
    #2. remove duplicate
    data.drop_duplicates(inplace=True)
    return data


# Add code to load in the data.
try:
    data_b = dvc.api.read(
       'data/raw/census.csv',
       repo='https://github.com/hciw66/nd0821-c3.git',
       mode='rb'
       )
    data = pd.read_csv(BytesIO(data_b))
    
    logging.info("INFO: data read in successful: data shape is {}.".format(data.shape))
except: 
    logging.error("ERROR: Can't read in data." )
    raise

data = clean_data(data)    

    
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=43)

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
try:
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    logging.info("INFO: process_data successful: X_train shape is {}.".format(X_train.shape))

    filename = repo_path / 'model/OneHotEncoder.pkl'
    pickle.dump(encoder, open(filename, 'wb'))
     
    filename = repo_path / 'model/LabelEncoder.pkl'
    pickle.dump(lb, open(filename, 'wb'))
    
except KeyError as e:
    logging.error("ERROR: " + str(e))
    
# Proces the test data with the process_data function.
try:
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb)
    logging.info("INFO: process_data successful: X_test shape is {}.".format(X_test.shape))
except:
    logging.error("ERROR: ")

# Train and save a model.

model = train_model(X_train, y_train)

preds = inference(model, X_test)

print(compute_model_metrics(y_test, preds))

# create a metric dictionary and save the data slice metrics
metric_dict = dict()
for feature in cat_features:
    metric_dict[feature] = dataslice_model_metrics(test, cat_features, 'salary', encoder, lb, feature, model)
filename =  repo_path / 'metrics/dataslice_metrics.pkl' 
pickle.dump(metric_dict, open(filename, 'wb'))

# write the output as text file
filename = repo_path / 'metrics/slice_output.txt'
with open(filename, 'w') as f: 
    for key, value in metric_dict.items(): 
        f.write('%s:%s\n' % (key, value))

# save model as a pickle file
filename = repo_path / 'model/RandomForest_model.pkl'
pickle.dump(model, open(filename, 'wb'))

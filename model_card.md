# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
Author: Ingrid Wu
Date: 07/11/2022

## Model Details
This model is trained as binary classifier with Random Forest model with hyper-parameter
   * parameters = {'max_depth':[10, 15, 20, 30], 
                  'n_estimators':[100, 150, 200, 250, 300],
                  'max_features' : ["sqrt","log2"]}
   The best model is saved as the model for the inference 
   The best hyper parameters are {'max_depth': 20, 'max_features': 'sqrt', 'n_estimators': 250} 
    * actually, the best model hyper-parameter has been output as {'max_depth': 20, 'max_features': 'sqrt', 'n_estimators': 250} some other time with very similar result.   
   
## Intended Use
* By providing user's information for 
  * age                int64
  * workclass         object
  * fnlgt              int64
  * education         object
  * education-num      int64
  * marital-status    object
  * occupation        object
  * relationship      object
  * race              object
  * sex               object
  * capital-gain       int64
  * capital-loss       int64
  * hours-per-week     int64
  * native-country    object

The model will predict will the user salary '>50K' or '<=50K' yearly.


## Training Data  and  Evaluation Data
The raw dataset can be downloaded from 
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data.
  
The raw data have lots of leading white spaces in the column name and string value. The raw data has being stripped white space and de-duplcated, then split to 80/20 and random_state=43 for training and testing. 


## Metrics

The metrics (precision : float, recall : float, fbeta : float) is 
(0.7741659538066724, 0.5880441845354126, 0.6683899556868538)
## Ethical Considerations
* There is no privacy information in the data set.

## Caveats and Recommendations
There is no location information in the training dataset. This is a short coming for the model since the salary is different for different location. 
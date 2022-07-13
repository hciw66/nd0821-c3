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

The model will predict will the user salary '>50K' or '<=50K' yearly

## Training Data


## Evaluation Data

## Metrics
_Please include the metrics used and your model's performance on those metrics._


## Ethical Considerations
* There is no privacy information in the data set.

## Caveats and Recommendations

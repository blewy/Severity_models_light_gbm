# Severity_models_light_gbm

Modelling Average cost for claims using Light_GBM - as a Tutorial on this algorithm.

Recently a new algorithm named Light GBM (lgbm) as released and I just wanted to try it out.
If you are an actuary or a data Scientist working on analising insurance data, you will like this prototype. I've use a dataset from kaggle wih information about claims cost.
In this tutorial I tried the lgbm to model the average cost (here half the risk tariff problem is solved... if you are and actuary, you are set for life). I just created some simple (and ugly) feature engineering, and train/valid/test data sets and try some models.

- The cross-valdation function (cv.lgbm) wasnt working for me so I used a validation set to test hyperparameters (with early stoping)

- In the code you can find the different programs implemented in a generic way that with a copy-past I could adapt this programs for other purposes. 
The names are sugestive, 
      1 - **Grid Search**: i tried a large grid and saved the best model; also saved the test results to be used as pre-samples for bayesian hyper-parameter optimization
      2 - **Random Search**: just created a huge gride (very thin) and sample a reasonable amount of parameters to test
      3 - **Bayesian Hyperparameter search** with the help of **rBayesianOptimization** package
      3.1-**Bayesian Hyperparameter search** usign the result obtoined from grid search as pre-samples
      4 - Simple Script to compare results
      
      


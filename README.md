# Severity_models_light_gbm

Modelling Average cost for claims using Light_GBM - a Tutorial.

Recently a new algorithm came out named Light GBM (lgbm) and I just wanted to try it out.
If you are an Actuary or a data Scientist working with insurance data, you will like this prototype. I've use a dataset from a kaggle competition about claims cost (I use these kind of data in my day to day job) [link:https://www.kaggle.com/c/allstate-claims-severity].

In this tutorial I tried the lgbm to model average cost per claims (half the risk tariff problem is solved... if you are an actuary you are set for life, just kidding). I created some simple (and ugly) features, train/valid/test data sets and tried some models.

- The cross-valdation function (cv.lgbm) wasn't working for me so I used a validation set to test hyperparameters (with early stoping) [now its working I missed a parameter]

- In the code you can find the different programs where I implemented in a generic way (so I think) that easely with a copy-past I could adapt this programs for other purposes. 

The names of the scripts are sugestive:

  1 - **Grid Search**: I tried a large grid and saved the best model; also saved the test results to be used as pre-samples for bayesian hyper-parameter optimization
      
  2 - **Random Search**: just created a huge grid (and an very thin one) and sample a reasonable amount of parameters to test
      
  3 - **Bayesian Hyperparameter search** with the help of **rBayesianOptimization** package
      
  3.1-**Bayesian Hyperparameter search** using the result obtained from grid search as pre-sample (R got stuck)
      
  4 - Simple Script to compare results
      
  
  Have fun and enjoy.
      


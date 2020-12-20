import os
import config
import model_dispatcher
import argparse 

import joblib
import pandas as pd 
from sklearn import metrics
from sklearn import tree 
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

#Algorithms to work with
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier



def run_tuning(models, X, y):
    kfold = StratifiedKFold(n_splits= config.FOLDS) #must be equal to FOLDS
    dict = {"Algorithm":[], "Best Score":[]}
    report = pd.DataFrame(dict)

    for model in models:
        print(model)
        print(models[model])
        mod = models[model]
        mod_params = model_dispatcher.model_param[model]
        gs_mod = GridSearchCV(mod, param_grid = mod_params, cv = kfold, scoring = "accuracy",
                        n_jobs = -1, verbose = 0)
        gs_mod.fit(X, y)
        gs_best = gs_mod.best_estimator_
        #Save the model
        joblib.dump(gs_mod, os.path.join(config.MODEL_OUTPUT,
         f"../models/model_{model}.bin"))
        report  = report.append({"Algorithm":model, 
            "Best Score":gs_mod.best_score_},
            ignore_index = True)
    print(report.sort_values(by = "Best Score", ascending = False))
    
    #Save the best 3 algorithms
    best = report.sort_values(by = "Best Score", ascending = False).head(3)
    best_models = list(best["Algorithm"].values)
    print(best_models)
    for model in best_models:
        joblib.dump(model.best_estimator, os.path.join(config.BEST_MODELS,
         f"../models/bestModels/model_{model}.bin"))
    
   


if __name__ == "__main__":
  
    target = config.TARGET
    num_folds = config.FOLDS
    seed = config.SEED
    X_train =  pd.read_csv("../input/X_train.csv")
    y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    scoring = config.SCORING
    models = model_dispatcher.MODELS
    print("MODEL TUNING RESULTS:")
    run_tuning(models, X_train, y_train)


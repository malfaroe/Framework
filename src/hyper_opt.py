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


kfold = StratifiedKFold(n_splits= config.FOLDS) #must be equal to FOLDS


h_file = pd.read_csv(config.HYPER_FILE)

#Mini dataset
# y = h_file.pop(h_file.iloc[:, config.TARGET].name).iloc[:50]
# X = h_file.iloc[:50, :]

#Normal dataset
y = h_file.pop(h_file.iloc[:, config.TARGET].name)
X = h_file
dict = {"Algorithm":[], "Best Score":[]}
report = pd.DataFrame(dict)


for model in model_dispatcher.models:
    print(model_dispatcher.models[model])
    mod = model_dispatcher.models[model]
    mod_params = model_dispatcher.model_param[model]
    gs_mod = GridSearchCV(mod, param_grid = mod_params, cv = kfold, scoring = "accuracy",
                    n_jobs = 1, verbose = 1)
    gs_mod.fit(X, y)
    gs_best = gs_mod.best_estimator_
    # print("Best score:", gs_mod.best_score_)
    report  = report.append({"Algorithm":model, 
        "Best Score":gs_mod.best_score_},
         ignore_index = True)
print(report.sort_values(by = "Best Score", ascending = False))



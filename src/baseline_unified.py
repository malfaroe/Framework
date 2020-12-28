# Baseline performance of both linear an ensemble models
#Selects the x best for further work in model tuning
#  Load libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot 
from pandas import read_csv 
from pandas import set_option
from pandas.plotting import scatter_matrix 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier


import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score


import config
import model_dispatcher
import os
import joblib




def run_cv(models, X, y):
    kfold = StratifiedKFold(n_splits= config.FOLDS) #must be equal to FOLDS
    dict = {"Algorithm":[], "Model_detail":[], "Score":[], 
    "Std_Error": []}
    report = pd.DataFrame(dict)

    for model in models:
        mod = models[model]
        cv_results = cross_val_score(estimator = models[model],
        X= X_train , y = y_train , scoring = scoring, cv = kfold)
        report  = report.append({"Algorithm":model,"Model_detail": mod,
         "Score": cv_results.mean(),
            "Std_Error":cv_results.std()}, 
            ignore_index = True)
    print(report[["Algorithm", "Score", "Std_Error"]].sort_values(by = "Score", 
    ascending = False))
    
    # #Save the best 5 algorithms
    # best = report.sort_values(by = "Score", ascending = False).head(5)
    # # best_models = best["Model_detail"].values
    # for model, name in zip(best["Model_detail"], best["Algorithm"]):
    #     joblib.dump(model, os.path.join(config.PRESELECTED_MODELS,
    #      f"model_{name}.bin"))

if __name__ == "__main__":
    target = config.TARGET
    num_folds = config.FOLDS
    seed = config.SEED
    X_train =  pd.read_csv("../input/X_train.csv")
    y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    
    scoring = config.SCORING
    models = model_dispatcher.MODELS
    print("UNIFIED BASELINE RESULTS:")
    run_cv(models, X_train, y_train)
    
 

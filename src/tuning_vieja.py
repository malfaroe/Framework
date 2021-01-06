#Tuning module for testing vieja randomsearch and pipeline diabetes

import os
import config
import model_dispatcher
import argparse 
import glob

import joblib
import pandas as pd 
from sklearn import metrics
from sklearn import tree 
import numpy as np
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score 



from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

#Algorithms to work with
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def ml_model(models, X_train, y_train):
    cv = RepeatedStratifiedKFold(n_splits=config.FOLDS, n_repeats=10, random_state=42)
    dict = {"Algorithm":[], "Model_detail":[], "Best Score":[], 
    "Std_test_score": [], "Validation Score":[], "Val_Error":[]}
    report = pd.DataFrame(dict)
    for model in models:
        print(model)
        print(models[model])
        mod = models[model]
        mod_params = model_dispatcher.model_param[model]
        random_search = RandomizedSearchCV(mod, mod_params, cv=cv, random_state= 42, n_jobs=-1, verbose=0 )
        pipe = make_pipeline(StandardScaler(),random_search)
        pipe.fit(X_train, y_train)
        gs_best = random_search.best_estimator_


        # #Save the model
        # joblib.dump(gs_best, os.path.join(config.MODEL_OUTPUT,
        #  f"../models/model_{model}.bin"))
        # report  = report.append({"Algorithm":model, "Model_detail": gs_best,
        #     "Best Score":accuracy_score(y_train, pipe.predict(X_train)), 
        #     "Std_test_score": random_search.cv_results_["std_test_score"].mean()},
        #     ignore_index = True)
        #Crossvalidation for validating in X_val
        cv_results = model_selection.cross_val_score(estimator = make_pipeline(StandardScaler(), gs_best),
        X= X_val , y = y_val , scoring = scoring, cv = cv)
    # ensemble.fit(X_train, y_train)
        report  = report.append({"Algorithm":model, "Model_detail": gs_best,
            "Best Score":accuracy_score(y_train, pipe.predict(X_train)), 
            "Std_test_score": random_search.cv_results_["std_test_score"].mean(),
            "Validation Score":cv_results.mean(), "Val_Error":cv_results.std()},
            ignore_index = True)
    print(report[["Algorithm", "Best Score", "Std_test_score", "Validation Score",
    "Val_Error"]].sort_values(by = "Val_Error", ascending = True))
        # y_pred_proba = pipe.predict_proba(X_test)[:,1]
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        # print("ROC Score : ",roc_auc_score(y_test, y_pred_proba))
        # print("Accuracy for train: ", accuracy_score(y_train, pipe.predict(X_train)))
        # print("Accuracy for test: " , accuracy_score(y_test, pipe.predict(X_test)))
     
     #Save the best 3 algorithms
    best = report.sort_values(by = "Val_Error", ascending = True).head(12)
    # best_models = best["Model_detail"].values
    for model, name in zip(best["Model_detail"], best["Algorithm"]):
        joblib.dump(model, os.path.join(config.BEST_MODELS,
         f"model_{name}.bin"))


if __name__ == "__main__":
    files = glob.glob("../models/bestModels/*")
    for f in files:
        os.remove(f)
    target = config.TARGET
    num_folds = config.FOLDS
    seed = config.SEED
    X_train =  pd.read_csv("../input/X_train.csv")
    y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    X_val =  pd.read_csv("../input/X_val.csv")
    y_val =  pd.read_csv("../input/y_val.csv").values.ravel()

    scoring = config.SCORING
    models = model_dispatcher.MODELS
    print("Shapes:", X_train.shape, y_train.shape)
    print("MODEL TUNING RESULTS WITH RANDOMSEARCH AND PIPELINE:")
    print("Tuning...")
    ml_model(models, X_train, y_train)
    #Aviso
    import beepy
    from beepy import beep
    beep(sound="ping")
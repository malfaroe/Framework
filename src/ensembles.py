##First time experimentation with ensemble learning 
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV 


from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score

import os
import config
import glob
import joblib
import model_dispatcher
import pandas as pd


def bagging(X_train, y_train):
    kfold = model_selection.StratifiedKFold(n_splits=10, random_state= 42)
    cart = DecisionTreeClassifier()
    num_trees = 100
    model = BaggingClassifier(base_estimator= cart, random_state= 42,
    n_estimators = num_trees)
    results = model_selection.cross_val_score(model,
    X_train, y_train, cv = kfold )
    print("Bagging Results:", results.mean())

def Boosting(X_train, y_train):
    seed = 42
    num_trees = 70
    kfold = model_selection.StratifiedKFold(n_splits = config.FOLDS,
    random_state= seed)
    model = AdaBoostClassifier( random_state= 42,
    n_estimators = num_trees)
    results = model_selection.cross_val_score(model,
    X_train, y_train, cv=kfold)
    print("Boosting Results:", results.mean())


def VotingEnsemble(estimators, X_train, y_train):
    print("Training...")
    kfold = model_selection.StratifiedKFold(n_splits = config.FOLDS)
    v_param_grid = {'voting':['soft',
                          'hard']} # tuning voting parameter
    ensemble = VotingClassifier(estimators)
    gsV = GridSearchCV(ensemble, 
                   param_grid = 
                   v_param_grid, 
                   cv = kfold, 
                   scoring = config.SCORING,
                   n_jobs = -1, 
                   verbose = 0)
    gsV.fit(X_train, y_train)
    v_best = gsV.best_estimator_

    name = "VotingEnsemble"
    
    # #save the votingclassifier
    joblib.dump(v_best, os.path.join(config.MODEL_OUTPUT,
         f"../models/ensembleModel/model_{name}.bin"))
    print("VotingClassifier CV Results:", gsV.best_score_, 
    gsV.cv_results_["std_test_score"].mean().std())
    #Validation in y_val using crossvalidation

    cv_results = model_selection.cross_val_score(estimator = v_best,
        X= X_val , y = y_val , scoring = scoring, cv = kfold)
    # ensemble.fit(X_train, y_train)
    print("VotingClassifier Validation Results using crossval:", cv_results.mean(),
    cv_results.std())

       



if __name__ == "__main__":
    target = config.TARGET
    num_folds = config.FOLDS
    seed = config.SEED
    X_train =  pd.read_csv("../input/X_train.csv")
    y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    X_val =  pd.read_csv("../input/X_val.csv")
    y_val =  pd.read_csv("../input/y_val.csv").values.ravel()
    #Reading/loading the best models
    all_files = glob.glob("../models/bestModels" + "/*.bin")
    estimators = []
    for filename in all_files:
        model = joblib.load(filename)
        estimators.append((filename, model))

    scoring = config.SCORING
    models = model_dispatcher.LINEAR_MODELS
    # bagging(X_train, y_train)
    # Boosting(X_train, y_train)
    VotingEnsemble(estimators, X_train, y_train)
    #Aviso
    import beepy
    from beepy import beep
    beep(sound="ping")



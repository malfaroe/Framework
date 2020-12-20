# Load libraries
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

import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score


import config
import model_dispatcher


def run_cv(models, X_train, y_train):
    results = []
    names = []
    for model in  models:
        kfold = StratifiedKFold(n_splits = num_folds)
        cv_results = cross_val_score(estimator = models[model], X = X_train, y = y_train, scoring = scoring, cv = kfold)
        results.append(cv_results)
        names.append(model)
        msg = "%s: %f (%f)" % (model, cv_results.mean(), cv_results.std()) 
        print(msg)
    #Plot the baseline performance of each algorithm
    fig = pyplot.figure()
    fig.suptitle("Baseline algorithm comparison")
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names) 
    pyplot.show()


if __name__ == "__main__":
    target = config.TARGET
    num_folds = config.FOLDS
    seed = config.SEED
    X_train =  pd.read_csv("../input/X_train.csv")
    y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    scoring = config.SCORING
    models = model_dispatcher.LINEAR_MODELS
    print("PRE-RESCALING:")
    run_cv(models, X_train, y_train)
    scaler = StandardScaler().fit(X_train)
    XRescaled = scaler.transform(X_train)
    print("Enter to continue...")
    print("")
    print("POST-RESCALING:")
    run_cv(models, XRescaled, y_train)
 

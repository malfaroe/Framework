#Testing loading models and predict
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


if __name__ == "__main__":
    target = config.TARGET
    num_folds = config.FOLDS
    seed = config.SEED
    X_train =  pd.read_csv("../input/X_train.csv")
    y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    X_val =  pd.read_csv("../input/X_val.csv")
    y_val =  pd.read_csv("../input/y_val.csv").values.ravel()


    scoring = config.SCORING
    model = joblib.load("../models/model_Gradient Boosting.bin")
    result = model.score(X_val, y_val)
    print(result)
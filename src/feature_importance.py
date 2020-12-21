
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd 
import numpy as np

import config
import os
import joblib 
import glob

import yellowbrick
from yellowbrick.model_selection import FeatureImportances

import warnings
warnings.filterwarnings("ignore")


##FEATURE IMPORTANCE

def feat_importance(all_files, X_train, y_train):
    names_classifiers = []
    for filename in all_files:
        model = joblib.load(filename)
        names_classifiers.append((filename, model.best_estimator_))

    for name, model in names_classifiers:
        viz = FeatureImportances(model)
        viz.fit(X_train, y_train)
        viz.show()



if __name__ == "__main__":
    
    X_train =  pd.read_csv("../input/X_train.csv")
    y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    all_files = glob.glob("../models/bestModels" + "/*.bin")
    feat_importance(all_files, X_train, y_train)

    
    #
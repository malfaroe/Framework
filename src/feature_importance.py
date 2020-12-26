
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectFromModel


##FEATURE IMPORTANCE


if __name__ == "__main__":
    
    X_train =  pd.read_csv("../input/train_final.csv")
    X_test =  pd.read_csv("../input/test_final.csv")
    y_train = X_train.pop(config.TARGET)


    # y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    all_files = glob.glob("../models/bestModels" + "/*.bin")

    #Split into train and test datasets
    X_t, X_val, y_t, y_val = train_test_split(X_train,y_train, test_size = 0.33, random_state = 1)

    #Fit a model
    model = LogisticRegression(solver = "liblinear")
    model.fit(X_t, y_t)

    #Evaluate the model
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_pred, y_val)

    #Results
    print("Initial shape of dataset:", X_train.shape)
    print('Accuracy of baseline model with all features: %.2f' % (acc*100))

    #Selectfrom Model (select the 20 best features)
    fs = SelectFromModel(RandomForestClassifier(n_estimators = 200), max_features = 25)
    #Fit
    fs.fit(X_t, y_t)
    #Transform train data
    X_train_fs = fs.transform(X_t)
    #Transform test data
    X_val_fs = fs.transform(X_val)
    feature_idx = fs.get_support()
    feature_name = X_train.columns[feature_idx]
    print("Selected features", feature_name)
    #Fit a model
    model = LogisticRegression(solver = "liblinear")
    model.fit(X_train_fs, y_t)

    #Evaluate the model
    y_pred = model.predict(X_val_fs)
    acc = accuracy_score(y_pred, y_val)

    #Results
    print('Accuracy of the selected feature model: %.2f' % (acc*100))

    X_train = pd.concat((y_train, X_train[feature_name]), axis = 1)
    print("Columnas X_train:", X_train.columns)
    
    if config.KAGGLE == True:
        X_test = X_test[feature_name] # actualizo columnas
        X_test.to_csv("../input/new_test_final.csv",index  = False)
        print("Shape of test:", X_test.shape)


    #Guardar el datset con las columnas seleccionadas para pasar a data_split
    print("Columnas finales X_train:", X_train.columns)
    X_train.to_csv("../input/new_train_final.csv",
    index  = False)

    
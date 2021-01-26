from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,GradientBoostingClassifier, AdaBoostClassifier
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
from lightgbm import LGBMClassifier
import xgboost 
# from xgboost import XGBClassifier

import numpy as np


MODELS =    {
    "decision_tree_gini":tree.DecisionTreeClassifier(criterion= "gini", random_state = 42),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion= "entropy",  random_state = 42),
    "RandomForest": RandomForestClassifier(random_state = 42),
    # # "ExtraTrees": ExtraTreesClassifier(random_state = 42),
    # # "GradientBoosting":GradientBoostingClassifier(random_state = 42),
     "CatBoostClassifier": CatBoostClassifier(random_state = 42, verbose = 0),
    # 'SVM': SVC(probability=True),
    # 'LinearDiscriminant': LinearDiscriminantAnalysis(),
    # "KNearest_Neighbour": KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(max_iter = 40000),
    # "AdaBoost":AdaBoostClassifier(DecisionTreeClassifier(),
    # random_state = 42),
    # "LGBM": LGBMClassifier(random_state = 42)
    
} 

LINEAR_MODELS = {'LogisticRegression': LogisticRegression(max_iter = 1000000),
'LinearDiscriminant': LinearDiscriminantAnalysis(),
"KNearest_Neighbour": KNeighborsClassifier(),
'CART': DecisionTreeClassifier(),
'GaussianNB': GaussianNB(),
'SVM': SVC()
}



#Parameters for hyperparameter optimization 

DTG_PARAMS = {"criterion": ["gini"],"max_depth":[5,10,15,20,25,30],
            "min_samples_split": range(2,50),
            "min_samples_leaf":range(1,50)}

DTE_PARAMS = {"criterion": ["entropy"],"max_depth":[5,10,15,20,25,30],
            "min_samples_split": range(2,50),
            "min_samples_leaf":range(1,50)}


RF_PARAMS = {"max_depth": [None], "max_features": [0.05,0.15, 0.25, 0.35],
                "min_samples_split": np.arange(2,30),
                "min_samples_leaf": np.arange(1,50),
                "bootstrap": [False],
                "n_estimators": [100,300, 1000],
                "criterion": ["gini"]}
XT_PARAMS = {"max_depth": [None], "max_features": [0.05,0.15, 0.25, 0.35],
                "min_samples_split": [2,3,10],
                "min_samples_leaf": [1,3,10],
                "bootstrap": [False],
                "n_estimators": [100,300],
                "criterion": ["gini"]}

GBC_PARAMS = {"loss": ["deviance"],
                 "n_estimators": [100,200, 300, 500, 1000],
                 "learning_rate": [0.1, 0.05, 0.01, 1],
                "max_depth": [4, 8], 
                "min_samples_leaf": [100, 150],
                 "max_features": [0.3, 0.1],
                }
ADA_PARAMS = {"base_estimator__criterion": ["gini", "entropy"],
                 "base_estimator__splitter":["best", "random"],
                     "algorithm":["SAMME", "SAMME.R"],
                     "n_estimators": [1,2,50, 100, 500],
                 "learning_rate": [0.0001, 0.001, 0.01, 0.1,0.5,  1.0, 1.5]}

CB_PARAMS = {'depth'         : [4,5,6,7,8,9, 10],
                 'learning_rate' : [0.01,0.02,0.03,0.04],
                  'iterations'    : [10, 20,30,40,50,60,70,80,90, 100]
                 }

SVM_PARAMS = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf','poly', 'sigmoid']}  

# KNN_PARAMS = {"n_neighbors": [3,5,11,19,23,31], 
#                 "weights": ["uniform", "distance"],
#                 "metric": ["euclidean", "manhattan"]}

KNN_PARAMS = {"n_neighbors": np.arange(1,50),
                "leaf_size" : np.arange(1,50),
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]}

LR_PARAMS = {"penalty" : ["l2"],
              "tol" : [0.0001,0.0002,0.0003],
              "max_iter": [100,200,300],
              "C" :[0.01, 0.1, 1, 10, 100],
              "intercept_scaling": [1, 2, 3, 4],
              "solver":['liblinear'],
              "verbose":[0]}

LDA_PARAMS = {"solver" : ["svd"],
              "tol" : [0.0001,0.0002,0.0003]}

LGBM_PARAMS = {
    'learning_rate': [0.005, 0.01],
    'n_estimators': [8,16,24],
    'num_leaves': [6,8,12,16], # large num_leaves helps improve accuracy but might lead to over-fitting
    'boosting_type' : ['gbdt', 'dart'], # for better accuracy -> try dart
    'objective' : ['binary'],
    'max_bin':[255, 510], # large max_bin helps improve accuracy but might slow down training progress
    'random_state' : [500],
    'colsample_bytree' : [0.64, 0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
               }

XGB_PARAMS = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }



model_param =   {
    "decision_tree_gini": DTG_PARAMS,
    "decision_tree_entropy": DTE_PARAMS,
    "RandomForest": RF_PARAMS,
    "ExtraTrees": XT_PARAMS,
    "GradientBoosting":GBC_PARAMS,
    "CatBoostClassifier": CB_PARAMS,
    "SVM": SVM_PARAMS,
    "KNearest_Neighbour": KNN_PARAMS,
    'LogisticRegression': LR_PARAMS,
    "LinearDiscriminant": LDA_PARAMS,
    "AdaBoost": ADA_PARAMS,
    "LGBM": LGBM_PARAMS,
  

   
} 


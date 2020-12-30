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

import numpy as np


MODELS =    {
    # "decision_tree_gini":tree.DecisionTreeClassifier(criterion= "gini", random_state = 42),
    # "decision_tree_entropy": tree.DecisionTreeClassifier(criterion= "entropy",  random_state = 42),
    "RandomForest": RandomForestClassifier(random_state = 42),
    # "ExtraTrees": ExtraTreesClassifier(random_state = 42),
    # "GradientBoosting":GradientBoostingClassifier(random_state = 42),
    "CatBoostClassifier": CatBoostClassifier(random_state = 42, verbose = 0),
    'SVM': SVC(),
    # 'LinearDiscriminant': LinearDiscriminantAnalysis(),
    "KNearest_Neighbour": KNeighborsClassifier(),
    # 'LogisticRegression': LogisticRegression(max_iter = 40000)

    
} 

LINEAR_MODELS = {'LogisticRegression': LogisticRegression(max_iter = 1000000),
'LinearDiscriminant': LinearDiscriminantAnalysis(),
"KNearest_Neighbour": KNeighborsClassifier(),
'CART': DecisionTreeClassifier(),
'GaussianNB': GaussianNB(),
'SVM': SVC()
}



#Parameters for hyperparameter optimization process

DTG_PARAMS = {"criterion": ["gini"],"max_depth":range(1,10),
            "min_samples_split": range(2,10),
            "min_samples_leaf":range(1,5)}

DTE_PARAMS = {"criterion": ["entropy"],"max_depth":range(1,10),
            "min_samples_split": range(2,10),
            "min_samples_leaf":range(1,5)}


RF_PARAMS = {"max_depth": [None], "max_features": [1,3,5],
                "min_samples_split": np.arange(2,30),
                "min_samples_leaf": np.arange(1,50),
                "bootstrap": [False],
                "n_estimators": [100,300, 1000],
                "criterion": ["gini"]}
XT_PARAMS = {"max_depth": [None], "max_features": [1,3, 5],
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

KNN_PARAMS = {"n_neighbors": [3,5,11,19,23,31], 
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
    "LinearDiscriminant": LDA_PARAMS

   
} 


from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
import numpy as np

import config
import os
import joblib 
import glob



# check scikit-learn version
import sklearn
print(sklearn.__version__)




if __name__ == "__main__":
    
    X_train =  pd.read_csv("../input/X_train.csv")
    y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    # all_files = glob.glob("../models/bestModels" + "/*.bin")
    col_names = X_train.columns

    model_name = glob.glob("../models/bestModels" + "/model_RandomForest.bin")
    model = joblib.load(model_name[0])
    model.best_estimator_.fit(X_train, y_train)
    print(model)

    # model = RandomForestRegressor()
    # # fit the model
    # model.fit(X, y)
    # # get importance
    # importance = model.feature_importances_
    # # summarize feature importance
    # for i,v in enumerate(importance):
    #     print('Feature: %0d, Score: %.5f' % (i,v))
    # # plot feature importance
    # pyplot.bar([x for x in range(len(importance))], importance)
    # pyplot.show()
    # importances = model.feature_importances_
    # idxs = np.argsort(importances)
    
    # for filename in all_files:
    #     model = joblib.load(filename[0])
    #     model.fit(X_train, y_train)
    #     importances = model.feature_importances_
    #     idxs = np.argsort(importances)
    #     plt.title('Feature Importances')
    #     plt.barh(range(len(idxs)), importances[idxs], align='center')
    #     plt.yticks(range(len(idxs)), [col_names[i] for i in idxs])
    #     plt.xlabel('Random Forest Feature Importance')
    #     plt.show()
        # names_classifiers.append((filename, model.best_estimator_))

    
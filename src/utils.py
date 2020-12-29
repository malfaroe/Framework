import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from sklearn.compose import make_column_transformer
from itertools import combinations 
import config 

import pandas as pd 
import numpy as np

import config
import os
import joblib 
import glob


def Rescaler(df, target):
#Rescaling utility
    y = df.pop(target)
    X = df
    test_ID = y.index
    scaler = StandardScaler().fit(X)
    XRescaled = scaler.transform(X)
    df_rescaled = pd.DataFrame(XRescaled, columns = X.columns, index = test_ID)
    df = pd.concat((y, df_rescaled), axis = 1)
    df.set_index(test_ID)
    print("Data has been rescaled...")
    return df

# # check scikit-learn version
# import sklearn
# print(sklearn.__version__)




# if __name__ == "__main__":
    
#     X_train =  pd.read_csv("../input/X_train.csv")
#     y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
#     # all_files = glob.glob("../models/bestModels" + "/*.bin")
#     col_names = X_train.columns

#     model_name = glob.glob("../models/bestModels" + "/model_RandomForest.bin")
#     model = joblib.load(model_name[0])
#     model.best_estimator_.fit(X_train, y_train)
#     print(model)

#     # model = RandomForestRegressor()
#     # # fit the model
#     # model.fit(X, y)
#     # # get importance
#     # importance = model.feature_importances_
#     # # summarize feature importance
#     # for i,v in enumerate(importance):
#     #     print('Feature: %0d, Score: %.5f' % (i,v))
#     # # plot feature importance
#     # pyplot.bar([x for x in range(len(importance))], importance)
#     # pyplot.show()
#     # importances = model.feature_importances_
#     # idxs = np.argsort(importances)
    
#     # for filename in all_files:
#     #     model = joblib.load(filename[0])
#     #     model.fit(X_train, y_train)
#     #     importances = model.feature_importances_
#     #     idxs = np.argsort(importances)
#     #     plt.title('Feature Importances')
#     #     plt.barh(range(len(idxs)), importances[idxs], align='center')
#     #     plt.yticks(range(len(idxs)), [col_names[i] for i in idxs])
#     #     plt.xlabel('Random Forest Feature Importance')
#     #     plt.show()
#         # names_classifiers.append((filename, model.best_estimator_))

    
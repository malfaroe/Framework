#Module for checking the correlation between the models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import config
import glob
import joblib 




if __name__ == "__main__":
    target = config.TARGET
    num_folds = config.FOLDS
    scoring = config.SCORING
    seed = config.SEED
    X_train =  pd.read_csv("../input/X_train.csv")
    y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    X_val =  pd.read_csv("../input/X_val.csv")
    y_val =  pd.read_csv("../input/y_val.csv").values.ravel()
    X_test = pd.read_csv(config.TEST_FILE)
    #Reading/loading models from bestModels folder
    all_files = glob.glob("../models/bestModels" + "/*.bin")
    estimators = []
    for filename in all_files:
        model = joblib.load(filename)
        estimators.append((filename, model))
    corr = pd.DataFrame()
    for name, model in estimators:
        corr = pd.concat((corr, (pd.Series(model.predict(X_test),
         name = name))), axis = 1)
    plt.figure(figsize = (3,3))
    sns.set(font_scale= 0.8)
    sns.heatmap(corr.corr(), annot = True)
    plt.show()


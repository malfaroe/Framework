
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



##FEATURE IMPORTANCE

def feat_importance(all_files, X_train):
    nrows = 2
    ncols = 1
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))
    names_classifiers = []
    nclassifier = 0
    # all_files = glob.glob("../models" + "/*.bin")
    for filename in all_files:
            model = joblib.load(filename)
            names_classifiers.append((filename, model.best_estimator_))

    for row in range(nrows):
        for col in range(ncols):
            name = names_classifiers[nclassifier][0]
            classifier = names_classifiers[nclassifier][1]
            indices = np.argsort(classifier.feature_importances_)[::-1][:40]
            g = sns.barplot(y=X_train.columns[indices][:40], 
            x = classifier.feature_importances_[indices][:40] , 
            orient='h',ax=axes[row][col])
            g.set_xlabel("Relative importance",fontsize=12)
            g.set_ylabel("Features",fontsize=12)
            g.tick_params(labelsize=9)
            g.set_title(name + " feature importance")
            nclassifier += 1
    plt.show()



if __name__ == "__main__":
    
    df = pd.read_csv(config.DF)
    target = config.TARGET
    num_folds = config.FOLDS
    kfold = StratifiedKFold(n_splits = num_folds)
    seed = config.SEED
    X_train =  pd.read_csv("../input/X_train.csv")
    y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    scoring = config.SCORING
    all_files = glob.glob("../models/bestModels" + "/*.bin")
    feat_importance(all_files, X_train)

    
    #


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


##LEARNING CURVES FOR ANALYSIS OF THE ALGORITHMS
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt


if __name__ == "__main__":

    df = pd.read_csv(config.DF)
    target = config.TARGET
    num_folds = config.FOLDS
    kfold = StratifiedKFold(n_splits = num_folds)
    seed = config.SEED
    X_train =  pd.read_csv("../input/X_train.csv")
    y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    scoring = config.SCORING
    all_files = glob.glob("../models" + "/*.bin")

    for filename in all_files:
        model = joblib.load(filename)
        g = plot_learning_curve(model.best_estimator_,filename,X_train,y_train,cv=kfold)
    #
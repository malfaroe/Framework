import os
import config
import model_dispatcher
import argparse 

import joblib
import pandas as pd 
from sklearn import metrics
from sklearn import tree 
import numpy as np


def run(folds, models):
    #Takes a fold and separate/rotulating the train and validation set of that fold
    #read the data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    ##Create a dataframe for results
    dict = {"Algorithm":[], "CV_accuracy":[], "Validation error":[]}
    report = pd.DataFrame(dict)
    for model in  models.keys():#Instance of the algorithm
        clf = models[model]
        acc_model = []
        for fold in range(folds): #identifying train and validation sets for that fold
            df_train = df[df["Kfold"] != fold].reset_index(drop = True)
            df_val = df[df["Kfold"] == fold].reset_index(drop = True)

            #Create a X_train and y_train set from df_train. These files must be arrays
            y_train = df_train.iloc[:, int(target)]
            X_train = df_train.drop(columns = y_train.name, axis = 1).values
        
            #Create a X_val and y_val set from df_train. These files must be arrays
            y_val = df_val.iloc[:, int(target)]
            X_val = df_val.drop(columns = y_val.name, axis = 1).values
            #Fit the algorithm to the train set
            clf.fit(X_train, y_train)

            #Predic with validation
            y_pred = clf.predict(X_val)
            #Compute accuracy for the prediction
            accuracy = metrics.accuracy_score(y_val, y_pred)
            acc_model.append(accuracy)
        report  = report.append({"Algorithm":model, 
        "CV_accuracy":np.mean(acc_model), "Validation error":np.round(np.std(acc_model), 4)},
         ignore_index = True)

        
        # print(f"Model = {model}, CV_Average_accuracy = {np.mean(acc_model)}")

        #Save the model
        joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"../models/dt_{fold}.bin"))
    print(report.sort_values(by = "CV_accuracy", ascending = False))

if __name__ == "__main__":
    target = config.TARGET
    #Run
    run(folds = config.FOLDS, models = model_dispatcher.models)




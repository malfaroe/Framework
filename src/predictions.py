import pandas as pd 
import numpy as np 

import os
import config
import joblib
import glob #For importing files





if __name__ == "__main__":
    target = config.TARGET
    #Read inputs
    model_name = glob.glob("../models/bestModels" + "/model_ExtraTrees.bin")
    X_test = pd.read_csv(config.TEST_FILE)
    X_train =  pd.read_csv(config.TRAINING_FILE)
    y_train = X_train.pop(config.TARGET)

    # #testing bugs
    # print(X_train.columns)
    # print(X_test.columns)

    # # #Adapt feature columns of test set
    # columns = X_train.columns
    # X_test = X_test[columns]
    
    #Predict
    model = joblib.load(model_name[0])
    print("Nans in test:", X_test.isnull().sum().sum())

    print("Model fitting...")
    model.fit(X_train, y_train)
    print("Predicting...")
    y_pred = model.predict(X_test)
    print("Prediction done.")

    #Save prediction
    joblib.dump(y_pred, os.path.join("../input",
         f"../input/y_pred"))

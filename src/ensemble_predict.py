import pandas as pd 
import numpy as np 

import os
import config
import joblib
import glob #For importing files





if __name__ == "__main__":
    target = config.TARGET
    #Read inputs
    model_name = glob.glob("../models/ensembleModel" + "/*.bin")
    X_test = pd.read_csv(config.TEST_FILE)
    # X_train =  pd.read_csv("../input/X_train.csv")

    # #Adapt feature columns of test set
    # columns = X_train.columns
    # X_test = X_test[columns]

    ##FIT CON TODO EL DATASET, NO SOLO X_TRAIN!!!!!!


    X_train =  pd.read_csv("../input/X_train.csv")
    y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    
    #Predict
    model = joblib.load(model_name[0])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #Save prediction
    joblib.dump(y_pred, os.path.join("../input",
         f"../input/y_pred"))

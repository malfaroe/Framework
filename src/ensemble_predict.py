import pandas as pd 
import numpy as np 

import os
import config
import joblib
import glob 




if __name__ == "__main__":
    target = config.TARGET
    #Read inputs
    model_name = glob.glob("../models/ensembleModel" + "/*.bin")
    X_test = pd.read_csv(config.TEST_FILE)
    X_train =  pd.read_csv(config.TRAINING_FILE)
    y_train = X_train.pop(config.TARGET)
    
    #Predict
    model = joblib.load(model_name[0])
    print("Model fitting...")
    model.fit(X_train, y_train)
    print("Predicting...")
    y_pred = model.predict(X_test)
    print("Ensemble prediction done.")

    #Save prediction
    joblib.dump(y_pred, os.path.join("../input",
         f"../input/y_ensemble_pred"))
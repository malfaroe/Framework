import pandas as pd 
import numpy as np 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

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
    #Predict
    model = joblib.load(model_name[0])
    print("Model fitting...")
    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(X_train, y_train)
    print("Predicting...")
    y_pred = pipe.predict(X_test)
    print("Prediction done.")


    #Save prediction
    joblib.dump(y_pred, os.path.join("../input",
         f"../input/y_ensemble_pred"))
    #Aviso
    import beepy
    from beepy import beep
    beep(sound="ping")

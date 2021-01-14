import pandas as pd 
import numpy as np 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler



import os
import config
import joblib
import glob #For importing files





if __name__ == "__main__":
    target = config.TARGET
    #Read inputs
    model_name = glob.glob("../models/bestModels" + "/model_LogisticRegression.bin")
    X_test = pd.read_csv(config.TEST_FILE)
    X_train =  pd.read_csv(config.TRAINING_FILE)
    y_train = X_train.pop(config.TARGET)

    
    #Predict
    model = joblib.load(model_name[0])
    print("Model fitting...")
    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(X_train, y_train)
    print("Predicting...")
    y_pred = pipe.predict(X_test)
    print("Prediction done.")
    print("LOAN SUBMISSION READY")

    
    #Submit
    test_ID = pd.read_csv(config.TEST_ID)
    submit = pd.DataFrame()
    submit = pd.concat((submit, test_ID), axis =1)
    submit[target] = y_pred
    dict = {1: "Y", 0:"N"}
    submit["Loan_Status"] = submit["Loan_Status"].map(dict)
    submit.to_csv("../datasets/loanCompetition/submission_single.csv", index = False)
    
   
    #Aviso
    import beepy
    from beepy import beep
    beep(sound="ping")
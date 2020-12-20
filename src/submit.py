import pandas as pd
import numpy as np

import config
import glob
import joblib

#Import input (ID, target, y_pred)

if __name__ == "__main__":
    target = config.TARGET
    id_file = glob.glob("../input" + "/test_ID")
    pred_file =  glob.glob("../input" + "/y_pred")
    # test_ID = joblib.load(id_file[0])
    test_ID = pd.read_csv(config.TEST_ID, index_col= 0)
    y_pred = joblib.load(pred_file[0])
    submit = pd.DataFrame()
    submit = pd.concat((submit, test_ID), axis =1)
    submit['Survived'] = y_pred
    # submit['PassengerId'] = test_ID
    # submit['Survived'] = y_pred
    submit.to_csv("../input/submission.csv", index = False)
    

    
  

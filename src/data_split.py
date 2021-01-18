#Splits dataset into train and validation sets
#Validation will be use for final model evaluation

import pandas as pd
import config
from sklearn.model_selection import train_test_split 

if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    target = config.TARGET
    seed = config.SEED
    # if kaggle is false create a test set and a val set

    y = df.pop(target)
    X = df
    

    if config.KAGGLE == False:
        #create a test and val set
        #Creates a test set
        X_t, X_test, y_t, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed)
        #Creates a validation set now
        X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size = 0.2, random_state = seed)
        X_train.to_csv("../input/X_train.csv", index  = False) 
        X_val.to_csv("../input/X_val.csv", index  = False)
        y_train.to_csv("../input/y_train.csv", index  = False)
        y_val.to_csv("../input/y_val.csv", index  = False) 
        X_test.to_csv("../input/X_test.csv", index  = False)
        y_test.to_csv("../input/y_test.csv", index  = False) 

    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = seed)
        X_train.to_csv("../input/X_train.csv", index  = False) 
        X_val.to_csv("../input/X_val.csv", index  = False)
        y_train.to_csv("../input/y_train.csv", index  = False)
        y_val.to_csv("../input/y_val.csv", index  = False) 


    


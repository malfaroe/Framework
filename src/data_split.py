#Splits dataset into train and validation sets
#Validation will be use for final model evaluation

import pandas as pd
import config
from sklearn.model_selection import train_test_split 


if __name__ == "__main__":
    df = pd.read_csv(config.DF, index_col = 0)
    target = config.TARGET
    seed = config.SEED
    # y = df.iloc[:, target]
    # X = df.drop(df.columns[target], axis=1)
    y = df.pop(target)
    X = df
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = seed)
    X_train.to_csv("../input/X_train.csv", index  = False) 
    X_val.to_csv("../input/X_val.csv", index  = False)
    y_train.to_csv("../input/y_train.csv", index  = False) 
    y_val.to_csv("../input/y_val.csv", index  = False) 

                
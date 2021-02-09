#Implementation of multiple regression using OLS (normal equation)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error 


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

#Matrix operations
from numpy.linalg import inv

import matplotlib.pyplot as plt


class OLS_Mult():
    def __init__(self):
        pass

    def fit(self, X,y):  
         #Computes vector of coefficients beta
        r = inv(np.dot(X.transpose(), X))
        m = np.dot(X.transpose(), y)
        beta = np.dot(r,m)
        return beta
    
    def predict(self, X):
        return np.dot(X, beta.transpose())

    def score(self, y_predict, y):
        return np.round(r2_score(y_predict, y),3)



if __name__ == "__main__":
    df = pd.read_csv("pez.csv")
    y = df.pop("Weight")
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,
     random_state = 42)
    ols = OLS_Mult()
    beta = ols.fit(X_train,y_train)
    y_pred = ols.predict(X_test)
    print("r2 score on test set:", ols.score(y_test,y_pred))
    rmse = mean_squared_error(y_test, y_pred, squared=True)
    print("RMSE on test set:", rmse)
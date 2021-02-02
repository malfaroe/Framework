import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error 


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


"""Implementation of OLS for simple regression"""


#Read data

class OLS():
    def __init__(self):
        pass

    #Fit
    #Computes b0 and b1 using OLS
    def fit(self, X,y):
        beta_1 = np.sum((X - np.mean(X))*(y - np.mean(y)))/np.sum((X - np.mean(X))**2)
        beta_0 = np.mean(y) - beta_1 * np.mean(X)

        return np.round(beta_0, 4), np.round(beta_1, 4)

    
    def predict(self, X):
        return (beta_0 + beta_1 * X)

    def score(self, y_predict, y):
        return np.round(r2_score(y_predict, y),3)



if __name__ == "__main__":
    df = pd.read_csv("slr.csv")
    print(df.shape)
    X = df["GPA"]
    y = df["SAT"]
    
    # X,y = make_regression(n_features= 1, n_samples=1000, n_informative= 1)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,
     random_state = 42)
    ols = OLS()
    beta_0, beta_1 = ols.fit(X_train,y_train)
    print(beta_0, beta_1)
    y_pred = ols.predict(X_test)
    print(ols.score(y_test,y_pred))
    rmse = mean_squared_error(y_test, y_pred, squared=True)
    print(rmse)
    #Visualizing full data
    plt.scatter(X,y,c="red",lw=0.5)
    plt.title(y.name + " Vs " + X.name +"(Full dataset)")
    plt.xlabel(X.name)
    plt.ylabel(y.name)
    plt.show()
    #visualising the traing set results
    plt.scatter(X_train,y_train,c="red",lw=0.5)
    plt.plot(X_train,ols.predict(X_train),c="blue",lw=0.5)
    plt.title(y.name + " Vs " + X.name +"(Training set)")
    plt.xlabel(X.name)
    plt.ylabel(y.name)
    plt.show()
    #visualising the test set results
    plt.scatter(X_test,y_test,c="red",lw=0.5)
    plt.plot(X_test,ols.predict(X_test),c="blue",lw=0.5)
    plt.title(y.name + " Vs " + X.name +"(Test set)")
    plt.xlabel(X.name)
    plt.ylabel(y.name)
    plt.show()
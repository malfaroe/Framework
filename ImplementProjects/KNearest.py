import pandas as pd
import numpy as np
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error 


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

""" Classifier implementing the K-Nearest neighbors vote

Operation:
- Use the predict method for predicting  the class of a X_test set, taking
the entire X_train set and k number of neighbors:
Example: y_predict = predict(X_train, X_test, target, k) returns an array of predictions

- Use the scorer method for getting a score of the accuracy for the predicted values:
Exammple: score = scorer(y_pred, y), where y_pred is the predicted values from
the predict method and y is the vector of observed values 

- Additionally: use the k_optimal method for getting the best k neighbors 
that optimize the score

Note: Distance measures are strongly affected by the scale of the input data.
Experiment with standardization and other data preparation methods in order to improve results."""

class KNearestClassifier():
    def __init__(self):
        pass
    
    def euclidean_distance(self, x1, x2): #distance between two vectors
        x1 = np.array(x1)
        x2 = np.array(x2)
        return np.sqrt(np.sum((x2 - x1)**2, axis = 1).astype("float"))

    def k_calc(self, df, x, target, k):#
        df["euc"] = self.euclidean_distance(df.iloc[:, df.columns!= target], x)
        k_values = list(df.sort_values(by = "euc", ascending = True)[:k][target].values) #Get the k nearest neighbors
        y_k = max(k_values)
        df.drop("euc", axis = 1, inplace = True)
        return y_k

    def predict(self, df, x, target, k):
        y_predict = [self.k_calc(df, x.iloc[i,:], target, k) for i in range(len(x))]
        return np.array(y_predict)

    def scorer(self, y_pred, y):
            return accuracy_score(y_pred, y)

    def k_optimal(self, train, X_test, y_test, target):
        scores = []
        for i in range(1, len(X_test)):
            y_pred = self.predict(train, X_test, target, k = i)
            scores.append(self.scorer(y_test, y_pred))
        maximo = (0,0)
        for k, sc in enumerate(scores):
            if sc > maximo[1]:
                maximo = (k+1, sc)
        return maximo


class KNearestRegressor():
    def __init__(self):
        pass
    
    def euclidean_distance(self, x1, x2): #distance between two vectors
        x1 = np.array(x1)
        x2 = np.array(x2)
        return np.sqrt(np.sum((x2 - x1)**2, axis = 1).astype("float"))

    def k_calc(self, df, x, target, k):
        df["euc"] = self.euclidean_distance(df.iloc[:, df.columns!= target], x)
        k_values = list(df.sort_values(by = "euc", ascending = True)[:k][target].values)#Get the k nearest neighbors
        y_k = np.mean(k_values)
        df.drop("euc", axis = 1, inplace = True)
        return y_k

    def predict(self, df, x, target, k):
        y_predict = [self.k_calc(df, x.iloc[i,:], target, k) for i in range(len(x))]
        return np.array(y_predict)

    def scorer(self, y_pred, y):
            return mean_squared_error(y_pred, y)

    def k_optimal(self, train, X_test, y_test, target):
        scores = []
        for i in range(1, len(X_test)):
            y_pred = self.predict(train, X_test, target, k = i)
            scores.append(self.scorer(y_test, y_pred))
        maximo = (0,0)
        for k, sc in enumerate(scores):
            if sc > maximo[1]:
                maximo = (k+1, sc)
        return maximo
    
    
# Testing
if __name__ == "__main__":
    df = pd.read_csv("abalone_m.csv")
    target = "Rings"
    seed = 42
    y = df.pop(target)
    X = df
    #Rescaling
    scaler = StandardScaler()
    XRescaled = scaler.fit_transform(X)
    X = pd.DataFrame(XRescaled, columns = X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    train = pd.concat((X_train, y_train), axis =1)
    test = pd.concat((X_test, y_test), axis =1)
    kn = KNearest()
    k_opt = kn.k_optimal(train, X_test,y_test, target)[0]
    y_pred = kn.predict(train, X_test, target, k = k_opt)
    print("Score:", kn.scorer(y_test, y_pred))
    print("Optimal k:", k_opt)

    
"""Next devs:
0. Why k = 1 is always the optimal???
1. Benchmark testing of classifier with sklearn module
2. KNearest Regressor
3. Plots of clustered classes?"""
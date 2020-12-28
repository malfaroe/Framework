import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
import plotly.graph_objs as go
import plotly.offline as py
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score 


from sklearn.feature_selection import VarianceThreshold



class Selector:
    def __init__(self):
        # self.target = target
        # self.df = df
        pass
        
    def rescale(self, df, target):
        #Rescale data if necessary
        X2 = df.copy()
        y2 = X2.pop(target)
        scaler = StandardScaler().fit(X2)
        XRescaled = scaler.transform(X2)
        X_rescaled = pd.DataFrame(XRescaled, columns = X2.columns)
        return pd.concat((y2, X_rescaled), axis = 1)

    def variance_selector(self, df, target):
        #Rescale data if necessary
        print("Variance Threshold feature selection:")
        print("")
        X2 = df.copy()
        y2 = X2.pop(target)
        print("Initial feats:", X2.shape[1])
        low = [col for col in X2.columns if X2[col].std() < 0.5]
        print("Estos son antes de escale:", low)
        scaler = StandardScaler().fit(X2)
        XRescaled = scaler.transform(X2)
        X_rescaled = pd.DataFrame(XRescaled, columns = X2.columns)
        low = [col for col in X_rescaled.columns if X_rescaled[col].std() < 0.5]
        print("Estos son despues de escale:", low)
        #Analysis of amount of variation and droping all features with low variance
        var_tresh = VarianceThreshold(threshold = 0.5)
        var_tresh.fit_transform(X_rescaled)
        data_transformed = X2.loc[:, var_tresh.get_support()]
        print("Selected feats:", data_transformed.columns)
        print("Removed features:", set(X2.columns) - set(data_transformed.columns))
        print("Final features:", data_transformed.shape[1])
        print("{} features with low variance removed".format(X2.shape[1] - data_transformed.shape[1]))
        #Rejoin
        df = pd.concat((y2, data_transformed), axis = 1)
        return df

    def calcDrop(self, res):
        # All variables with correlation > cutoff
        all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))
        
        # All unique variables in drop column
        poss_drop = list(set(res['drop'].tolist()))

        # Keep any variable not in drop column
        keep = list(set(all_corr_vars).difference(set(poss_drop)))
        
        # Drop any variables in same row as a keep variable
        p = res[ res['v1'].isin(keep)  | res['v2'].isin(keep) ][['v1', 'v2']]
        q = list(set(p['v1'].tolist() + p['v2'].tolist()))
        drop = (list(set(q).difference(set(keep))))

        # Remove drop variables from possible drop 
        poss_drop = list(set(poss_drop).difference(set(drop)))
        
        # subset res dataframe to include possible drop pairs
        m = res[ res['v1'].isin(poss_drop)  | res['v2'].isin(poss_drop) ][['v1', 'v2','drop']]
            
        # remove rows that are decided (drop), take set and add to drops
        more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
        for item in more_drop:
            drop.append(item)
            
        return drop

    def corrX_new(self, df, cut) :
        
        # Get correlation matrix and upper triagle
        corr_mtx = df.corr().abs()
        avg_corr = corr_mtx.mean(axis = 1)
        up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(np.bool))

        dropcols = list()

        res = pd.DataFrame(columns=(['v1', 'v2', 'v1.target', 
                                        'v2.target','corr', 'drop' ]))

        for row in range(len(up)-1):
            col_idx = row + 1
            for col in range (col_idx, len(up)):
                if(corr_mtx.iloc[row, col] > cut):
                    if(avg_corr.iloc[row] > avg_corr.iloc[col]): 
                        dropcols.append(row)
                        drop = corr_mtx.columns[row]
                    else: 
                        dropcols.append(col)
                        drop = corr_mtx.columns[col]
                    
                    s = pd.Series([ corr_mtx.index[row],
                    up.columns[col],
                    avg_corr[row],
                    avg_corr[col],
                    up.iloc[row,col],
                    drop],
                    index = res.columns)
            
                    res = res.append(s, ignore_index = True)

        dropcols_names = self.calcDrop(res)
        print("{} features removed".format(len(dropcols_names)))
        print("Features removed:", dropcols_names)
        print("Selected features:", df.shape[1] - len(dropcols_names))
        selected_cols = set(df.columns) - set(dropcols_names)
        print("Selected:", selected_cols)
        return df[selected_cols]

    def corr_target(self, df):
        #Removes all feats with correlation under threshold
        remove_features = [feat for feat in df.columns if df.corr().abs()[["Survived"]].loc[feat, :][0] < 0.05]
        selected_cols = set(df.columns) - set(remove_features)
        df_test_2 = df[selected_cols]
        selected_cols


if __name__ == "__main__":
    df = pd.read_csv("../input/krt.csv")
    print(df.head(2))
    slc = Selector()
    df_r = slc.rescale(df, target = "Survived")
    df_v = slc.variance_selector(df_r, target = "Survived")
    df_corr_f = slc.corrX_new(df_v, cut = 0.65)
    df_corr_target = slc.corr_target(df_corr_f)



        
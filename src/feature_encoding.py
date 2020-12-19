#Encodes categorical features using get dummies
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


from sklearn.compose import make_column_transformer

from itertools import combinations 

import config 



df = pd.read_csv("../input/data_feat_gen.csv")
print(df.columns)
print("Initial shape:", df.shape)

#Selecciona todas las cols excepto target
df_sel = df.loc[:, df.columns != config.TARGET]

cat_feats = df_sel.select_dtypes(include= object).columns
num_feats = df_sel.select_dtypes(exclude= object).columns


# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(df_sel[cat_feats]).toarray())

# Rejoin
df_sel = df_sel[num_feats].join(enc_df)



#Get back together with target
df = pd.concat((df[config.TARGET], df_sel), axis = 1)
print("Processed shape:", df.shape)
#Save the data with new features
if config.KAGGLE == True:
    df.to_csv("../input/data_final.csv", index  = False)
    train = df[df[config.TARGET] != -1]
    print("Train columns:", train.columns)
    train.to_csv("../input/train_final.csv",
    index  = False)
    test = df[df[config.TARGET] == -1]
    test.drop(config.TARGET, axis = 1, inplace = True)
    test.to_csv("../input/test_final.csv",
    index  = False)

else:
    df[df[config.TARGET != -1]].to_csv("../input/train_final.csv",
    index  = False)

print(test.columns)
#Si se trata de una competencia de Kaggle genera train y test por separado
#Si no, solamente genera el train procesado


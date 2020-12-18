#Encodes categorical features using get dummies
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


from sklearn.compose import make_column_transformer

from itertools import combinations 

import config 



df = pd.read_csv("../input/data_feat_gen.csv")
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


print("Processed shape:", df.shape)

#Get back together with target
df = pd.concat((df[config.TARGET], df_sel), axis = 1)

#Save the data with new features
df.to_csv("../input/data_final.csv", index  = False)




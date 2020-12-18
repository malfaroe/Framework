#Testing a massive feature generator both for num and cat

import pandas as pd 
import numpy as np 
import config

from itertools import combinations 

#Input data structure: train+test+target
#Processing all but target

df = pd.read_csv(config.HYPER_FILE)
#Selecciona todas las cols excepto target
df_sel = df.loc[:, df.columns != config.TARGET]
print("df_sel columns:",df_sel.columns)
print(config.TARGET)

#Splitting cat and nums
cat_feats = df_sel.select_dtypes(include= object).columns
num_feats = df_sel.select_dtypes(exclude = object).columns


print("Initial features", df_sel.shape[1])
print("Cat feats:", cat_feats)
print("Num feats:", num_feats)


#Categorical processing
# Crear combinaciones de categoricals
pairs = list(combinations(cat_feats, 2))
for pair in pairs:
    df_sel[pair[0] + "_" + pair[1]] = df_sel[pair[0]].astype(str)+ "_" 
    + df_sel[pair[1]].astype(str)


print("Total new features:", df_sel.shape[1])
print(df_sel.columns)



#Numerical features
#binning
#polynomial


#Get back together with target
df = pd.concat((df[config.TARGET], df_sel), axis = 1)

#Save the data with new features
df.to_csv("../input/data_feat_gen.csv", index  = False)
print("Final features and target:", df.columns)



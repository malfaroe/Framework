#Testing a massive feature generator both for num and cat

import pandas as pd 
import numpy as np 
import config

from itertools import combinations 
from sklearn.preprocessing import PolynomialFeatures


#Input data structure: train+test+target
#Processing all but target

df = pd.read_csv(config.INPUT_FILE, index_col= 0)
print("Entry file:", df.columns)
#Selecciona todas las cols excepto target
df_sel = df.loc[:, df.columns != config.TARGET]
print("df_sel columns:",df_sel.columns)

#Splitting cat and nums
cat_feats = df_sel.select_dtypes(include= object).columns
num_feats = df_sel.select_dtypes(exclude = object).columns


print("Initial features:", df_sel.shape[1])
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



#PART 2: #Create new numerical feats: binning, polynomial feats

#binning
#polynomial

#4. Polynomial regressor of order 3 with ConstructArea (1,a,a2,a3)
print("Polynomial regressor of order 2:")
poly_2 = PolynomialFeatures(degree=2, interaction_only=False,
include_bias=False) #instanciamos
X_cubic = cubic.fit_transform(x_train[:,0].reshape(-1,1)) # se crean todos los features nuevo


#Get back together with target
df = pd.concat((df[config.TARGET], df_sel), axis = 1)

#Save the data with new features
df.to_csv("../input/data_feat_gen.csv", index  = False)
print("Final features:", df.shape[1])
print("Final features and target:", df.columns)



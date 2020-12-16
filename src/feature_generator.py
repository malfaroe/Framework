#Testing a massive feature generator

import pandas as pd 
import numpy as np 

from itertools import combinations 



df = pd.read_csv("../input/train_cat.csv")
# df = df.iloc[:1000, :] # recorto

cat_feats = df.select_dtypes(include= object).columns

#Crear features que cuentan cuantas ocurrencias
#hay de cada id por clase dentro del feat

for col in cat_feats:
    df["count" +"_"+ col] = df.groupby([col])["id"].transform("count")

#Crear combinaciones de categoricals
pairs = list(combinations(cat_feats, 2))
for pair in pairs:
    df[pair[0] + "_" + pair[1]] = df[pair[0]].astype(str)+ "_" 
    + df[pair[1]].astype(str)


print("Generated features:", df.shape[1])

#Save the data with new features
df.to_csv("../input/data_feat_gen.csv", index  = False)



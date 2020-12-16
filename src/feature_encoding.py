#Encodes categorical features using get dummies
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


from sklearn.compose import make_column_transformer

from itertools import combinations 



df = pd.read_csv("../input/data_feat_gen.csv")
df = df.iloc[:100, :]
print("Initial shape:", df.shape)
cat_feats = df.select_dtypes(include= object).columns
num_feats = df.select_dtypes(exclude= object).columns


# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(df [cat_feats]).toarray())

# Rejoin
df = df[num_feats].join(enc_df)


print("Processed shape:", df.shape)
#Save
df.to_csv("../input/catdat.csv", index  = False)


#sequence
import data_split
import baseline

import subprocess
import pandas as pd

from utils import Rescaler

# subprocess.run("python feature_generator.py & feature_encoding.py", shell=True)
# subprocess.run("python feature_importance.py", shell=True)

# subprocess.run("python data_split.py & python baseline.py", shell=True)
# subprocess.run("python baseline_ensemble.py & python model_tuning.py", shell=True)
# subprocess.run("python validation.py", shell=True)
df = pd.read_csv("../input/krt.csv")
df_2 = Rescaler(df, target = "Survived")
print("Data has been rescaled...")


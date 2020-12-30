#sequence
import data_split
import baseline

import subprocess
import pandas as pd

from utils import Rescaler

subprocess.run("python feature_generator.py & feature_encoding.py", shell=True)
subprocess.run("python feature_selection.py", shell=True)

subprocess.run("python data_split.py & python baseline_unified.py", shell=True)
subprocess.run("python python model_tuning.py & ensembles.py", shell=True)
subprocess.run("python validation.py", shell=True)



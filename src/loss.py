#sequence
import data_split
import baseline

import subprocess

subprocess.run("python feature_generator.py & feature_encoding.py", shell=True)
subprocess.run("python data_split.py & python baseline.py", shell=True)
subprocess.run("python baseline_ensemble.py & python model_tuning.py", shell=True)
subprocess.run("python validation.py", shell=True)
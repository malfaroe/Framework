import pandas as pd
import config

test = pd.read_csv(config.TEST_FILE)
train = pd.read_csv(config.TRAINING_FILE)
print(test.columns)
print( train.columns)
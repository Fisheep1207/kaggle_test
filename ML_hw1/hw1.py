import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_raw_data = pd.read_csv("data/convid.train.csv")
train_raw_data.info()
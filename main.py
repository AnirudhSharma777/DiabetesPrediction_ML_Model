#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import dtale as dt


dataSet = pd.read_csv("diabetes.csv")
print(dataSet.head())
dt.show(dataSet)

#%%
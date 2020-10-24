import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
pd.options.mode.chained_assignment = None

seed = 12345
np.random.seed(seed)

df = pd.read_csv('atlas-higgs-challenge-2014-v2.csv')

#EventId column is useless because pandas.dataframe has a default index 
df.drop('EventId', axis=1, inplace=True)

Epoch_Value = 40

feature_list = df.columns.tolist()
feature_list.remove('Weight')
feature_list.remove('Label')
feature_list.remove('KaggleSet')
feature_list.remove('KaggleWeight')
feature_list.remove('PRI_jet_num')


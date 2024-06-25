import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import PowerTransformer

from functions import *


# 読み込み数
TestSize = 0.5
r = 6378.137


def cycle(number, x_test, y_test, train_df):
    start_time = time.perf_counter()
    # データ読み込み
    print('-----------')
    print("loading data")
    df_init = train_df.iloc[:number, :]
    df_init = df_init.dropna(how='any')
    df_init = df_init.drop(['key'],axis=1)

    print("Preprocessing...")
    df_init = add_datetime_info(df_init)
    df_init = calc_distance(df_init, r)
    df_init = df_init[df_init.distance > 0]
    df_init = apply_boxcox_transform_sklearn(df_init)
    df_init = cleanup(df_init)

    x_train0 = df_init.iloc[:, 1:13]
    y_train0 = df_init.iloc[:, 0]

    print(number)
    print(len(df_init))
    pred(x_train0, x_test, y_train0, y_test)

    end_time = time.perf_counter()
    print("time: %.3f s" % (end_time - start_time))

    gc.collect()



# データ読み込み
print("loading data")
df = pd.read_csv('train.csv')
df = df.sample(n=521000, random_state=777)
df0 = df.iloc[20001:, :]
df = df.iloc[:20000, :]
df = df.dropna(how='any')
df = df.drop(['key'],axis=1)

df = add_datetime_info(df)
df = calc_distance(df, r)
df = df[df.distance > 0]
df = apply_boxcox_transform_sklearn(df)
df = cleanup(df)

x_train, x, y_train, y = train_test_split(df.iloc[:, 1:13], df.iloc[:, 0],test_size=TestSize, random_state=777)

gc.collect()

for i in (5000, 10000, 50000, 100000, 200000, 300000, 400000, 500000):
    cycle(i, x, y, df0)


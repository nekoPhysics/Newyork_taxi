import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
import optuna
import lightgbm as lgb
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer


# 日時の情報を付加
def add_datetime_info(dataset):
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'], format="%Y-%m-%d %H:%M:%S UTC")
    
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['weekday'] = dataset.pickup_datetime.dt.weekday
    dataset['year'] = dataset.pickup_datetime.dt.year
    return dataset


def datetime_Trigonometric_encoder(dataset, col):
    dataset[col + '_cos'] = np.cos(2 * np.pi * dataset[col] / (dataset[col].max() + 1 ))
    dataset[col + '_sin'] = np.sin(2 * np.pi * dataset[col] / (dataset[col].max() + 1 ))
    return dataset


def datetime_maker(dataset):
    dataset = datetime_Trigonometric_encoder(dataset, "hour")
    dataset = datetime_Trigonometric_encoder(dataset, "day")
    dataset = datetime_Trigonometric_encoder(dataset, 'weekday')
    dataset = datetime_Trigonometric_encoder(dataset, "month")
    return dataset


# 直線距離の情報を付加
def calc_distance(dataset, r):
    dataset['distance'] = r * np.arccos((np.sin(dataset.pickup_latitude * np.pi/180) * np.sin(dataset.dropoff_latitude * np.pi/180))
                                         + (np.cos(dataset.pickup_latitude * np.pi/180) * np.cos(dataset.dropoff_latitude * np.pi/180) * np.cos((dataset.dropoff_longitude * np.pi/180) - (dataset.pickup_longitude * np.pi/180))))
    dataset.loc[(dataset['pickup_latitude'] == dataset['dropoff_latitude'])&(dataset['pickup_longitude'] == dataset['dropoff_longitude']), 'distance'] = 0.000095
    
    dataset['distance'] = list(dataset['distance'])

    return dataset


def is_airport(df):
    #KJFK 40.639901, -73.806465 ~ 40.660645, -73.777591
    #KEWR 40.687314, -74.187967 ~ 40.697815, -74.176204
    df['is_airport'] = False
    df.loc[(df.pickup_latitude >= 40.639901) & (df.pickup_latitude <= 40.660645) & (df.pickup_longitude >= -73.806465) & (df.pickup_longitude <= -73.777591),'is_airport'] = True
    df.loc[(df.dropoff_latitude >= 40.639901) & (df.dropoff_latitude <= 40.660645) & (df.dropoff_longitude >= -73.806465) & (df.dropoff_longitude <= -73.777591),'is_airport'] = True
    df.loc[(df.pickup_latitude >= 40.687314) & (df.pickup_latitude <= 40.697815) & (df.pickup_longitude >= -74.187967) & (df.pickup_longitude <= -74.176204),'is_airport'] = True
    df.loc[(df.dropoff_latitude >= 40.687314) & (df.dropoff_latitude <= 40.697815) & (df.dropoff_longitude >= -74.187967) & (df.dropoff_longitude <= -74.176204),'is_airport'] = True
    return df


# box-cox変換
def apply_boxcox_transform_sklearn(df):
    # データフレームから 'fare_amount' と 'distance' カラムを抽出
    # fare_amount_data = df['fare_amount'].values.reshape(-1, 1)
    distance_data = df['distance'].values.reshape(-1, 1)
    
    # PowerTransformer を使用して Box-Cox 変換を適用
    transformer = PowerTransformer(method="box-cox", standardize=False)  # standardize=False はデータのスケーリングを無効にします
    # fare_amount_transformed = transformer.fit_transform(fare_amount_data)
    distance_transformed = transformer.fit_transform(distance_data)
    
    # 変換後のデータをデータフレームに戻す
    # df['fare_amount'] = fare_amount_transformed
    df['distance'] = distance_transformed
    
    return df


# 欠損値の除去
def cleanup(df):
    return df[(df.fare_amount > 0) & 
              (df.distance > 0) &
              (df.passenger_count > 0) & 
              (df.passenger_count < 6) & 
              (df.pickup_longitude > -75) &
              (df.pickup_longitude < -72) &
              (df.pickup_latitude > 40) &
              (df.pickup_latitude < 42) &
              (df.dropoff_longitude > -75) &
              (df.dropoff_longitude < -72) &
              (df.dropoff_latitude > 40) &
              (df.dropoff_latitude < 42)]


def dropout(df, Pvalue, lower_limit, upper_limit):
    df['zscore'] = 0
    for i in np.arange(0, 280, 0.1):
        if len(df[df.fare_amount == i]) <= 1:
            continue
        if i >= upper_limit or i <= lower_limit:
            continue

        mean = df[df.fare_amount == i].distance.mean()
        std = df[df.fare_amount == i].distance.std()

        df.loc[df['fare_amount'] == i, 'zscore'] = (df[df.fare_amount == i].distance - mean) / std

    return df[(df.zscore < Pvalue)].drop(['zscore'],axis=1)



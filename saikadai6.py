import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer

# 読み込み数
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)
number = 62500
TestSize = 0.2
r = 6378.137

Pvalue = 2.58
# 信頼区間-> Pvalue
# 90%    -> 1.65
# 95%    -> 1.96
# 99%    -> 2.58
# 99.73% -> 3

# dropoutを適用する賃金帯（境界含む）
lower_limit = 14.1
upper_limit = 60

# データ読み込み
print("loading data")
train_df = pd.read_csv('/Users/hironeko1234/Documents/プログラミング/最終課題1/new-york-city-taxi-fare-prediction 2/train.csv')
train_df = train_df.sample(n=number, random_state=777)
print("loading data")


# 日時の情報を付加
def add_datetime_info(dataset):
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
    
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['weekday'] = dataset.pickup_datetime.dt.weekday
    dataset['year'] = dataset.pickup_datetime.dt.year
    
    return dataset.drop(['pickup_datetime'],axis=1)


train_df = add_datetime_info(train_df)

# 直線距離の情報を付加
def calc_distance(dataset, r):
    dataset['distance'] = r * np.arccos((np.sin(dataset.pickup_latitude * np.pi/180) * np.sin(dataset.dropoff_latitude * np.pi/180))
                                         + (np.cos(dataset.pickup_latitude * np.pi/180) * np.cos(dataset.dropoff_latitude * np.pi/180) * np.cos((dataset.dropoff_longitude * np.pi/180) - (dataset.pickup_longitude * np.pi/180))))
    dataset.loc[(dataset['pickup_latitude'] == dataset['dropoff_latitude'])&(dataset['pickup_longitude'] == dataset['dropoff_longitude']), 'distance'] = 0.000095
    
    dataset['distance'] = list(dataset['distance'])

    return dataset


train_df = calc_distance(train_df, 6378.137)


def is_airport(df):
    #KJFK 40.639901, -73.806465 ~ 40.660645, -73.777591
    #KEWR 40.687314, -74.187967 ~ 40.697815, -74.176204
    df['is_airport'] = False
    df.loc[(df.pickup_latitude >= 40.639901) & (df.pickup_latitude <= 40.660645) & (df.pickup_longitude >= -73.806465) & (df.pickup_longitude <= -73.777591),'is_airport'] = True
    df.loc[(df.dropoff_latitude >= 40.639901) & (df.dropoff_latitude <= 40.660645) & (df.dropoff_longitude >= -73.806465) & (df.dropoff_longitude <= -73.777591),'is_airport'] = True
    df.loc[(df.pickup_latitude >= 40.687314) & (df.pickup_latitude <= 40.697815) & (df.pickup_longitude >= -74.187967) & (df.pickup_longitude <= -74.176204),'is_airport'] = True
    df.loc[(df.dropoff_latitude >= 40.687314) & (df.dropoff_latitude <= 40.697815) & (df.dropoff_longitude >= -74.187967) & (df.dropoff_longitude <= -74.176204),'is_airport'] = True
    return df


train_df = is_airport(train_df)
train_df = train_df.dropna(how='any')
train_df = train_df.drop(['key'],axis=1)


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


train_df = apply_boxcox_transform_sklearn(train_df)

# 欠損値の除去
def cleanup(df):
    return df[(df.fare_amount > 0) & 
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


train_df = cleanup(train_df)


def pred(X, x, Y, y, model='RFR'):
    # モデルのトレーニング
    model = RFR(n_jobs=-1, random_state=777) # ランダムフォレスト

    model.fit(X, Y)

    # 回帰
    test_pred = model.predict(x)
    train_pred = model.predict(X)

    # 評価
    test_r2 = r2_score(y, test_pred)
    train_r2 = r2_score(Y, train_pred)

    test_mse = mean_squared_error(y, test_pred)
    train_mse = mean_squared_error(Y, train_pred)

    test_rmse = np.sqrt(mean_squared_error(y, test_pred))
    train_rmse = np.sqrt(mean_squared_error(Y, train_pred))

    print("R2     -> test : %.3f  train : %.3f" % (test_r2, train_r2))
    print("MSE   -> test : %.3f  train : %.3f" % (test_mse, train_mse))
    print("KAGGLE -> test : %.3f  train : %.3f" % (test_rmse, train_rmse))

    # 変数重要度
    feature_importances = pd.DataFrame({
        "features": x.columns,
        "importances": model.feature_importances_
    })
    print(feature_importances)


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


train_df = dropout(train_df, Pvalue, lower_limit, upper_limit)
train_df = is_airport(train_df)


def pred(x_train, x_test, y_train, y_test):
    # 学習データと評価データを作成

    # #データを標準化
    # sc = StandardScaler()
    # sc.fit(x_train) #学習用データで標準化
    # x_train_std = sc.transform(x_train)
    # x_test_std = sc.transform(x_test)

    # モデルのトレーニング
    # print("fitting...")

    model = RFR(n_jobs=-1, random_state=777) # ランダムフォレスト
    # model = GBR(random_state=777) # 勾配ブースティング木
    # model = MLPR(hidden_layer_sizes=(40,20,10,5), max_iter=1000, random_state=777, verbose=True) # ニューラルネットワーク

    # print(model)
    model.fit(x_train, y_train)
    # print('...complete')

    # 回帰　
    # print("predicting...")
    test_pred = model.predict(x_test)
    train_pred = model.predict(x_train)
    # print("...complete")

    # 評価
    # 決定係数(R2)
    test_r2 = r2_score(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)

    # 二乗平均平方根誤差 (RMSE)
    test_mae = mean_squared_error(y_test, test_pred)
    train_mae = mean_squared_error(y_train, train_pred)

    # 二乗平均平方根誤差 (RMSE)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

    print("R2     -> test : %.3f  train : %.3f" % (test_r2, train_r2))
    print("RMSE   -> test : %.3f  train : %.3f" % (test_mae, train_mae))
    print("KAGGLE -> test : %.3f  train : %.3f" % (test_rmse, train_rmse))

    # 変数重要度
    feature_importances = pd.DataFrame({
        "features": x_test.columns,
        "importances": model.feature_importances_
    })
    print(feature_importances)


x_train, x, y_train, y = train_test_split(train_df.iloc[:, 1:13], train_df.iloc[:, 0],test_size=TestSize, random_state=777)

pred(x_train, x, y_train, y)
import time
import gc
import datetime
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna.integration.lightgbm as lgb_tune

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
import optuna
import lightgbm as lgb
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score



# 読み込み数
number = 900000
r = 6378.137
Pvalue = 3
lower_limit =14.1
upper_limit = 60

# データ読み込み
train_df = pd.read_csv('new-york-city-taxi-fare-prediction 2/train.csv')
train_df = train_df.sample(n=number, random_state=777)
test_df = pd.read_csv('/Users/hironeko1234/Documents/プログラミング/最終課題1/new-york-city-taxi-fare-prediction 2/test.csv')
# test_df = train_df.iloc[50000:, :]
# train_df = train_df.iloc[:50000, :]
train_df = train_df.dropna(how='any')
df_imp = train_df
df_imp = df_imp.dropna(how='any')

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
    dataset[col + '_cos'] = np.cos(2 * np.pi * dataset[col] / dataset[col].max())
    dataset[col + '_sin'] = np.sin(2 * np.pi * dataset[col] / dataset[col].max())
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

# 欠損値の除去
def cleanup2(df):
    return df[(df.distance > 0)]
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


df_imp = add_datetime_info(df_imp)
test_df = add_datetime_info(test_df)
df_imp['hour'] = df_imp['hour'].astype('int')
df_imp['day'] = df_imp['day'].astype('int')
df_imp['month'] = df_imp['month'].astype('int')
df_imp['weekday'] = df_imp['weekday'].astype('int')
df_imp['year'] = df_imp['year'].astype('int')
test_df['hour'] = test_df['hour'].astype('int')
test_df['day'] = test_df['day'].astype('int')
test_df['month'] = test_df['month'].astype('int')
test_df['weekday'] = test_df['weekday'].astype('int')
test_df['year'] = test_df['year'].astype('int')
test_key = test_df['key']
df_imp = df_imp.drop('key', axis=1)
test_df = test_df.drop('key', axis=1)
df_imp = df_imp.drop('pickup_datetime', axis=1)
test_df = test_df.drop('pickup_datetime', axis=1)
df_imp = calc_distance(df_imp, r)
test_df = calc_distance(test_df, r)
df_imp = cleanup(df_imp)
test_df = cleanup2(test_df)
df_imp = apply_boxcox_transform_sklearn(df_imp)
test_df = apply_boxcox_transform_sklearn(test_df)
df_imp = dropout(df_imp, Pvalue, lower_limit, upper_limit)
print(df_imp)
print(test_df)
# 学習データと評価データを作成
# x_train = df_imp.iloc[:50000, 1:13]
# x = df_imp.iloc[50000:, 1:13]
# y_train = df_imp.iloc[:50000, 0]
# y = df_imp.iloc[50000:, 0]
# x_sub = df_imp.iloc[:, 1:13]
train_feat_df=df_imp#train_df分の長さまで
test_feat_df=df_imp[-len(test_df):]#後ろから数えてtest_df分の長さを取り出し

#学習に使う列。数値しかない列を抽出
features = ['pickup_longitude', 
            'pickup_latitude', 
            'dropoff_longitude', 
            'dropoff_latitude', 
            'passenger_count',  
            'hour', 
            'day', 
            'month', 
            'weekday',
            'year',
            'distance'
    ]
#求めたい列
target = ["fare_amount"]

#学習で使う用に選定した列の値をトレーニング用のx,y テスト用のxに格納(pandasデータフレームからnumpyのndarray型に)
x_train_df = train_feat_df[features].values #train.csvの訓練用データの特徴量
y_train_df = train_feat_df[target].values #train.csvの訓練用データの目的変数price
x_test_df = test_feat_df[features].values #train.csvの検証用データの特徴量
y_test_df = test_feat_df[target].values #train.csvの検証用データの目的変数price
x_sub = test_df[features].values # test.csvの特徴量

train_set = lgb.Dataset(x_train_df, y_train_df, free_raw_data=False)
test_set = lgb.Dataset(x_test_df, y_test_df, free_raw_data=False)

params = {
        # 回帰問題
        'objective': 
        'regression', 
        'random_state': 777, 
        'boosting_type': 'gbdt', 
        'verbose': -1, 
        'feature_pre_filter': False, 
        'lambda_l1': 5.33826773600148e-06, 
        'lambda_l2': 1.3970683921150133e-08, 
        'num_leaves': 251, 
        'feature_fraction': 0.6, 
        'bagging_fraction': 1.0, 
        'bagging_freq': 0, 
        'min_child_samples': 100, 
        'num_iterations': 10000,
        # 学習用の指標 (RMSE)
        'metrics': 'rmse',
        'num_rounds':300,
        'early_stopping_round': 50,
    }



model = lgb.train(params, train_set=train_set, valid_sets=[test_set])
# fit_params = {
#         'eval_set': [(x_train, y_train)]
#         }
prediction = model.predict(x_sub)

# # GBDT
# dtrain = xgb.DMatrix(x_train, label=y_train)
# dtest = xgb.DMatrix(x, label=y)
# dsub = xgb.DMatrix(x_sub)

# xgb_params = {
#     "max_depth": 8,
#     "eta": 0.20714087656180552,
#     "subsample": 0.5065576674056198,
#     "colsample_bytree": 0.677669592992848,
#     "objective": "reg:squarederror",
#     "eval_metric": "rmse",
#     'seed': 777
# }

# evals = [(dtrain, 'train'), (dtest, 'eval')]
# model = xgb.train(xgb_params,
#                 dtrain,
#                 num_boost_round=1000,
#                 early_stopping_rounds=10,
#                 evals=evals,
#                 )

# RF
# model = RFR(n_estimators=400, random_state=777, n_jobs=-1, min_samples_leaf=8, max_features=5)
# model.fit(x_train, y_train)

# # # 回帰　
# test_pred = model.predict(x_train)
# train_pred = model.predict(x_sub)
# # pred_GBDT = GBDT.predict(x_test)

# # 回帰
# # print("predicting...")
# test_pred = model.predict(x_sub)
# train_pred = model.predict(x_train)
# # print("...complete")

# # 評価
# # 決定係数(R2)
# test_r2 = r2_score(y_sub, test_pred)
# train_r2 = r2_score(y_train, train_pred)

# # 二乗平均平方根誤差 (RMSE)
# test_mae = mean_squared_error(y_sub, test_pred)
# train_mae = mean_squared_error(y_train, train_pred)

# # 二乗平均平方根誤差 (RMSE)
# test_rmse = np.sqrt(mean_squared_error(y_sub, test_pred))
# train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

# print("R2     -> test : %.3f  train : %.3f" % (test_r2, train_r2))
# print("MSE   -> test : %.3f  train : %.3f" % (test_mae, train_mae))
# print("KAGGLE -> test : %.3f  train : %.3f" % (test_rmse, train_rmse))

# # 変数重要度
# feature_importances = pd.DataFrame({
#     "features": x_sub.columns,
#     "importances": model.feature_importances_
# })
# print(feature_importances)
submission = pd.DataFrame({
        "key": test_key,
        "fare_amount": prediction
})

submission.to_csv('submission.csv',index=False)


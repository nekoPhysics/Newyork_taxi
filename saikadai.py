import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as seb
import torch
import requests
import subprocess

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor as RFR


# 読み込み数
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)
number = 12500
TestSize = 0.2
r = 6378.137

# データ読み込み
print("loading data")
train_df = pd.read_csv('/Users/hironeko1234/Documents/プログラミング/最終課題1/new-york-city-taxi-fare-prediction 2/train.csv')
train_df = train_df.sample(n=number, random_state=777)
# test_df = pd.read_csv('new-york-city-taxi-fare-prediction 2/test.csv')


# 距離計算
def calc_distance(dataset):
    dataset['distance'] = r * np.arccos((np.sin(dataset.pickup_latitude * np.pi/180) * np.sin(dataset.dropoff_latitude * np.pi/180))
                                         + (np.cos(dataset.pickup_latitude * np.pi/180) * np.cos(dataset.dropoff_latitude * np.pi/180) * np.cos((dataset.dropoff_longitude * np.pi/180) - (dataset.pickup_longitude * np.pi/180))))
    dataset.loc[(dataset['pickup_latitude'] == dataset['dropoff_latitude'])&(dataset['pickup_longitude'] == dataset['dropoff_longitude']), 'distance'] = 0
    
    return dataset


print("add distance info")
train_df = calc_distance(train_df)
print(train_df)

# 欠損値の除去
def cleanup(df):
    return df[(df.fare_amount > 0) & (df.passenger_count > 0)]


# 欠損値の除去
# def cleanup(df):
#     return df[(df.fare_amount > 0) & (df.passenger_count > 0) & (df.passenger_count < 8)
#             & (df.pickup_longitude < -70) & (df.pickup_longitude > -80) & (df.pickup_latitude > 35) & (df.pickup_latitude < 45)
#             & (df.dropoff_longitude < -70) & (df.dropoff_longitude > -80) & (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) & (df.distance > 0)]

train_df = cleanup(train_df)
test_df = cleanup(train_df)


# テキストファイルへ保存
def save_output(df):
    file_path = "/Users/hironeko1234/Documents/プログラミング/最終課題1/statics.csv"
    with open(file_path, mode='a') as file:
        file.write('\n'.join(df))
    print("save complete")
    return()


train_descibe = train_df.describe()
print(train_descibe)
print(type(train_descibe))
train_descibe.to_csv('/Users/hironeko1234/Documents/プログラミング/最終課題1/statics.csv')
# save_output(train_descibe)

# print(test_df)
# データの列名を取得
# train_descibe = train_df.columns
# test_columns_list = test_df.columns
# save_output()
# print(train_descibe)
# print(test_columns_list)


# # 欠損値の確認
# defect_train = train_df.isnull().sum().sort_values(ascending=False)
# defect_test = test_df.isnull().sum().sort_values(ascending=False)
# # print(defect_train)
# # print(defect_test)




# a = train_df[train_df["dropoff_longitude"].isnull()|train_df["dropoff_latitude"].isnull()]
# print(a)
# defectD_train = train_df.drop(a.index,axis=0, inplace=True)
# train_df = train_df.isnull().sum().sort_values(ascending=False)
# print(train_df)

# 料金の頻度をヒストグラムで表示
# sns.histplot(data = train_df, x = "fare_amount", kde = True, bins = 100)
# plt.show()

# # 乗車人数をヒストグラムで表示
# sns.histplot(data = train_df, x = 'passenger_count')
# plt.xlim(0,10)
# plt.ylim(0,1000000)
# plt.show()

# 乗車日時をdatetime型に変換
# train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
# test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])

# # 緯度経度の散布図を表示
# print("DRAWING GRAPH 1...")
# # pickup_latitude (緯度)
# # pickup_longitude(経度)
# plt.scatter(train_df['pickup_longitude'], train_df['pickup_latitude'], c='g', alpha=0.1)
# plt.title("Scatter plot for (pickup)latitude and longitude")
# plt.xlabel("pickup_longitude")
# plt.ylabel("pickup_latitude")
# # train_df.plot.scatter(x='pickup_longitude', y='pickup_latitude', alpha=0.2)
# plt.savefig("map2.png")
# plt.show()

# print("DRAWING GRAPH 2...")
# # pickup_latitude (緯度)
# # pickup_longitude(経度)
# plt.scatter(train_df['dropoff_longitude'], train_df['dropoff_latitude'], c='r', alpha=0.1)
# plt.title("Scatter plot for (dropoff)latitude and longitude")
# plt.xlabel("dropoff_longitude")
# plt.ylabel("dropoff_latitude")
# # train_df.plot.scatter(x='pickup_longitude', y='pickup_latitude', alpha=0.2)
# plt.savefig("map2dropoff.png")
# plt.show()

# print("DRAWING GRAPH3...")
# plt.scatter(train_df['pickup_longitude'], train_df['pickup_latitude'], c='g', label="pickup", alpha=0.05)
# plt.scatter(train_df['dropoff_longitude'], train_df['dropoff_latitude'], c='b', label="dropoff", alpha=0.05)
# plt.title("Scatter plot for pickup and dropoff point (latitude,longitude)")
# plt.xlabel("longitude")
# plt.ylabel("latitude")
# plt.legend()
# plt.savefig("map_pic_drp.png")
# plt.show()

# 距離と料金の関係
print("DRAWING GRAPH4...")
plt.scatter(train_df['passenger_count'], train_df['fare_amount'], c='c', label="plot", alpha=0.05)
x = train_df['passenger_count']
y = train_df['fare_amount']
plt.title("Scatter plot of passengers and fare_amount")
plt.xlabel("passenger_count")
plt.ylabel("fare_amount")
# slope, intercept = np.polyfit(x, y, 1)
# plt.plot(x, slope(x), c="r", label="Ism")
# equation_text = f'y = {slope:.2f}x + {intercept:.2f}'
# plt.text(2, 25, equation_text, fontsize=12, color='r')
# plt.legend()
plt.savefig("Scatter plot for Scatter plot of passengers and fare_amount_2")
plt.show()

# plt.title("Scatter plot for distance and fare")
# plt.xlabel("distance")
# plt.ylabel("fare")
# slope, intercept = np.polyfit(x, y, 1)
# plt.plot(x, slope * x + intercept, c="r", label="Ism")  # 直線式を作成してプロット
# equation_text = f'y = {slope:.2f}x + {intercept:.2f}'
# plt.text(2, 25, equation_text, fontsize=12, color='r')
# plt.legend()
# plt.savefig("Scatter plot for distance and fare.png")
# plt.show()

# 人数と料金に相関があるかを調査
# print("DRAWING GRAPH 2...")
# plt.bar(train_df['passenger_count'], train_df['fare_amount'], alpha=0.2)
# # train_df.plot.scatter(x='passenger_count', y='fare_amount', alpha=0.2)
# plt.savefig("train_passandfare2.png")
# plt.show()
# plt.bar(train_df['passenger_count'], train_df['fare_amount'], alpha=0.2)
# # train_df.plot.scatter(x='pickup_latitude', y='pickup_longitude', alpha=0.2)
# plt.savefig("test_passandfare.png")
# plt.show()


# Data columns (total 8 columns):
#  #   Column             Dtype  
# ---  ------             -----  
#  0   key                object 
#  1   fare_amount        float64
#  2   pickup_datetime    object 
#  3   pickup_longitude   float64
#  4   pickup_latitude    float64
#  5   dropoff_longitude  float64
#  6   dropoff_latitude   float64
#  7   passenger_count    int64  



# def add_datetime_info(dataset=0):
#     dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
    
#     dataset['hour'] = dataset.pickup_datetime.dt.hour
#     dataset['day'] = dataset.pickup_datetime.dt.day
#     dataset['month'] = dataset.pickup_datetime.dt.month
#     dataset['weekday'] = dataset.pickup_datetime.dt.weekday
#     dataset['year'] = dataset.pickup_datetime.dt.year
    
#     return dataset.drop(['pickup_datetime'],axis=1)

# train_df = add_datetime_info(train_df)
# print(train_df)

# print(train_df.info())
# # 学習データと評価データを作成
# x_train, x_test, y_train, y_test = train_test_split(train_df.iloc[:, 2:12], train_df.iloc[:, 1],test_size=TestSize, random_state=777)

# print(x_train.info())
# print(x_test)
# print(y_train.info())
# print(x_test)

# 距離計算
# def calc_distance(dataset):
#     dataset['distance'] = r * np.arccos((np.sin(dataset.pickup_latitude * np.pi/180) * np.sin(dataset.dropoff_latitude * np.pi/180))
#                                          + (np.cos(dataset.pickup_latitude * np.pi/180) * np.cos(dataset.dropoff_latitude * np.pi/180) * np.cos((dataset.dropoff_longitude * np.pi/180) - (dataset.pickup_longitude * np.pi/180))))
#     dataset.loc[(dataset['pickup_latitude'] == dataset['dropoff_latitude'])&(dataset['pickup_longitude'] == dataset['dropoff_longitude']), 'distance'] = 0
    
#     return dataset

# print(train_df)
# print("add distance info")
# train_df = calc_distance(train_df)


# def rush_hour(dataset):
#     rush_time = []
#     for h in dataset['hour']:
#         if 16 <= h <= 20:
#             rush_time.append(1)
#         else:
#             rush_time.append(0)
    
#     dataset['rush_time'] = rush_time
#     return dataset

# print("add rush_hour info")
# train_df = rush_hour(train_df)
# print(train_df)
# # 学習データと評価データを作成
# print("spliting...")
# x_train, x_test, y_train, y_test = train_test_split(train_df.iloc[:, 2:14], train_df.iloc[:, 1],test_size=TestSize, random_state=777)
# print('...complete')


# #データを標準化
# sc = StandardScaler()
# sc.fit(x_train) #学習用データで標準化
# x_train_std = sc.transform(x_train)
# x_test_std = sc.transform(x_test)

# # モデルの学習(勾配ブースティング)
# model = GradientBoostingRegressor()
# model.fit(x_train, y_train)

# # 回帰　
# pred_GBDT = GBDT.predict(x_test)

# # 評価
# # 決定係数(R2)
# r2_GBDT = r2_score(y_test, pred_GBDT)

# # 平均絶対誤差(MAE)
# mae_GBDT = mean_absolute_error(y_test, pred_GBDT)

# # 二乗平均平方根誤差 (RMSE)
# rmse = mean_squared_error(y_test, pred_GBDT)

# print("R2 : %.3f" % r2_GBDT)
# print("MAE : %.3f" % mae_GBDT)
# print("RMSE : %.3f"% rmse)

# # 変数重要度
# print("feature_importances = ", GBDT.feature_importances_)

# # ランダムフォレスト
# model = RFR(n_jobs=-1, max_features=6, random_state=777)
# model.fit(x_train, y_train)
# pred_model = model.predict(x_test)


# # 回帰　
# pred_GBDT = model.predict(x_test)
# train_pred_GBDT = model.predict(x_train)

# # 評価
# # 決定係数(R2)
# r2_GBDT = r2_score(y_test, pred_GBDT)
# output_r2_GBDT = "R2 : %.3f" % r2_GBDT

# # 平均絶対誤差(MAE)
# mae_GBDT = mean_absolute_error(y_test, pred_GBDT)
# output_mae_GBDT = "MAE : %.3f" % mae_GBDT
# # 二乗平均平方根誤差 (RMSE)
# rmse = mean_squared_error(y_test, pred_GBDT)
# output_rmse = "RMSE : %.3f"% rmse

# print("R2 : %.3f" % r2_GBDT)
# print("MAE : %.3f" % mae_GBDT)
# print("RMSE : %.3f"% rmse)

# output_text = output_r2_GBDT, output_mae_GBDT, output_rmse

# # 通知
# def send_message():
#     headers = {
#       'Authorization': 'vgbZVqyHclpUSdYXziaBveLmxIA5PhhCdNIMlJU5r5S',
#     }

#     files = {
#       'message': (None, output_text),
#     }

#     requests.post('https://notify-api.line.me/api/notify', headers=headers, files=files)
#     return()

# send_message()

# # テキストファイルへ保存
# def save_output():
#     file_path = "/Users/hironeko1234/Documents/プログラミング/最終課題1/statics.txt"
#     with open(file_path, mode='a') as file:
#         file.write('\n'.join(output_text))
#     print("save complete")
#     return()

# save_output()

# # 変数重要度
# # print("feature_importances = ", model.feature_importances_)
# feature_importances = pd.DataFrame({
#     "features": x_test.columns,
#     "importances": model.feature_importances_
# })
# print(feature_importances)



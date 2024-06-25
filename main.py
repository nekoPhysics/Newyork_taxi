import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor as RFR


# 読み込み数
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)
number = 125000
TestSize = 0.2
r = 6378.137

# データ読み込み
print("loading data")
train_df = pd.read_csv('train.csv')
train_df = train_df.sample(n=number, random_state=777)
test_df = pd.read_csv('test.csv')

# print(train_df)
# print(test_df)

# データの列名を取得
# train_columns_list = train_df.columns
# test_columns_list = test_df.columns

# print(train_columns_list)
# print(test_columns_list)


# # 欠損値の確認
# defect_train = train_df.isnull().sum().sort_values(ascending=False)
# defect_test = test_df.isnull().sum().sort_values(ascending=False)
# # print(defect_train)
# # print(defect_test)

# 欠損値の除去

def cleanup(df):
    return df[(df.fare_amount > 0) & (df.passenger_count > 0) & (df.passenger_count < 7)
            & (df.pickup_longitude < -70) & (df.pickup_longitude > -80) & (df.pickup_latitude > 35) & (df.pickup_latitude < 45)
            & (df.dropoff_longitude < -70) & (df.dropoff_longitude > -80) & (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45)]

print("cleaning data")
train_df = cleanup(train_df)
test_df = cleanup(train_df)

# a = train_df[train_df["dropoff_longitude"].isnull()|train_df["dropoff_latitude"].isnull()]
# print(a)
# defectD_train = train_df.drop(a.index,axis=0, inplace=True)
# train_df = train_df.isnull().sum().sort_values(ascending=False)
# print(train_df)

# # 料金の頻度をヒストグラムで表示
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
# plt.scatter(train_df[pickup_latitude], train_df[pickup_longitude])
# plt.show()
# train_df.plot.scatter(x='pickup_latitude', y='pickup_longitude', alpha=0.2)
# plt.savefig("emap.png")
# plt.show()

# 人数と料金に相関があるかを調査
# train_df.plot.scatter(x='passenger_count', y='fare_amount', alpha=0.2)
# plt.savefig("e10passandfare.png")
# plt.show()
# test_df.plot.scatter(x='pickup_latitude', y='pickup_longitude', alpha=0.2)
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


def add_datetime_info(dataset):
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
    
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['weekday'] = dataset.pickup_datetime.dt.weekday
    dataset['year'] = dataset.pickup_datetime.dt.year
    
    return dataset.drop(['pickup_datetime'],axis=1)


print("add datetime info")
train_df = add_datetime_info(train_df)

# 距離計算
def calc_distance(dataset):
    dataset['distance'] = r * np.arccos((np.sin(dataset.pickup_latitude * np.pi/180) * np.sin(dataset.dropoff_latitude * np.pi/180))
                                         + (np.cos(dataset.pickup_latitude * np.pi/180) * np.cos(dataset.dropoff_latitude * np.pi/180) * np.cos((dataset.dropoff_longitude * np.pi/180) - (dataset.pickup_longitude * np.pi/180))))
    dataset.loc[(dataset['pickup_latitude'] == dataset['dropoff_latitude'])&(dataset['pickup_longitude'] == dataset['dropoff_longitude']), 'distance'] = 0
    
    return dataset


print("add distance info")
train_df = calc_distance(train_df)

# 学習データと評価データを作成
print("spliting...")
x_train, x_test, y_train, y_test = train_test_split(train_df.iloc[:, 2:13], train_df.iloc[:, 1],test_size=TestSize, random_state=777)
print('...complete')


# #データを標準化
# sc = StandardScaler()
# sc.fit(x_train) #学習用データで標準化
# x_train_std = sc.transform(x_train)
# x_test_std = sc.transform(x_test)

# # モデルの学習
# GBDT = GradientBoostingRegressor()
# GBDT.fit(x_train, y_train)

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

# ランダムフォレスト
print("fitting...")
model = RFR(n_jobs=1, max_features=6, random_state=777)
model.fit(x_train, y_train)
pred_model = model.predict(x_test)
print('...complete')

# 回帰　
print("predicting...")
test_pred = model.predict(x_test)
train_pred = model.predict(x_train)
print("...complete")

# 評価
# 決定係数(R2)
test_r2 = r2_score(y_test, test_pred)
train_r2 = r2_score(y_train, train_pred)

# 平均絶対誤差(MAE)
test_mae = mean_absolute_error(y_test, test_pred)
train_mae = mean_absolute_error(y_train, train_pred)

# 二乗平均平方根誤差 (RMSE)
test_rmse = mean_squared_error(y_test, test_pred)
train_rmse = mean_squared_error(y_train, train_pred)

print("R2   -> test : %.3f  train : %.3f" % (test_r2, train_r2))
print("MAE  -> test : %.3f  train : %.3f" % (test_mae, train_mae))
print("RMSE -> test : %.3f  train : %.3f" % (test_rmse, train_rmse))

# 変数重要度
# print("feature_importances = ", model.feature_importances_)
feature_importances = pd.DataFrame({
    "features": x_test.columns,
    "importances": model.feature_importances_
})
print(feature_importances)


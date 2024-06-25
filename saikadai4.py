import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as seb
import math

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor as RFR
from decimal import Decimal,  ROUND_HALF_UP
from scipy.stats import norm, shapiro, t
import scipy.stats as st
from sklearn.preprocessing import power_transform, PowerTransformer

# 読み込み数
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)
number = 12500
TestSize = 0.25
r = 6378.137
path = "/Users/hironeko1234/Documents/プログラミング/最終課題1/statics/Q-Q"

# データ読み込み
print("loading data")
train_df = pd.read_csv('/Users/hironeko1234/Documents/プログラミング/最終課題1/new-york-city-taxi-fare-prediction 2/train.csv')
train_df = train_df.sample(n=number, random_state=777)


# 距離計算
def calc_distance(dataset):
    dataset['distance'] = r * np.arccos((np.sin(dataset.pickup_latitude * np.pi/180) * np.sin(dataset.dropoff_latitude * np.pi/180))
                                         + (np.cos(dataset.pickup_latitude * np.pi/180) * np.cos(dataset.dropoff_latitude * np.pi/180) * np.cos((dataset.dropoff_longitude * np.pi/180) - (dataset.pickup_longitude * np.pi/180))))
    dataset.loc[(dataset['pickup_latitude'] == dataset['dropoff_latitude'])&(dataset['pickup_longitude'] == dataset['dropoff_longitude']), 'distance'] = 0.000095

    return dataset


print("add distance info")
train_df = calc_distance(train_df)


# 欠損値の除去
def cleanup1(df):
    return df[(df.fare_amount > 0) & (df.passenger_count > 0) & (df.passenger_count < 8)
            & (df.pickup_longitude < -70) & (df.pickup_longitude > -80) & (df.pickup_latitude > 35) & (df.pickup_latitude < 45)
            & (df.dropoff_longitude < -70) & (df.dropoff_longitude > -80) & (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45)]


train_df = cleanup1(train_df)

print("Cleanup compleat")


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

print("box-cox compleat")


# 日時の情報を付加
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

sort_df = pd.DataFrame(columns=train_df.columns)


def remove_out(df):
    # thresholdを計算
    # 平均
    mean = st.tmean(df['distance'])
    # 不偏分散
    var = st.tvar(df['distance'])
    # 信頼区間(%)
    alpha = 0.95
    # 自由度
    deg_of_freedom = len(train_df) - 1
    # 標準偏差
    scale = math.sqrt(var/len(train_df['distance']))
    # t分布における区間推定
    upper_threshold, lower_threshold = st.t.interval(alpha, deg_of_freedom, loc=mean, scale=scale)

    if lower_threshold > upper_threshold:
        lower_threshold, upper_threshold = upper_threshold, lower_threshold

    if not math.isnan(upper_threshold) and not math.isnan(lower_threshold):
        upper_threshold = int(upper_threshold)
        lower_threshold = int(lower_threshold)
        sort_df = df.sort_values('distance')
        # dfをthresholdでdropして返す
        sort_df = sort_df.reset_index()
        return sort_df.iloc[:, lower_threshold:(len(sort_df) - upper_threshold)]
    else:
        pass


_fare_amount = 16.5


def dropout(df):
    df_cleaned = pd.DataFrame([])
    for i in np.arange(0, 288, 0.1):
        if len(df[df.fare_amount == i]) == 0:
            continue

        if (len(df[df.fare_amount == i]) < 4) or (i >= _fare_amount):
            df_cleaned = pd.concat([df_cleaned, df[df.fare_amount == i]])

            continue

        df_cleaned = pd.concat([df_cleaned, remove_out(df[df.fare_amount == i])])

    return df_cleaned




def remove_out(df):
    # thresholdを計算
    # 平均
    mean = st.tmean(df['distance'])
    # 不偏分散
    var = st.tvar(df['distance'])
    # 信頼区間(%)
    alpha = 0.95
    # 自由度
    deg_of_freedom = len(train_df) - 1
    # 標準偏差
    scale = math.sqrt(var/len(train_df['distance']))
    # t分布における区間推定
    upper_threshold, lower_threshold = st.t.interval(alpha, deg_of_freedom, loc=mean, scale=scale)

    if lower_threshold > upper_threshold:
        lower_threshold, upper_threshold = upper_threshold, lower_threshold

    if not math.isnan(upper_threshold) and not math.isnan(lower_threshold):
        upper_threshold = int(upper_threshold)
        lower_threshold = int(lower_threshold)
        sort_df = df.sort_values('distance')
        # dfをthresholdでdropして返す
        sort_df = sort_df.reset_index()
        return sort_df.iloc[:, lower_threshold:(len(sort_df) - upper_threshold)]
    else:
        pass


_fare_amount = 16.5


def dropout(df):
    df_cleaned = pd.DataFrame([])
    for i in np.arange(0, 288, 0.1):
        if len(df[df.fare_amount == i]) == 0:
            continue

        if (len(df[df.fare_amount == i]) < 4) or (i >= _fare_amount):
            df_cleaned = pd.concat([df_cleaned, df[df.fare_amount == i]])

            continue

        df_cleaned = pd.concat([df_cleaned, remove_out(df[df.fare_amount == i])])

    return df_cleaned


# def plot_qq_plot(X):
#     plt.figure(figsize=(6, 6))
#     filt_train_df = train_df[train_df['fare_amount'] == 6.5]
#     st.probplot(filt_train_df['distance'], dist="norm", plot=plt)
#     plt.title('Q-Q Plot')
#     plt.xlabel('Theoretical Quantiles')
#     plt.ylabel('Sample Quantiles')
#     plt.savefig(path + 'box6.5 outlier deleated' + '.png')
#     plt.close()


# plot_qq_plot(train_df)


def pred(x_train, x_test, y_train, y_test):

    # モデルのトレーニング
    print("fitting...")

    model = RFR(n_jobs=-1, random_state=777, max_depth=18, min_samples_split=9, min_samples_leaf=4) # ランダムフォレスト
    # model = GBR(random_state=777) # 勾配ブースティング木

    model.fit(x_train, y_train)

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
    # test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_rmse = mean_squared_error(y_test, test_pred)
    # train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_rmse = mean_squared_error(y_train, train_pred)

    print("R2   -> test : %.3f  train : %.3f" % (test_r2, train_r2))
    print("MAE  -> test : %.3f  train : %.3f" % (test_mae, train_mae))
    print("RMSE -> test : %.3f  train : %.3f" % (test_rmse, train_rmse))

    # 変数重要度
    feature_importances = pd.DataFrame({
        "features": x_test.columns,
        "importances": model.feature_importances_
    })
    print(feature_importances)


train_df = train_df.drop(['key'],axis=1)
print(train_df.info())
x_train, x_test, y_train, y_test = train_test_split(train_df.iloc[:, 1:12], train_df.iloc[:, 0],test_size=TestSize, random_state=777)

train_df = dropout(train_df)
print("Outlier removal completed")
print(train_df)

x_train1, x_test1, y_train1, y_test1 = train_test_split(train_df.iloc[:, 1:12], train_df.iloc[:, 0],test_size=TestSize, random_state=777)
pred(x_train1, x_test, y_train1, y_test)

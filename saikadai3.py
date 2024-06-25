import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as seb
import torch
import requests
import subprocess
import scipy.stats as stats

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import PowerTransformer
from decimal import Decimal,  ROUND_HALF_UP
from scipy.stats import norm, shapiro, t


# 読み込み数
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)
number = 12500
TestSize = 0.25
r = 6378.137

# データ読み込み
print("loading data")
train_df = pd.read_csv('/Users/hironeko1234/Documents/プログラミング/最終課題1/new-york-city-taxi-fare-prediction 2/train.csv')
train_df = train_df.sample(n=number, random_state=777)
test_df = pd.read_csv('/Users/hironeko1234/Documents/プログラミング/最終課題1/new-york-city-taxi-fare-prediction 2/test.csv')


# 距離計算
def calc_distance(dataset):
    dataset['distance'] = r * np.arccos((np.sin(dataset.pickup_latitude * np.pi/180) * np.sin(dataset.dropoff_latitude * np.pi/180))
                                         + (np.cos(dataset.pickup_latitude * np.pi/180) * np.cos(dataset.dropoff_latitude * np.pi/180) * np.cos((dataset.dropoff_longitude * np.pi/180) - (dataset.pickup_longitude * np.pi/180))))
    dataset.loc[(dataset['pickup_latitude'] == dataset['dropoff_latitude'])&(dataset['pickup_longitude'] == dataset['dropoff_longitude']), 'distance'] = 0
    
    return dataset


print("add distance info")
train_df = calc_distance(train_df)
test_df = calc_distance(test_df)

# 欠損値の除去
def cleanup1(df):
    return df[(df.fare_amount > 0) & (df.passenger_count > 0) & (df.passenger_count < 8)
            & (df.pickup_longitude < -70) & (df.pickup_longitude > -80) & (df.pickup_latitude > 35) & (df.pickup_latitude < 45)
            & (df.dropoff_longitude < -70) & (df.dropoff_longitude > -80) & (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) & (df.distance > 0)]


train_df = cleanup1(train_df)

# box-cox変換
def apply_boxcox_transform_sklearn(df):
    # データフレームから 'fare_amount' と 'distance' カラムを抽出
    # fare_amount_data = df['fare_amount'].values.reshape(-1, 1)
    distance_data = df['distance'].values.reshape(-1, 1)
    
    # PowerTransformer を使用して Box-Cox 変換を適用
    transformer = PowerTransformer(method='box-cox', standardize=False)  # standardize=False はデータのスケーリングを無効にします
    # fare_amount_transformed = transformer.fit_transform(fare_amount_data)
    distance_transformed = transformer.fit_transform(distance_data)
    
    # 変換後のデータをデータフレームに戻す
    # df['fare_amount'] = fare_amount_transformed
    df['distance'] = distance_transformed
    
    return df


train_df = apply_boxcox_transform_sklearn(train_df[train_df['fare_amount'] == 6.5])
print(train_df.describe())

# 0.1を四捨五入し計算する関数
def round_list_to_one_decimal(value_list):
    rounded_list = [Decimal(value).quantize(Decimal('0.5'), rounding=ROUND_HALF_UP) for value in value_list]
    return rounded_list

path = "/Users/hironeko1234/Documents/プログラミング/最終課題1/statics/Q-Q"

# 各料金における分布を描画
# def drwGraph(X):
    # plt.figure(figsize=(10, 6))
    # plt.hist(filt_train_df['distance'], density=True, bins=200, color='c', alpha=0.7)
    # plt.title('Distribution of distance in' + '$' + str(X))
    # plt.xlabel('Distance(km)')
    # plt.ylabel('abundance')

    # # 正規分布のパラメータを推定
    # mu, sigma = norm.fit(filt_train_df['distance'])
    # x_range = plt.xlim()
    # x = np.linspace(x_range[0], x_range[1], 100)
    # p = norm.pdf(x, mu, sigma)
    # plt.plot(x, p, 'k', linewidth=2, label='Normal fit')

    # # シャピロ・ウィルク検定
    # shapiro_stat, shapiro_p_value = shapiro(filt_train_df['distance'])
    # plt.text(0.5, 0.9, f"P-value: {shapiro_p_value:.4f}", transform=plt.gca().transAxes, fontsize=12)
    # # 正規分布仮定の母集団平均値の95%信頼区間
    # mean = np.mean(filt_train_df['distance'])
    # std = np.std(filt_train_df['distance'], ddof=1)
    # t_score = t.ppf(0.975, df=len(filt_train_df['distance']) - 1)
    # conf_interval = (mean - t_score * std / np.sqrt(len(filt_train_df['distance'])), 
    #                  mean + t_score * std / np.sqrt(len(filt_train_df['distance'])))
    # plt.axvline(mean, color='r', linestyle='dashed', linewidth=1)
    # plt.axvline(conf_interval[0], color='g', linestyle='dotted', linewidth=1, label='95% CI')
    # plt.axvline(conf_interval[1], color='g', linestyle='dotted', linewidth=1)
    # plt.text(conf_interval[0], 0.01, f"{conf_interval[0]:.2f}", color='g', fontsize=10)
    # plt.text(conf_interval[1], 0.01, f"{conf_interval[1]:.2f}", color='g', fontsize=10)       

    # plt.savefig(path + '6.5' + 'box.png')
    # plt.close()

# filt_train_df = train_df[train_df['fare_amount'] == values]
# drwGraph(filt_train_df)
# rounded_fare_values = round_list_to_one_decimal(fare_values)

# for X in rounded_fare_values:
#     filt_train_df = train_df[train_df['fare_amount'] == X]
#      # サンプル数が3未満の場合はスキップ
#     if len(filt_train_df) < 3:
#         continue
    
#     drwGraph(X)


# Q-Qプロットを描画
def plot_qq_plot(X):
    plt.figure(figsize=(6, 6))
    stats.probplot(filt_train_df['distance'], dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.savefig(path + 'b6.5' + '.png')
    plt.close()

# filt_train_df = train_df[train_df['fare_amount'] == 3.5]
filt_train_df = train_df
plot_qq_plot(filt_train_df)
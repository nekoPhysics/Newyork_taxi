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
from decimal import Decimal,  ROUND_HALF_UP
from scipy.stats import norm, shapiro, t
import scipy.stats as stats



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
# def cleanup1(df):
#     return df[(df.fare_amount > 0) & (df.passenger_count > 0) & (df.passenger_count < 8)
#             & (df.pickup_longitude < -73.75) & (df.pickup_longitude > -73.8) & (df.pickup_latitude > 40.62) & (df.pickup_latitude < 40.66)
#             & (df.dropoff_longitude < -70) & (df.dropoff_longitude > -80) & (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) & (df.distance > 0)]

def cleanup1(df):
    return df[(df.fare_amount > 0) & (df.passenger_count > 0) & (df.passenger_count < 8)
            & (df.pickup_longitude < -70) & (df.pickup_longitude > -80) & (df.pickup_latitude > 35) & (df.pickup_latitude < 45)
            & (df.dropoff_longitude < -70) & (df.dropoff_longitude > -80) & (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) & (df.distance > 0)]


train_df = cleanup1(train_df)
# test_df = cleanup1(test_df)

print(train_df)

# カラム'fare_amount'の各パーセンタイルに対応する値を計算
percentiles = np.arange(10, 101, 5)
percentile_values = np.percentile(train_df['fare_amount'], percentiles)

# パーセンタイルと対応する値を表示
for p, value in zip(percentiles, percentile_values):
    print(f'{p}% : {value}')

# 0.1を四捨五入し計算する関数
# def round_list_to_one_decimal(value_list):
#     rounded_list = [Decimal(value).quantize(Decimal('0.5'), rounding=ROUND_HALF_UP) for value in value_list]
#     return rounded_list

# fare_values = [0.5 * i for i in range(601)]
# # rounded_fare_values = round_list_to_one_decimal(fare_values)
# path = "/Users/hironeko1234/Documents/プログラミング/最終課題1/statics/Q-Q"

# 各料金における分布を描画
# def drwGraph(X):
#     filt_train_df = train_df[train_df['fare_amount'] == X]
#     plt.figure(figsize=(10, 6))
#     plt.hist(filt_train_df['distance'], density=True, bins=200, color='c', alpha=0.7)
#     plt.title('Distribution of distance in' + '$' + str(X))
#     plt.xlabel('Distance(km)')
#     plt.ylabel('abundance')

#     # 正規分布のパラメータを推定
#     mu, sigma = norm.fit(filt_train_df['distance'])
#     x_range = plt.xlim()
#     x = np.linspace(x_range[0], x_range[1], 100)
#     p = norm.pdf(x, mu, sigma)
#     plt.plot(x, p, 'k', linewidth=2, label='Normal fit')

#     # シャピロ・ウィルク検定
#     shapiro_stat, shapiro_p_value = shapiro(filt_train_df['distance'])
#     plt.text(0.5, 0.9, f"P-value: {shapiro_p_value:.4f}", transform=plt.gca().transAxes, fontsize=12)
#     # 正規分布仮定の母集団平均値の95%信頼区間
#     mean = np.mean(filt_train_df['distance'])
#     std = np.std(filt_train_df['distance'], ddof=1)
#     t_score = t.ppf(0.975, df=len(filt_train_df['distance']) - 1)
#     conf_interval = (mean - t_score * std / np.sqrt(len(filt_train_df['distance'])), 
#                      mean + t_score * std / np.sqrt(len(filt_train_df['distance'])))
#     plt.axvline(mean, color='r', linestyle='dashed', linewidth=1)
#     plt.axvline(conf_interval[0], color='g', linestyle='dotted', linewidth=1, label='95% CI')
#     plt.axvline(conf_interval[1], color='g', linestyle='dotted', linewidth=1)
#     plt.text(conf_interval[0], 0.01, f"{conf_interval[0]:.2f}", color='g', fontsize=10)
#     plt.text(conf_interval[1], 0.01, f"{conf_interval[1]:.2f}", color='g', fontsize=10)       

#     plt.savefig(path + str(X) + '.png')
#     plt.close()

# for X in rounded_fare_values:
#     filt_train_df = train_df[train_df['fare_amount'] == X]
#      # サンプル数が3未満の場合はスキップ
#     if len(filt_train_df) < 3:
#         continue
    
#     drwGraph(X)

# def plot_qq_plot(X):
#     plt.figure(figsize=(6, 6))
#     stats.probplot(filt_train_df['distance'], dist="norm", plot=plt)
#     plt.title('Q-Q Plot')
#     plt.xlabel('Theoretical Quantiles')
#     plt.ylabel('Sample Quantiles')
#     plt.savefig(path + '20' + '.png')
#     plt.close()

# filt_train_df = train_df[train_df['fare_amount'] == 20]
# plot_qq_plot(filt_train_df)



# for X in rounded_fare_values:
#     filt_train_df = train_df[train_df['fare_amount'] == X]
#      # サンプル数が3未満の場合はスキップ
#     if len(filt_train_df) < 3:
#         continue
    
#     plot_qq_plot(X)




# 料金の分布
# plt.figure(figsize=(10, 6))
# plt.hist(train_df['fare_amount'], density=False, bins=300)
# plt.title('Distribution of FareAmount(<30)')
# plt.savefig('Distribution of Fare_amount', dpi=500)
# plt.show()

# 距離の分布
# plt.figure(figsize=(10, 6))
# plt.hist(train_df['fare_amount'], density=False, bins=100)
# plt.title('Distribution of fare_amount (JFK)')
# plt.savefig('Distribution of fare_amount(JFK).png', dpi=500)
# plt.show()

# 運賃の最頻値(Mode)を計算
# mode_fare = train_df['fare_amount'].mode()[0]
# print("Mode of Fare:", mode_fare)

# 距離の最頻値を計算
# mode_distance = train_df['distance'].mode()[0]
# print("Mode of distance:", mode_distance)

# print(train_df.describe())
# output_text = train_df.describe()

# # 距離と料金に関するグラフの作成
# print("DRAWING GRAPH4...")
# plt.scatter(train_df['distance'], train_df['fare_amount'], c='b', label="plot", alpha=0.05)
# x = train_df['distance']
# y = train_df['fare_amount']
# plt.title("Scatter plot for distance and fare")
# plt.xlabel("distance")
# plt.ylabel("fare")
# slope, intercept = np.polyfit(x, y, 1)
# plt.plot(x, slope * x + intercept, c="r", label="Ism")  # 直線式を作成してプロット
# equation_text = f'y = {slope:.2f}x + {intercept:.2f}'
# plt.text(2, 25, equation_text, fontsize=12, color='r')
# plt.legend()
# plt.savefig("Scatter plot for distance and fare_3.png")
# plt.show()

# テキストファイルへ保存
# def save_output():
#     file_path = "/Users/hironeko1234/Documents/プログラミング/最終課題1/statics/statics.csv"
#     with open(file_path, mode='a') as file:
#         file.write('\n'.join(output_text))
#     print("save complete")
#     return()

# save_output()


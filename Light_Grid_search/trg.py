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
from seaborn_analyzer import regplot
import optuna
import lightgbm as lgb
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# 別ファイルfnc.pyをインポート、*でワイルドカード（モジュール内の変数や定義を全てインポート)

from fnc import *

# 読み込み数
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)
TestSize = 0.2
r = 6378.137
Pvalue = 3
lower_limit =14.1
upper_limit = 60


def lern_cycle(train_df):
    Execution_time = time.perf_counter()
    # データ読み込み
    print("LOADING DATA...PLEASE WAIT...")
    df_imp = train_df
    df_imp = df_imp.dropna(how='any')
    # df_imp = df_imp.drop(['key'],axis=1)

    print("PROCESSING...")
    df_imp = add_datetime_info(df_imp)
    # df_imp = datetime_maker(df_imp)
    print(df_imp)
    df_imp['hour'] = df_imp['hour'].astype('int')
    df_imp['day'] = df_imp['day'].astype('int')
    df_imp['month'] = df_imp['month'].astype('int')
    df_imp['weekday'] = df_imp['weekday'].astype('int')
    df_imp['year'] = df_imp['year'].astype('int')
    df_imp = df_imp.drop('key', axis=1)
    df_imp = df_imp.drop('pickup_datetime', axis=1)
    df_imp = calc_distance(df_imp, r)
    df_imp = cleanup(df_imp)
    # df_imp = df_imp.drop(['pickup_datetime'])
    # df_imp = df_imp.drop(['hour', 'day', 'month', 'weekday'], axis=1)
    df_imp = apply_boxcox_transform_sklearn(df_imp)
    df_imp = dropout(df_imp, Pvalue, lower_limit, upper_limit)
    print(df_imp)
    print(df_imp.describe)
    print(df_imp.dtypes)

    # objective variable
    Object_V = 'fare_amount'
    # explanatory variables
    Expl_V = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'year', 'hour', 'day', 'month', 'weekday','distance']
    y_nd = df_imp[Object_V].values
    x_nd = df_imp[Expl_V].values

    # LightGBMの実装
    x_train, x, y_train, y = train_test_split(df_imp.iloc[:, 1:13], df_imp.iloc[:, 0], test_size=TestSize, random_state=777)
    # x_test = df_test.iloc[:, 1:13]
    train_set = lgb.Dataset(x_train, y_train)
    test_set = lgb.Dataset(x, y, reference=train_set)
    # x_test_set = lgb.Dataset(x_test)
    # 使用するチューニング対象外のパラメータ
    params = {
        'objective': 'regression',  # 最小化させるべき損失関数
        'metric': 'rmse',  # 学習時に使用する評価指標(early_stoppingの評価指標にも同じ値が使用される)
        'random_state': 777,  # 乱数シード
        'boosting_type': 'gbdt',  # boosting_type
        'n_estimators': 10000,  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
        'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
        # 'early_stopping_rounds': 10  # ここでearly_stoppingを指定
        }
    # モデル作成
    # model = lgb.train(params, train_set=train_set, valid_sets=[train_set, test_set])
    light_gbm = lgb_tune.train(
                    params,
                    train_set,
                    valid_sets=test_set,
                    num_boost_round=10000,
                    early_stopping_rounds=1000,
                    verbose_eval=50)
    # 学習時fitパラメータ指定 (early_stopping用のデータeval_setを渡す)
    fit_params = {
        'eval_set': [(x_train, y_train)]
        }
    # # クロスバリデーションして予測値ヒートマップを可視化
    # cv = KFold(n_splits=3, shuffle=True, random_state=777)  # KFoldでクロスバリデーション分割指定
    # regplot.regression_heat_plot(model, x=x_nd, y=y_nd, x_colnames=Expl_V,
    #                             pair_sigmarange = 0.5, rounddigit_x1=3, rounddigit_x2=3,
    #                             cv=cv, display_cv_indices=0,
    #                             fit_params=fit_params, validation_fraction=None)
    end_time = time.perf_counter()
    print("time: %.3f s" % (end_time - Execution_time))
    print(light_gbm.params)
    # 評価
    # train_pred = model.predict(x_train)
    # test_pred = model.predict(x, num_iteration=model.best_iteration)
    # # 決定係数(R2)
    # # y = np.array(y)
    # # test_set = np.array(test_set)
    # # test_r2 = r2_score(y, test_pred)
    # # train_r2 = r2_score(y_train, test_pred)

    # # 二乗平均平方根誤差 (RMSE)
    # test_mae = mean_squared_error(y, test_pred)
    # train_mae = mean_squared_error(y_train, train_pred)

    # # 二乗平均平方根誤差 (RMSE)
    # test_rmse = np.sqrt(mean_squared_error(y, test_pred))
    # train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

    # # print("R2     -> test : %.3f  train : %.3f" % (test_r2, train_r2))
    # print("MSE   -> test : %.3f  train : %.3f" % (test_mae, train_mae))
    # print("RMSE -> test : %.3f  train : %.3f" % (test_rmse, train_rmse))

    # # 変数重要度
    # feature_importance = pd.DataFrame({
    #     "features": x.columns,
    #     "importances": model.feature_importance(importance_type='gain')
    # })
    # print(feature_importance)


print("PLE LOADING DATA...")
df = pd.read_csv('/Users/hironeko1234/Documents/プログラミング/最終課題1/new-york-city-taxi-fare-prediction 2/train.csv')
df_test = pd.read_csv('/Users/hironeko1234/Documents/プログラミング/最終課題1/new-york-city-taxi-fare-prediction 2/test.csv')
df = df.sample(n=905000, random_state=777)

# df = add_datetime_info(df)
# df = calc_distance(df, r)
# df = df[df.distance > 0]
# df = apply_boxcox_transform_sklearn(df)
# df = cleanup(df)

# LightGBMの実装
# x_train, x, y_train, y = train_test_split(df.iloc[:, 1:13], df.iloc[:, 0],test_size=TestSize, random_state=777)
lern_cycle(df)

gc.collect()

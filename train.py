import math
import os
import warnings

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import jpx_tokyo_market_prediction
import xgboost as xgb
from scipy import stats
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                     TimeSeriesSplit, cross_val_score)

warnings.filterwarnings('ignore')


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


SEED = 42
seed_everything(SEED)

input_dir = '~/Downloads/jpx-tokyo-stock-exchange-prediction'

train = pd.read_csv(
    os.path.join(
        input_dir,
        "train_files/stock_prices.csv"),
    parse_dates=["Date"])

train = train.drop(
    columns=[
        'RowId',
        'ExpectedDividend',
        'AdjustmentFactor',
        'SupervisionFlag']).dropna().reset_index(
            drop=True)

print(train.head())


def add_features(feats):
    # 20個工作日收盤價回報
    feats["return_1month"] = feats["Close"].pct_change(20)
    # 終値の40営業日リターン
    feats["return_2month"] = feats["Close"].pct_change(40)
    # 終値の60営業日リターン
    feats["return_3month"] = feats["Close"].pct_change(60)
    # 終値の20営業日ボラティリティ
    feats["volatility_1month"] = (
        np.log(feats["Close"]).diff().rolling(20).std()
    )
    # 終値の40営業日ボラティリティ
    feats["volatility_2month"] = (
        np.log(feats["Close"]).diff().rolling(40).std()
    )
    # 終値の60営業日ボラティリティ
    feats["volatility_3month"] = (
        np.log(feats["Close"]).diff().rolling(60).std()
    )
    # 終値と20営業日の単純移動平均線の乖離
    feats["MA_gap_1month"] = feats["Close"] / (
        feats["Close"].rolling(20).mean()
    )
    # 終値と40営業日の単純移動平均線の乖離
    feats["MA_gap_2month"] = feats["Close"] / (
        feats["Close"].rolling(40).mean()
    )
    # 終値と60営業日の単純移動平均線の乖離
    feats["MA_gap_3month"] = feats["Close"] / (
        feats["Close"].rolling(60).mean()
    )

    #威廉指標
    #20天內的最高價
    #20天內的最低價
    feats["higest_20"] = feats["Close"].rolling(20).max()
    feats["lowest_20"] = feats["Close"].rolling(20).min()
    
    feats["higest_40"] = feats["Close"].rolling(40).max()
    feats["lowest_40"] = feats["Close"].rolling(40).min()
    
    feats["higest_60"] = feats["Close"].rolling(60).max()
    feats["lowest_60"] = feats["Close"].rolling(60).min()

    feats["Williams_20day"] = (feats["Close"]-feats["higest_20"]) / (feats["higest_20"] - feats["lowest_20"])
    feats["Williams_40day"] = (feats["Close"]-feats["higest_40"]) / (feats["higest_40"] - feats["lowest_40"])
    feats["Williams_60day"] = (feats["Close"]-feats["higest_60"]) / (feats["higest_60"] - feats["lowest_60"])
    return feats


train = add_features(train)

print(train.head())


def feval_rmse(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'rmse', mean_squared_error(y_true, y_pred), False


def feval_pearsonr(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'pearsonr', stats.pearsonr(y_true, y_pred)[0], True


def calc_spread_return_per_day(df, portfolio_size=200, toprank_weight_ratio=2):
    assert df['Rank'].min() == 0
    assert df['Rank'].max() == len(df['Rank']) - 1
    weights = np.linspace(
        start=toprank_weight_ratio,
        stop=1,
        num=portfolio_size)
    purchase = (df.sort_values(by='Rank')[
                'Target'][:portfolio_size] * weights).sum() / weights.mean()
    short = (df.sort_values(by='Rank', ascending=False)[
             'Target'][:portfolio_size] * weights).sum() / weights.mean()
    return purchase - short


def calc_spread_return_sharpe(
        df: pd.DataFrame, portfolio_size=200, toprank_weight_ratio=2):
    buf = df.groupby('Date').apply(
        calc_spread_return_per_day,
        portfolio_size,
        toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio  # , buf


def add_rank(df):
    df["Rank"] = df.groupby("Date")["Target"].rank(
        ascending=False, method="first") - 1
    df["Rank"] = df["Rank"].astype("int")
    return df


def fill_nan_inf(df):
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    return df


def check_score(df, preds, Securities_filter=[]):
    tmp_preds = df[['Date', 'SecuritiesCode']].copy()
    tmp_preds['Target'] = preds

    # Rank Filter. Calculate median for this date and assign this value to the
    # list of Securities to filter.
    tmp_preds['target_mean'] = tmp_preds.groupby(
        "Date")["Target"].transform('median')
    tmp_preds.loc[tmp_preds['SecuritiesCode'].isin(
        Securities_filter), 'Target'] = tmp_preds['target_mean']

    tmp_preds = add_rank(tmp_preds)
    df['Rank'] = tmp_preds['Rank']
    score = round(
        calc_spread_return_sharpe(
            df,
            portfolio_size=200,
            toprank_weight_ratio=2),
        5)
    score_mean = round(
        df.groupby('Date').apply(
            calc_spread_return_per_day,
            200,
            2).mean(),
        5)
    score_std = round(
        df.groupby('Date').apply(
            calc_spread_return_per_day,
            200,
            2).std(),
        5)
    print(
        f'Competition_Score:{score}, rank_score_mean:{score_mean}, rank_score_std:{score_std}')


list_spred_h = list(
    (train.groupby('SecuritiesCode')['Target'].max() -
     train.groupby('SecuritiesCode')['Target'].min()).sort_values()[
        :1000].index)
list_spred_l = list(
    (train.groupby('SecuritiesCode')['Target'].max() -
     train.groupby('SecuritiesCode')['Target'].min()).sort_values()[
        1000:].index)

# Training just with Securities with hight target_spread and validated
# with Securities with low target_spread.

# features = ['High', 'Low', 'Open', 'Close', 'Volume', 'return_1month', 'return_2month', 'return_3month', 'volatility_1month', 'volatility_2month', 'volatility_3month',
#             'MA_gap_1month', 'MA_gap_2month', 'MA_gap_3month']

features = ['High', 'Low', 'Open', 'Close', 'Volume', 'return_1month', 'return_2month', 'return_3month', 'volatility_1month', 'volatility_2month', 'volatility_3month',
            'MA_gap_1month', 'MA_gap_2month', 'MA_gap_3month','Williams_20day','Williams_40day','Williams_60day']
# features =['High','Low','Open','Close','Volume',]
train = fill_nan_inf(train)

params_lgb = {
    'learning_rate': 0.005,
    'metric': 'None',
    'objective': 'regression',
    'boosting': 'gbdt',
    'verbosity': 0,
    'n_jobs': -1,
    'force_col_wise': True}

tr_dataset = lgb.Dataset(train[train['SecuritiesCode'].isin(list_spred_h)][features],
                         train[train['SecuritiesCode'].isin(list_spred_h)]["Target"], feature_name=features)
vl_dataset = lgb.Dataset(train[train['SecuritiesCode'].isin(list_spred_l)][features],
                         train[train['SecuritiesCode'].isin(list_spred_l)]["Target"], feature_name=features)

# 1ep = batch size * iterations = training data
model = lgb.train(params=params_lgb,
                  train_set=tr_dataset,
                  valid_sets=[tr_dataset, vl_dataset],
                  num_boost_round=1,  # Number of boosting iterations. # 3000
                  feval=feval_pearsonr,
                  callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=True), lgb.log_evaluation(period=100)])

# Ranking filtering by Securities with previous list based in target
# spread and target mean.

test = pd.read_csv(os.path.join(
    input_dir,
    "supplemental_files/stock_prices.csv"), parse_dates=["Date"])
test = test.drop(
    columns=[
        'RowId',
        'ExpectedDividend',
        'AdjustmentFactor',
        'SupervisionFlag'])
test = add_features(test)
test = fill_nan_inf(test)
preds = model.predict(test[features])
print(math.sqrt(mean_squared_error(preds, test.Target)))

sample_submission = pd.read_csv(
    "../input/jpx-tokyo-stock-exchange-prediction/example_test_files/sample_submission.csv")

# env = jpx_tokyo_market_prediction.make_env()   # initialize the environment
# iter_test = env.iter_test()    # an iterator which loops over the test files
# for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
#     prices = add_features(prices)
#     prices['Target'] = model.predict(fill_nan_inf(prices)[features])
#     prices['target_mean']=prices.groupby("Date")["Target"].transform('median')
#     prices.loc[prices['SecuritiesCode'].isin(list_spred_h),'Target']=prices['target_mean']
#     prices = add_rank(prices)
#     sample_prediction['Rank'] = prices['Rank']
#     env.predict(sample_prediction)

# sample_prediction.head(5)

check_score(test, preds)
check_score(test, preds, list_spred_h)
check_score(test, preds, list_spred_l)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
グラフ生成する
"""
import pandas as pd
from module import util

pd.options.display.max_rows = 50
pd.options.display.max_columns = 50
pd.options.display.width = 1000

# データを読み込む
rating, user, movie = util.read_ml20m_data()

# 90日間ずつにデータを分割する
train_test_days = 90
n_hold = 10
one_day_sec = 60 * 60 * 24
max_seconds = rating.Timestamp.max()

seps = sorted([max_seconds - (i) * one_day_sec * train_test_days for i in range(n_hold+2)])
sep_indexs = [(seps[i-1]<=rating.Timestamp)&(rating.Timestamp<seps[i]) for i in range(1, n_hold+2)]

# 最新の日付だけ取り出す。
latest_rating = rating.loc[sep_indexs[-1], :]

# データ全体を可視化
import matplotlib.pyplot as plt

# データを表示
print(latest_rating[:20])
print(movie[:20])

# y軸をlogでスケールしない場合
plt.ylim(1,10**4.12)
latest_rating.groupby(by='UserID')['MovieID'].count().hist(
        bins=100, range=(0,1500), color='black')
plt.title('count of rating by userID in 90days')

plt.ylim(1, 10**4.12)
latest_rating.groupby(by='MovieID')['UserID'].count().hist(
        bins=100, range=(0,1500), color='grey')
plt.title('count of rating by MovieID in 90days')


# y軸をlogでスケールした場合
plt.yscale('log')
plt.ylim(1,10**4.12)
latest_rating.groupby(by='UserID')['MovieID'].count().hist(
        bins=100, range=(0,1500), color='black')
plt.title('count of rating by userID in 90days (log scaled)')

plt.yscale('log')
plt.ylim(1,10**4.12)
latest_rating.groupby(by='MovieID')['UserID'].count().hist(
        bins=100, range=(0,1500), color='grey')
plt.title('count of rating by MovieID in 90days (log scaled)')


# userID と itemID の度数分布
cont_User_ID  = latest_rating.groupby(by='UserID')['MovieID'].count()
cont_Movie_ID = latest_rating.groupby(by='MovieID')['UserID'].count()

latest_rating['rating_cnt_of_UserID'] = list(cont_User_ID[latest_rating['UserID']])
latest_rating['rating_cnt_of_MovieID'] = list(cont_Movie_ID[latest_rating['MovieID']])

fig = plt.figure(figsize=(12,10))
ax1 = latest_rating.plot.scatter(
        x='rating_cnt_of_UserID', y='rating_cnt_of_MovieID'
        , c='grey', alpha=0.003, s=1)

# 前の期間と後の期間でのユーザーとアイテムのレビュー数
before_rating = rating.loc[sep_indexs[-2], :]

## ユーザーの前後期間のレビュー数を確認
a = latest_rating.groupby(by='UserID')['MovieID'].count()
b = before_rating.groupby(by='UserID')['MovieID'].count()

ab = pd.concat([a,b], axis=1).fillna(0)
ab.columns = ['rating_cnt_in_this_term', 'rating_cnt_in_before_term']
ax1 = ab.plot.scatter(
        x='rating_cnt_in_before_term', y='rating_cnt_in_this_term'
        , c='grey', alpha=0.05, s=2)
plt.xlim(-10, 750);plt.ylim(-20, 1500)
plt.title('comparison rating cnt by user between this term(=90days) and before')

## アイテムの前後期間のレビュー数を確認
a = latest_rating.groupby(by='MovieID')['UserID'].count()
b = before_rating.groupby(by='MovieID')['UserID'].count()

ab = pd.concat([a,b], axis=1).fillna(0)
ab.columns = ['rating_cnt_in_this_term', 'rating_cnt_in_before_term']
ax1 = ab.plot.scatter(
        x='rating_cnt_in_before_term', y='rating_cnt_in_this_term'
        , c='grey', alpha=0.05, s=2)
plt.xlim(-10, 1000);plt.ylim(-10, 1000)
plt.title('comparison rating cnt by item between this term(=90days) and before')




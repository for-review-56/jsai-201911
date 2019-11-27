#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
グラフ生成し、headとtailの効率的な分岐点を模索する。
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.module import util

pd.options.display.max_rows = 50
pd.options.display.max_columns = 50
pd.options.display.width = 1000

# データを読み込む
rating, user, movie = util.read_ml20m_data()

# 90日間ずつにデータを分割する
train_test_days = 30
n_hold = 10
one_day_sec = 60 * 60 * 24
max_seconds = rating.Timestamp.max()

seps = sorted([max_seconds - (i) * one_day_sec * train_test_days for i in range(n_hold+2)])
sep_indexs = [(seps[i-1]<=rating.Timestamp)&(rating.Timestamp<seps[i]) for i in range(1, n_hold+2)]

# 最新の日付だけ取り出す。
df_rating = rating.loc[sep_indexs[-1], :]

def search_best_head_tail():
    result = {'user':[], 'item':[]}
    for i,sep_index in enumerate(sep_indexs):
        df_rating = rating.loc[sep_index, :]

        user_ids = df_rating['UserID'].values
        best = gini_clustering(user_ids)
        sep_rate = len(best['tail_ids']) / (len(best['tail_ids']) + len(best['head_ids']))
        result['user'].append(sep_rate)
        
        item_ids = df_rating['MovieID'].values
        best = gini_clustering(item_ids)
        sep_rate = len(best['tail_ids']) / (len(best['tail_ids']) + len(best['head_ids']))
        result['item'].append(sep_rate)
        
    print("best rate of user is {}".format(np.mean(result['user'])))
    print("best rate of item is {}".format(np.mean(result['item'])))
        
        

def get_lorenz_curve_chart(fp_png='output/lorenz_curve.png'):
    '''ローレンツ曲線を出力する。10個の期間ごとに曲線を引く'''
    fig = plt.figure(figsize=(12, 5))
    ax_user = fig.add_subplot(121)
    ax_item = fig.add_subplot(122)
    
    ax_user.plot([0,1], [0,1], c='grey', linestyle='dashed')
    ax_item.plot([0,1], [0,1], c='grey', linestyle='dashed')
    for sep_index in sep_indexs:
        df_rating = rating.loc[sep_index, :]
        x, y = get_lorenz_curve_xy(df_rating['UserID'])
        ax_user.plot(x, y, c='black')
        x, y = get_lorenz_curve_xy(df_rating['MovieID'])
        ax_item.plot(x, y, c='black')
    
    ax_user.set_xlabel('ordered User ids by (tail← →head)')
    ax_item.set_xlabel('ordered Item ids by (tail← →head)')
    ax_user.set_ylabel('cumulative frequency rate')
    
    plt.savefig(fp_png)




from collections import Counter
def gini_clustering(ids):
    '''
    現在、head,tailの2分類にしか対応していません。
    
    ids = [1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4,5,5,6,6,7]
    gini_clustering(ids)
    '''
    _ids = np.array(ids)
    
    dict_count = Counter(ids)

    cnt_ids = np.array(list(dict_count.keys()))
    cnt_counts = np.array(list(dict_count.values()))

    seps = set(cnt_counts)
    best_sep, best_gini = None, np.inf
    result = {}
    for sep in seps:
        h_idx, t_idx = cnt_counts>sep, cnt_counts<=sep
        h_cnt_ids, t_cnt_ids = cnt_ids[h_idx], cnt_ids[t_idx]
        h_ids, t_ids = _ids[np.in1d(_ids, h_cnt_ids)], _ids[np.in1d(_ids, t_cnt_ids)]        
        _gini = len(h_ids)*ids_gini_coefficient(h_ids) + len(t_ids)*ids_gini_coefficient(t_ids)
        result[sep]=_gini
        if _gini < best_gini:
            best_sep, best_gini = sep, _gini
            head_ids, tail_ids = h_cnt_ids, t_cnt_ids
    
    return {'head_ids':head_ids, 'tail_ids':tail_ids, 'best_sep':best_sep}
            

def ids_gini_diversity_index(ids):
    '''
    決定木などに使われるジニ不純度。
    不均等の指標であるジニ係数とは計算も意味も違うので注意。
    
    ids = ['a','b','c','d',]
    ids_gini_diversity_index(ids)
     > 0.75 # 完全に不純な状態

    ids = ['a','a','a','a',]
    ids_gini_diversity_index(ids)
     > 0.0 # 純粋な状態
    '''
    sum_count = len(ids)
    count_rates = [count/sum_count for _id,count in Counter(ids).items()]        
    return 1 - np.sum(np.array(count_rates)**2)




def ids_gini_coefficient(ids):
    '''
    所得の不均等などで利用される指標。ジニ係数。
    
    ids = ['a','b','c','d',]
    ids_gini_coefficient(ids)
     > 0.0 # 完全に出現率が均等

    ids = ['a','a','a','a',]
    ids_gini_coefficient(ids)
     > 0.0 # 完全に出現率が均等

    ids = [1,1,1,2]
    ids_gini_coefficient(ids)
     > 0.25
    ids = [1,1,1,2,10,10,10,20]
    ids_gini_coefficient(ids)
     > 0.25 # データを増やしても、不均等さが同じなら同じ値になる。

    ids = [4]
    ids_gini_coefficient(ids)
     > 0.0

    ids = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    ids_gini_coefficient(ids)
     > 0.45999999999999996 # 大きな出現率の格差

    ids = [0]*1000 + list(range(1,100))
    ids_gini_coefficient(ids)
     > 0.8999181073703366 # 100人のうち1人がほとんどを独占する場合。    
    '''
    n_ids = len(set(ids))
    if n_ids==0:
        return 1.0

    # ローレンツ曲線のx軸(idの累積比)とy軸(count_ratesの累積比)をvectorとして求める。    
    bin_width = 1 / n_ids
    x, y = get_lorenz_curve_xy(ids)    
            
    # ローレンツ曲線の下側面積 under_area を計算
    under_area = 0
    for i in range(len(y)):
        if i == 0:
            continue        
        under_area += bin_width*y[i-1] + (1/2)*bin_width*(y[i]-y[i-1])
    
    # gini係数を算出
    _gini = ((1/2) - under_area) / (1/2)
    
    return _gini


def get_lorenz_curve_xy(ids):
    sum_count = len(ids)
    n_ids = len(set(ids))

    count_rates = [(_id,count/sum_count) for _id,count in Counter(ids).items()]
    count_rates = sorted(count_rates, key=lambda x: x[1]) # 出現頻度の少ないid順に並び替える。
    # ローレンツ曲線のx軸(idの累積比)とy軸(count_ratesの累積比)をvectorとして求める。    
    bin_width = 1 / n_ids
    x, y = [0], [0]    
    for i,count_rate in enumerate(count_rates):
        x.append( x[-1] + bin_width )
        y.append( y[-1] + count_rate[1] )
    
    return x, y















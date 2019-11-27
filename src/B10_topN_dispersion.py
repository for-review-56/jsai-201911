#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
レコメンデーションの各手法を学習し、N-topごとの提案アイテムの多様性（分散）を表示する
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import sys
import random

from src.module import util

# --- models ---
from src.module.get_CF_varidation_arrays import get_CF_varidation_arrays
from src.module.Suprise_algo_wrapper import algo_wrapper
from src.module.two_way_aspect_model import two_way_aspect_model
from src.module.MF import MF
from src.module.ContentBoostedCF import ContentBoostedCF
from src.module.RankingList import RankingListMean, RankingListCnt, RankingListTotal
from surprise import SVD # SVD algorithm
from surprise import KNNBasic # A basic collaborative filtering algorithm.
from surprise import BaselineOnly # Algorithm predicting the baseline estimate for given user and item.
from surprise import NormalPredictor # Algorithm predicting a random rating

############################
# Set validable
if len(sys.argv) > 1:
    random_seed = int(sys.argv[1])
else:
    random_seed = 12345

# set_random_seed
random.seed(random_seed)
np.random.seed(random_seed)

# set validation parameteres
train_test_days = 14
n_hold = 1
topN = [5,10,20]

############################
# Read DataSet
# setting

# set directory to save resutl
DIR_output = 'pickle'
rating, user, movie = util.read_ml20m_data()

## set item_attributes
Genres = set()
for genres in movie['Genres']:
    for genre in genres.split('|'):
        Genres.add(genre)
Genres = sorted(list(Genres))

get_attributes = lambda genres: [int(Ge in genres.split('|')) for Ge in Genres]
item_attributes = {row['MovieID']:get_attributes(row['Genres']) for idx, row in movie.iterrows()}


####################################
# split train and test indexes
one_day_sec = 60 * 60 * 24
max_seconds = rating.Timestamp.max()

seps = sorted([max_seconds - i * one_day_sec * train_test_days for i in range(n_hold+2)])
sep_indexs = [(seps[i-1]<=rating.Timestamp)&(rating.Timestamp<seps[i]) for i in range(1, n_hold+2)]

####################################
# set dataset
user_ids = rating['UserID'].values[sep_indexs[0]]
item_ids = rating['MovieID'].values[sep_indexs[0]]
values   = rating['Rating'].values[sep_indexs[0]]


# define random model
class RandomModel:
    def __init__(self):
        pass
    def fit(self, *args, **kargs):
        pass
    def predict(self, user_ids, item_ids):
        return np.random.rand(len(user_ids))
        

# --- Set models ---
svd       = algo_wrapper(SVD())
userbased = algo_wrapper(KNNBasic(k=50, sim_options={'user_based':True, 'name':'cosine'}))
itembased = algo_wrapper(KNNBasic(k=50, sim_options={'user_based':False, 'name':'cosine'}))
baseline  = algo_wrapper(BaselineOnly())
randommodel = RandomModel()
mf_item_attributes = MF(n_latent_factor=100)




#my_contentbased = ContentBasedCF()
my_contentbased = MF(n_latent_factor=0)
my_contentboosted = ContentBoostedCF(pure_content_predictor=my_contentbased)

models = {
        #"userbased": userbased,
        #"itembased": itembased,
        #"baseline": baseline,
        "randommodel": randommodel,
        #"two_way_aspect_Z050": two_way_aspect_model(item_attributes=item_attributes, Z=50,),
        #"mf_item_attributes": mf_item_attributes,
        #"my_mf_100": MF(n_latent_factor=100),
        #"my_contentboosted": my_contentboosted,
        #"RankingListMean": RankingListMean(),
        #"RankingListCnt": RankingListCnt(),
        "RankingListTotal": RankingListTotal(),
        }

# --- varidation ---
def get_entropy(model_name, n_user_id=1000, seed=123):
    if model_name in ['mf_item_attributes', 'my_contentbased', 'my_contentboosted']:
        models[model_name].fit(
                user_ids, item_ids, values,
                user_attributes=dict(), item_attributes=item_attributes
                )
    else:
        models[model_name].fit(
                user_ids, item_ids, values,
                )
    
    list_item_ids = list(set(item_ids))
    n_item_ids   = len(list_item_ids)
    result_dict = {ntop:[] for ntop in topN}
    np.random.seed(seed)
    random_user_ids = np.random.choice(list(set(user_ids)), size=n_user_id)
    for user_id in random_user_ids:
        if model_name in ['mf_item_attributes', 'my_contentbased', 'my_contentboosted']:        
            predictied = models[model_name].predict(
                    [user_id]*n_item_ids, list_item_ids, item_attributes=item_attributes
                    )
        else:
            predictied = models[model_name].predict(
                    [user_id]*n_item_ids, list_item_ids
                    )            
        sort_func = lambda x: x[1]
        sorted_item_ids = sorted(zip(list_item_ids, predictied), key=sort_func, reverse=True)
        for ntop in result_dict:
            result_dict[ntop] += [i[0] for i in sorted_item_ids[:ntop]]

    entropy_dict = {k:_entropy(v) for k,v in result_dict.items()}
    
    return entropy_dict

from scipy.stats import entropy
def _entropy(list_of_ids):
    """
    list_of_ids = [1,1,2,3,2]
    """
    set_ = list(set(list_of_ids))
    prob = [list_of_ids.count(s) / len(list_of_ids) for s in set_]
    return entropy(prob)

    
if __name__=='__main__':
    # 並列処理は一つ一つの処理時間を揃えた方が早いので以下のように記述する。
    results = []
    for model_name in list(models.keys()):
        print("=== START {} ===".format(model_name))
        result = get_entropy(model_name)
        result['model_name'] = model_name
        results.append(result)
    
    import pandas as pd
    pd.DataFrame(results).to_csv('output/B01_entropy.txt', index=False)

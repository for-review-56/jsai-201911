#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データを指定したn_holdに分割し、modelsごとにテストを行う。
それぞれのテスト結果をpickleファイルにして出力する。
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import pickle
import os
import sys
import random

from src.module import util

# --- models ---
from src.module.get_CF_varidation_arrays import get_CF_varidation_arrays
from src.module.Suprise_algo_wrapper import algo_wrapper
from src.module.two_way_aspect_model import two_way_aspect_model
from src.module.MF import MF
from src.module.ContentBoostedCF import ContentBoostedCF
from src.module.RankingList import RankingListMean, RankingListTotal, RankingListCnt
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
train_test_days = 30
n_hold = 10
topN = [5,10,20]

############################
# Read DataSet
# setting

# set directory to save resutl
DIR_output = 'pickle'

print('レコメンデーションのデータを読み込みます')
rating, user, movie = util.read_netflix_data()
#rating, user, movie = util.read_ml20m_data()
#rating, user, movie = util.read_mllatest_data()

## set item_attributes
Genres = set()
if 'Genres' in movie.columns:
    print('movieのGenres情報を整理します')
    for genres in movie['Genres']:
        for genre in genres.split('|'):
            Genres.add(genre)
    Genres = sorted(list(Genres))
    get_attributes = lambda genres: [int(Ge in genres.split('|')) for Ge in Genres]
    item_attributes = {row['MovieID']:get_attributes(row['Genres']) for idx, row in movie.iterrows()}
else:
    print('movieのGenres情報がありません。このまま処理をつづけます')

####################################
# set dataset
user_ids = rating['UserID'].values
item_ids = rating['MovieID'].values
values   = rating['Rating'].values


####################################
# split train and test indexes
one_day_sec = 60 * 60 * 24
max_seconds = rating.Timestamp.max()

seps = sorted([max_seconds - i * one_day_sec * train_test_days for i in range(n_hold+2)])
sep_indexs = [(seps[i-1]<=rating.Timestamp)&(rating.Timestamp<seps[i]) for i in range(1, n_hold+2)]


# --- Set models ---
svd       = algo_wrapper(SVD())
userbased = algo_wrapper(KNNBasic(k=50, sim_options={'user_based':True, 'name':'cosine'}))
itembased = algo_wrapper(KNNBasic(k=50, sim_options={'user_based':False, 'name':'cosine'}))
randommodel = algo_wrapper(NormalPredictor())
mf_item_attributes = MF(n_latent_factor=100)


#my_contentbased = ContentBasedCF()
my_contentbased = MF(n_latent_factor=0)
my_contentboosted = ContentBoostedCF(pure_content_predictor=my_contentbased)

models = {
        #"svd": svd,
        "userbased": userbased,
        "itembased": itembased,
        "randommodel": randommodel,
        #"two_way_aspect_Z050": two_way_aspect_model(item_attributes=item_attributes, Z=50,),
        #"two_way_aspect_Z100": two_way_aspect_model(item_attributes=item_attributes, Z=100,),
        #"two_way_aspect_Z200": two_way_aspect_model(item_attributes=item_attributes, Z=200,),
        #"two_way_aspect_Z400": two_way_aspect_model(item_attributes=item_attributes, Z=400,),
        "mf_item_attributes": mf_item_attributes,
        "my_mf_010": MF(n_latent_factor=10),
        "my_mf_050": MF(n_latent_factor=50),
        "my_mf_100": MF(n_latent_factor=100),
        "my_mf_200": MF(n_latent_factor=200),
        #"my_contentbased": my_contentbased,
        #"my_contentboosted": my_contentboosted,
        #"RankingListMean": RankingListMean(),
        "RankingListTotal": RankingListTotal(),
        #"RankingListCnt": RankingListCnt(),
        }

# --- varidation ---
n_random_selected_item_ids = 1000

def get_varidation_arrays(arg_dict):
    """
    example of arg_dict:
        arg_dict = {
                'model_name': 'my_contentbased',
                'hold': 2
                }
    """
    print("START on {}".format(arg_dict))
    model_name = arg_dict['model_name']
    i = arg_dict['hold']

    train_user_ids = user_ids[sep_indexs[i]]
    train_item_ids = item_ids[sep_indexs[i]]
    train_values   = values[sep_indexs[i]]
    test_user_ids = user_ids[sep_indexs[i+1]]
    test_item_ids = item_ids[sep_indexs[i+1]]
    test_values   = values[sep_indexs[i+1]]
    if model_name in ['mf_item_attributes', 'my_contentbased', 'my_contentboosted']:
        validation_arrays =  get_CF_varidation_arrays(
                    train_user_ids, train_item_ids, train_values,
                    test_user_ids, test_item_ids, test_values,
                    models[model_name],
                    n_random_selected_item_ids=n_random_selected_item_ids,
                    remove_still_interaction_from_test=True,
                    random_seed=random_seed,
                    topN=topN,
                    user_attributes=dict(), item_attributes=item_attributes
                )
    else:
        validation_arrays =  get_CF_varidation_arrays(
                    train_user_ids, train_item_ids, train_values,
                    test_user_ids, test_item_ids, test_values,
                    models[model_name],
                    n_random_selected_item_ids=n_random_selected_item_ids,
                    remove_still_interaction_from_test=True,
                    random_seed=random_seed,
                    topN=topN,
                )
    file_name = os.path.join(DIR_output, 'validation__model_name={}__random_seed={}__train_test_days={}__topN={}__hold={}.pickle'.format(model_name, random_seed, train_test_days, topN, i))
    pickle.dump(validation_arrays, open(file_name, 'wb'))
    print("END on {}".format(arg_dict))



if __name__=='__main__':
    import multiprocessing as mp
    pool = mp.Pool()

    # 並列処理は一つ一つの処理時間を揃えた方が早いので以下のように記述する。
    for model_name in list(models.keys()):
        print("==== START {} ====".format(model_name))
        list_of_arg_dict = list()
        for hold in range(n_hold):
            list_of_arg_dict.append({'model_name':model_name, 'hold':hold})
        results = pool.map(get_varidation_arrays, list_of_arg_dict)

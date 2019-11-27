#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A01_get_pickle_for_validation.py の結果、生成された *.pickleファイルを参照して、評価結果をtsvファイルで出力する。
"""

import os
import pickle
import re
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product

# head, tail の分岐点
Q=[80,85,90,95]

def main():
    DIR = 'pickle'
    
    pickle_file_list = [os.path.join(DIR, p) for p in os.listdir(DIR) if re.match('^.*.pickle', p)]
    
    frame = []
    for pickle_file in pickle_file_list:
        for q in Q:
            frame.append(parse(pickle_file, q))
    
    result = pd.concat(frame, ignore_index=True)
    result.to_csv('output/A02_validation_result_Q={}.tsv'.format(Q), index=False, sep='\t')


def parse(pickle_file, q):#追加
    """
    pickle_file = pickle_file_list[0]
    """
    model_name, hold = 'nothing', 'nothing'
    for file_name_part in pickle_file.split('__'):
        if re.match('model_name=', file_name_part):
            model_name = file_name_part.split('=')[1]
        if re.match('train_test_days=', file_name_part):
            train_test_days = file_name_part.split('=')[1]
        if re.match('hold=', file_name_part):
            hold = file_name_part.split('=')[1].replace('.pickle', '')

    vr = pickle.load(open(pickle_file, 'br'))
    random_seed = vr['VALIDATION_PARAMETERs']['random_seed']
    topN = vr['VALIDATION_PARAMETERs']['topN']
    remove_still_interaction_from_test = vr['VALIDATION_PARAMETERs']['remove_still_interaction_from_test']
        
    # --- categorize items and users for Validation ---    
    cnt_dict = Counter(vr['train_item_ids'])
    head_cnt = np.percentile(list(cnt_dict.values()), q=q)#追加
    head_item_ids = [k for k,cnt in cnt_dict.items() if cnt >= head_cnt]
    tail_item_ids = [k for k,cnt in cnt_dict.items() if cnt <  head_cnt]
    #tail_item_ids, head_item_ids = cluster_ids_by_freaquence(vr['train_item_ids'], n_cluster=2)
    no_trained_item_ids  = np.unique(vr['test_item_ids'][~np.in1d(vr['test_item_ids'], np.unique(vr['train_item_ids']))])

    cnt_dict = Counter(vr['train_user_ids'])
    head_cnt = np.percentile(list(cnt_dict.values()), q=q)#追加
    head_user_ids = [k for k,cnt in cnt_dict.items() if cnt >= head_cnt]
    tail_user_ids = [k for k,cnt in cnt_dict.items() if cnt <  head_cnt]
    #tail_user_ids, head_user_ids = cluster_ids_by_freaquence(vr['train_user_ids'], n_cluster=2)
    no_trained_user_ids  = np.unique(vr['test_user_ids'][~np.in1d(vr['test_user_ids'], np.unique(vr['train_user_ids']))])


    # --- set the numbers of id ---
    def count_unique_in(array, items):
        """
        arrayの中に含まれるitemsの数とユニーク数を返却する。
        array = np.array([1,3,3,5,5,5,7])
        items = np.array([3,7,13])
        
        return: (3, 2)
        """
        array_in = array[np.in1d(array, items)]
        return array_in.size, len(set(array_in))
    
    # --- Validation on all items and user ---
    vr['test_values'] = np.array(vr['test_values'])
    vr['predicted_values'] = np.array(vr['predicted_values'])
    vr['test_good_hit_result'] = {k:np.array(v) for k,v in vr['test_good_hit_result'].items()}
        
    # --- Validation on [head, tail, no_train]
    dict_user_ids = dict(head=head_user_ids, tail=tail_user_ids, no_tarin=no_trained_user_ids, all=None)
    dict_item_ids = dict(head=head_item_ids, tail=tail_item_ids, no_train=no_trained_item_ids, all=None)
    result_metrics = list()
    for user_key, item_key in product(dict_user_ids, dict_item_ids):
        metrics = common_process(vr, dict_user_ids[user_key], dict_item_ids[item_key])
        metrics['segment'] = "u={}&i={}".format(user_key, item_key)
        result_metrics.append(metrics)

    # --- RETURN ---
    def wrapper(dict_metric):
        return convert_df(
                        dict_metric
                        ,model_name=model_name
                        ,random_seed=random_seed
                        ,train_test_days=train_test_days
                        ,topN=topN
                        ,remove_still_interaction_from_test=remove_still_interaction_from_test
                        ,hold=hold
                        ,q=q #追加
                          )
    frame = []
    for metrics in result_metrics:
        df_ = wrapper(metrics)
        frame.append(df_)
    
    return pd.concat(frame, axis=0, ignore_index=True)


def get_metrix(test_values, predicted_values, test_good_hit_result):
    errores = np.array(predicted_values) - np.array(test_values)
    MAE = np.mean(np.abs(errores))
    RMSE = np.sqrt(np.mean(np.square(errores)))

    return_dict = dict(MAE=MAE, RMSE=RMSE)
    
    for topN, hit_result in test_good_hit_result.items():
        if np.array(hit_result).size != 0:
            recall    = sum(np.array(hit_result)) / np.array(hit_result).size
            precision = recall / topN
        else:
            recall, precision = np.nan, np.nan
        return_dict['topN={}_recall'.format('%02d'%topN)] = recall
        return_dict['topN={}_precision'.format('%02d'%topN)] = precision
        
    return return_dict
    

def convert_df(dict_metrix, **keyargs):
    """
    Arguments:
        dict_metrix [dictionary]:
            the returned values from get_metrix()
            ex)
                {'MAE': 0.7656306513941742,
                 'RMSE': 0.9728788696895713,
                 'precision': 0.0058041648205582625,
                 'recall': 0.05804164820558263}
        **keyargs:
            the {column: value} you want to put in df
    """
    df_ = pd.DataFrame([dict_metrix])
    for col, val in keyargs.items():
        try:
            df_.loc[0, col] = val
        except:
            df_.loc[0, col] = str(val)            
    return df_

def common_process(vr, user_ids=None, item_ids=None):
    bo_index     = np.array([True]*vr['test_item_ids'].shape[0])
    bo_hit_index = np.array([True]*vr['test_good_item_ids'].shape[0])
    if user_ids is not None:
        bo_index     &= np.in1d(vr['test_user_ids'], user_ids)
        bo_hit_index &= np.in1d(vr['test_good_user_ids'], user_ids)
    if item_ids is not None:
        bo_index     &= np.in1d(vr['test_item_ids'], item_ids)
        bo_hit_index &= np.in1d(vr['test_good_item_ids'], item_ids)
        
    test_values, predicted_values = vr['test_values'][bo_index], vr['predicted_values'][bo_index]
    test_good_hit_result = {k:v[bo_hit_index] for k,v in vr['test_good_hit_result'].items()}
    
    return_df = get_metrix(test_values, predicted_values, test_good_hit_result)
    
    return_df['test_sample_size']     = bo_index.sum()
    return_df['test_hit_sample_size'] = bo_hit_index.sum()

    return return_df

def cluster_ids_by_freaquence(_ids, n_cluster=2):
    """
    get a id-list abscendly ordered by the freazuence in _ids.
    
    EXAMPLE
    -------------
    _ids = [1,1,1,1,2,2,3,3,3,3,3,4]
    n_cluster = 2
    cluster_ids_by_freaquence(_ids, n_cluster)
     > [[4, 2, 1], [3]]
    n_cluster = 3
    cluster_ids_by_freaquence(_ids, n_cluster)
     > [[4, 2, 1], [3], []]

    _ids = [1,1,1,2,2,2,3,3,3,4,5,6]
    n_cluster = 3
    cluster_ids_by_freaquence(_ids, n_cluster)
     > [[4, 5, 6, 1], [2], [3]]
    """
    _id_rates = [(_id, cnt/len(_ids)) for _id,cnt in Counter(_ids).items()]
    _id_rates = sorted(_id_rates, key=lambda x: x[1])
    
    accum_rate = 0
    result = []
    for i in range(n_cluster):
        thresh_hold = (i+1)/n_cluster
        _result = []
        for _id,rate in _id_rates:
            _result.append(_id)
            accum_rate += rate
            if accum_rate > thresh_hold:
                break            
        _id_rates = [_id_rate for _id_rate in _id_rates if _id_rate[0] not in _result]
        result.append(_result)
    return result
            
    

if __name__ == '__main__':
    main()


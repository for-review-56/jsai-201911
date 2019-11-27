#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RankingList
"""

import numpy as np
import sys
from . import util

'''
import numpy as np
import sys
from src.module import util
'''

class RankingListMean:
    def __init__(self, random_seed=None):
        self.RankingList = RankingList()
    def fit(self, *args):
        return self.RankingList.fit(*args)
    def predict(self, *args):
        return self.RankingList.predict(*args, predict_type='mean')

class RankingListTotal:
    def __init__(self, random_seed=None):
        self.RankingList = RankingList()
    def fit(self, *args):
        return self.RankingList.fit(*args)
    def predict(self, *args):
        return self.RankingList.predict(*args, predict_type='total')

class RankingListCnt:
    def __init__(self, random_seed=None):
        self.RankingList = RankingList()
    def fit(self, *args):
        return self.RankingList.fit(*args)
    def predict(self, *args):
        return self.RankingList.predict(*args, predict_type='cnt')


class RankingList:
    def __init__(self, random_seed=None):
        """
        Arguments
        --------------------
            - random_seed [int]: 
                random seed to set in np.random.seed()

        Example
        --------------------
        user_ids = [1,1,1,2,2,3]
        item_ids = [0,1,2,0,1,2]
        ratings  = [2,2,5,2,3,1]
        self = RankingList()
        self.fit(user_ids, item_ids, ratings)
        self.predict(user_ids, item_ids)
        """
        
        # set random_seed
        if random_seed:
            np.random.seed(random_seed)

        
    def fit(self, user_ids, item_ids, ratings):            
        """
        Arguments
        --------------------
            - user_ids [array-like-object]: 
                the array of user id.
            - item_ids [array-like-object]: 
                the array of item id.
            - ratings [array-like-object]: 
                the array of rating.
        
        """
        # Set up before fit
        set_item_ids = set(item_ids)
        dict_score = {
                item_id:{'total':0, 'cnt':0,} 
                for item_id in set_item_ids
                }
        for i, r in zip(item_ids, ratings):
            dict_score[i]['total'] += r
            dict_score[i]['cnt'] += 1
        for ds in dict_score.values():
            ds['mean'] = ds['total'] / ds['cnt']
        
        # set up
        self.dict_score = dict_score
        
        return self
    
    def predict(self, user_ids, item_ids, predict_type='mean'):
        """
        Arguments:
            user_ids [array-like object]:
                pass
            item_ids [array-like object]:
                pass
            predict_type [str]:
                'mean' or 'total' or 'cnt'
        """        
        # predict
        results = [self._predict(i, predict_type) for i in item_ids]
        return np.array(results)
    
    def _predict(self, i, predict_type):
        if i in self.dict_score:
            return self.dict_score[i][predict_type]
        else:
            return 0
    
    

if __name__ == 'How to use it.':
    # Usage
    user_ids = [1,1,1,2,2,3,3]
    item_ids = [0,1,2,0,1,2,0]
    ratings  = [2,2,5,2,3,1,1]

    RL = RankingList()
    RL.fit(user_ids, item_ids, ratings)
    RL.predict(user_ids, item_ids)
    
    RLT = RankingListTotal()
    RLT.fit(user_ids, item_ids, ratings)
    RLT.predict(user_ids, item_ids)
    
    RLM = RankingListMean()
    RLM.fit(user_ids, item_ids, ratings)
    RLM.predict(user_ids, item_ids)
    
    RLC = RankingListCnt()
    RLC.fit(user_ids, item_ids, ratings)
    RLC.predict(user_ids, item_ids)

    RLC.predict(['unfitted_user'], ['unfitted_item'])

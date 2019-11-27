#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module was on:
    [Hilmi 2008: A Random Walk Method for Alleviating the Sparsity Problem in Collaborative Filtering]
"""
from copy import deepcopy
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics.pairwise import cosine_similarity


""" プログラミング用
user_ids = user_ids[sep_indexs[0]]
item_ids = item_ids[sep_indexs[0]]
values   = values[sep_indexs[0]]
alpha=0.80
beta=0.99
self = RandomWalkCF()
"""

class RandomWalkCF:
    def __init__(self):
        """
        This is Simple Random Walk Collaborative Filtering.
        """
        pass
            
    def fit(self
            ,user_ids, item_ids, values
            ,alpha=0.80, beta=0.99
            ,DEBUG=False
            ):
        """
        Argument:
            user_ids [np.array; (n_samples,)]:
                the id must be int.
            item_ids [np.array; (n_samples,)]:
                the id must be int.
            values [np.array; (n_samples, )]:
                the values resulted from iteractions between user_id and item_id.
            alpha [float, 0.~1.]:
                the probability to continue random walk at each step.
            beta [float, 0.~1.]:
                the probability to move from item to item as random walk.
                with probability (1-beta), jumpts to arbitrary item.
                                
        """
        self.mean_value = sum(values)/len(values)
        # get item-item similar matrix as S from the user-item rating matrix R
        Rating = pd.DataFrame({'user_ids':user_ids, 'item_ids':item_ids, 'values':values})
        R = pd.pivot_table(Rating, 'values', 'user_ids', 'item_ids', aggfunc='max').fillna(0)

        #S = adjusted_cosine_similarity(R)
        S = cosine_similarity(R.T)
        for i in range(S.shape[0]):
            S[i,i] = 0

        S = scale_S_as_probabilities(S)

        m = S.shape[0]

        P = list()
        for i in range(m):
            # i = 0 
            sum_ = S[i, :].sum()
            if sum_!=0:
                prob = lambda j : (beta * S[i,j] / sum_) + (1 - beta) / m
            else:
                prob = lambda j : 0 + 1 / m
            map_ge = map(prob, range(m))
            P.append(list(map_ge)) 

        P = pd.DataFrame(P)
                            
        hat_P = np.dot(alpha * P, np.linalg.inv(np.eye(m) - alpha * P))
        hat_R = np.dot(R, hat_P)
        self.unscale_R = deepcopy(hat_R)
        hat_R = self._scale(hat_R, R)
        
        self.hat_R = pd.DataFrame(hat_R, index=R.index, columns=R.columns)
                        

    def predict(self, user_ids, item_ids):
        """
        Argument:
            user_ids [np.array; (n_samples,)]:
                the id must be (0 ~ n_user).
            item_ids [np.array; (n_samples,)]:
                the id must be (0 ~ n_item).
        """        
        return_list = [self._predict(user_id, item_id) for user_id, item_id in zip(user_ids, item_ids)]
        return np.array(return_list)                

    def _scale(self, hat_R, R):
        """
        In our implementation,we linearly scaled up each row of values such that the maximum of each row corresponds to 5.
        """
        _min, _max = R[R>0].min().min(), R[R>0].max().max()
        for i in range(hat_R.shape[0]):
            hat_r = hat_R[i, :]
            hat_r_scaled_0_1     = (hat_r - hat_r.min()) / (hat_r.max() - hat_r.min())
            hat_r_scaled_min_max = hat_r_scaled_0_1 * (_max - _min) + _min
            hat_R[i, :] = hat_r_scaled_min_max
        return hat_R
    
    def _predict(self, user_id, item_id):
        bo_user = user_id in self.hat_R.index
        bo_item = item_id in self.hat_R.columns
        if bo_user and bo_item:
            return self.hat_R.loc[user_id, item_id]        
        else:
            return self.mean_value
        '''
        elif bo_user:
            #return self.hat_R.loc[user_id, :].mean()
            return self.hat_R.loc[user_id, item_id]
        elif bo_item:
            #return self.hat_R.loc[:, item_id].mean()
            return self.hat_R.loc[user_id, item_id]
        else:
            return self.hat_R.values.mean()
        '''
        

def adjusted_cosine_similarity(R):
    m = R.shape[1]
    R_np = R.values
    
    no_0_mean = lambda array: array[array>0].mean()
    R_np_mean = np.apply_along_axis(no_0_mean, axis=1, arr=R_np)[:, None]
    
    # R_np - R_np_mean ignoring 0
    diff_R = R_np - R_np_mean
    diff_R[R_np==0] = 0
    
    
    def S_ij(ij):
        i,j = ij

        if i == j: # 普通の類似行列なら1だが、これは本質的には遷移確率変数なので対角は0の方が良い。
            return i,j,0

        U = (R_np[:, [i,j]]>0).all(axis=1)
        U = np.array(range(R_np.shape[0]))[U]
        
        if U.size==0:
            return i,j,0

        numerator   = 0
        denominator01, denominator02 = 0, 0
        
        for u in U:            
            numerator += diff_R[u,i] * diff_R[u,j]
            denominator01 += diff_R[u,i]**2
            denominator02 += diff_R[u,j]**2

        denominator = (np.sqrt(denominator01)*np.sqrt(denominator02))
        sim = numerator / denominator if denominator>0 else 0
        return i,j,sim

    S = np.zeros(shape=(m, m))
    comb = itertools.combinations_with_replacement(range(m), 2)
    
    for i,j,sim in map(S_ij, comb):
        S[i,j], S[j,i] = sim, sim
    
    return S

def scale_S_as_probabilities(S):    
    ''' 論文より抜粋
    Common approaches produce symmetric 
    similarity matrices; but we destroy symmetry while computing
    transition probabilities by normalizing the sum of each
    row to 1
    '''
    # 上記の抜粋に基づいて、similarityを変換する。    
    S -= S.min(axis=1)[:,None] 
    S /= S.sum(axis=1)[:,None]
    S[np.isnan(S)] = 0
    
    return S

if __name__ == '__main__':
    n_user = 50
    n_item = 100
    # Answer R
    R = np.random.choice(range(1,6), size=(n_user,n_item))
    # Usage
    user_ids = np.random.choice(range(n_user), size=500)
    item_ids = np.random.choice(range(n_item), size=500)
    values   = np.array([R[u,i] for u,i in zip(user_ids, item_ids)])

    # set simple model    
    rwcf = RandomWalkCF()
    rwcf.fit(user_ids, item_ids, values, beta=0.99)    
    predicted = rwcf.predict(user_ids, item_ids)
    random_predicted = np.random.choice(range(1,6), size=500)

    # insample test
    print(np.abs(predicted - values).mean())
    print(np.abs(random_predicted - values).mean())

    # all test
    from itertools import product
    u_i = list(product(range(n_user), range(n_item)))
    user_ids = [u_i[_][0] for _ in range(len(u_i))]
    item_ids = [u_i[_][1] for _ in range(len(u_i))]
    values   = [R[u,i] for u,i in u_i]
    
    predicted = rwcf.predict(user_ids, item_ids)
    random_predicted = np.random.choice(range(1,6), size=len(u_i))

    print(np.abs(predicted - values).mean())
    print(np.abs(random_predicted - values).mean())
    
    # new id
    rwcf.predict([999999],[1])
    rwcf.predict([1],[999999])    
    rwcf.predict([999999],[99999999])
    rwcf.predict([1,2,2],[1,1,999])

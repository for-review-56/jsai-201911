#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:11:48 2018

@author: takumi_uchida
"""

import numpy as np
from scipy import spatial
from src.module import util
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedCF:
    
    def __init__(self, knn=500):
        """
        Arguments:
            knn [int]:
                the number of similar id which is used in predicting.
        """
        self.knn = knn

    def fit(self, user_ids, item_ids, ratings, item_attributes, user_attributes='NotUse'):
        """                
        Arguments:
            - user_ids [array-like-object]: 
                the array of user id.
            - item_ids [array-like-object]: 
                the array of item id.
            - ratings [array-like-object]: 
                the array of rating.
            - item_attributes [dictinary]:
                dictinary which key is item_id and value is vector of item attributes.
                ex) {'item00' : [0,1,0], 'item01': [.5,0,.5]]}
        """
        
        # set id transformer
        user_ids_transformer = util.id_transformer()
        item_ids_transformer = util.id_transformer()        
        self.user_ids = user_ids_transformer.fit_transform(user_ids)
        item_ids_transformer.fit(list(item_attributes.keys()))
        self.item_ids = item_ids_transformer.transform(item_ids)        
        self.user_ids_transformer = user_ids_transformer
        self.item_ids_transformer = item_ids_transformer
        self.mean_ratings = np.mean(ratings)
        
        # set rating
        self.rating = {(u,i):r for u,i,r in zip(self.user_ids, self.item_ids, ratings)}
        
        # set item-attributes matrix
        dim_attribute = len(list(item_attributes.values())[0])
        n_item_ids    = len(item_attributes)
        self.item_attributes = np.zeros(shape=(n_item_ids, dim_attribute))
        for i, vec in item_attributes.items():
            _i = item_ids_transformer.transform([i], unknown=None)[0]
            if _i is not None:
                self.item_attributes[_i, :] = vec
        
        # get item similarity matrix
        item_similarity_matrix = cosine_similarity(self.item_attributes, self.item_attributes)
        self.knn_id_sim = {}    
        for i in range(item_similarity_matrix.shape[0]):
            sim = item_similarity_matrix[i,:]
            top_ids = sim.argsort()[:-(self.knn+1):-1]
            self.knn_id_sim[i] = {'top_ids':top_ids, 'sims':sim[top_ids]}
        
        
        return self
        

    def predict(self, user_ids, item_ids, item_attributes={}, user_attributes='NotUse'):
        """
        Arguments:
            user_ids [array-like object]:
                pass
            item_ids [array-like object]:
                pass
            item_attributes [dict]:
                pass
        """
        tf_us = self.user_ids_transformer.transform(user_ids, unknown=None)
        tf_is = self.item_ids_transformer.transform(item_ids, unknown=None)
        item_attrs = [item_attributes.get(item_id, None) for item_id in item_ids]
        cash_predicted = {}
        predicted_list = []
        for tf_u, tf_i, item_attr in zip(tf_us, tf_is, item_attrs):
            key = str(tf_u)+str(tf_i)+str(item_attr)
            if key in cash_predicted:
                predicted = cash_predicted[key]
            else:
                predicted = self._predict(tf_u, tf_i, item_attr)
                cash_predicted[key] = predicted
            predicted_list.append(predicted)
        
        return_array = np.array(predicted_list)
        return_array[np.isnan(return_array)] = self.mean_ratings
        return return_array


    def _predict(self, u, i, attributes=None):
        """
        Arguments:
            u [int]:
                a transformed user_id.
            i [int]:
                a transformed item_id.
            attributes [array like object]:
                a vector of item attributes.
        """
        if (u is None) and ((i is None) and (attributes is None)):
            return self.mean_ratings   
        if (u is None) or ((i is None) and (attributes is None)):
            if (u is None):
                keys = [u_i for u_i in self.rating if u_i[1]==i]
            if ((i is None) and (attributes is None)):
                keys = [u_i for u_i in self.rating if u_i[0]==u]
            if keys:
                return np.mean([self.rating.get(key) for key in keys])
            else:
                return self.mean_ratings
        
        # if both of u and i is not None.
        if i in self.knn_id_sim:
            knn_item_ids, similarities = self.knn_id_sim[i]['top_ids'], self.knn_id_sim[i]['sims']
        else:
            if attributes is not None:
                knn_item_ids, similarities = self._get_similar_item_ids(attributes)
            else:
                raise 'item_id was not fitted, and attributes is None.'
            
        _rating = np.array([self.rating.get((u,_i), np.nan) for _i in knn_item_ids])
        bo_index = ~np.isnan(_rating)
        if any(bo_index):
            sum_similarities = similarities[bo_index].sum() if similarities[bo_index].sum() > 0 else 1
            predicted = (_rating[bo_index] * similarities[bo_index]).sum() / sum_similarities
        else:
            keys = [u_i for u_i in self.rating if u_i[1]==i]
            predicted = np.mean([self.rating.get(key) for key in keys])
        return predicted
            

    def _get_similar_item_ids(self, attributes):
        """
        Arguments:
            attributes [array like object]:
                a vector of item attributes.
                
        Example:
            attributes = [0,1,1]
        """
        # i = 2
        func1d = lambda array: my_cosine_similarity(array, attributes)
        sims = np.apply_along_axis(func1d, axis=1, arr=self.item_attributes)
        sims[np.isnan(sims)] = 0. # fill np.nan with 0.
        
        # filter top self.knn similarity score
        top_ids = sims.argsort()[:-(self.knn+1):-1]
        
        return top_ids, sims[top_ids]
    
def my_cosine_similarity(array1, array2):
    sim = 1 - spatial.distance.cosine(array1, array2)
    return sim

    
    


if __name__ == '__main__':
    # Usage
    ## 下記のデータは明らかに、item_attribute = [負の影響, 影響なし, 正の影響]になっている。
    """
    # それなりの大量データでの結果
    import numpy as np
    user_ids = np.random.choice(range(100), size=1000)
    item_ids = np.random.choice(range(500), size=1000)
    ratings  = np.random.choice(range(1,6), size=1000)
    item_attributes = {i:np.random.choice([0,1], size=18) for i in range(5000)}
    knn = 500
    CBCF = ContentBasedCF(knn)
    CBCF.fit(user_ids, item_ids, ratings, item_attributes)

    user_ids = np.random.choice(range(100), size=1000)
    item_ids = np.random.choice(range(500), size=1000)

    CBCF.predict(user_ids, item_ids, item_attributes) 
    self = CBCF
    """
    # 用意したデータセットでは、明らかに、itemの属性で[正の影響、影響なし、負の影響]になるようになっている
    user_ids = [1,1,1,1,1,5,5,5,5,5]
    item_ids = [1,2,3,4,5,1,2,3,4,5]
    ratings  = [1,3,3,5,5,1,3,3,5,5]
    item_attributes = {
            1:[0,1,1],
            2:[1,0,1],
            3:[1,1,1],
            4:[1,0,0],
            5:[1,1,0],
            90:[1,0,0],
            91:[0,1,0],
            92:[0,0,1],
            }
    
    knn = 100
    CBCF = ContentBasedCF(knn)
    CBCF.fit(user_ids, item_ids, ratings, item_attributes)
    CBCF.predict(user_ids, item_ids, item_attributes)
    self = CBCF
    
    # outsample
    CBCF.predict([1], item_ids=[99], item_attributes={99:[1,0,0]})
    CBCF.predict([1], item_ids=[90], item_attributes={})
    CBCF.predict([1], item_ids=[99], item_attributes={99:[0,1,0]})
    CBCF.predict([1], item_ids=[91], item_attributes={})
    CBCF.predict([1], item_ids=[99], item_attributes={99:[0,0,1]})
    CBCF.predict([1], item_ids=[92], item_attributes={})

    CBCF.predict([99], item_ids=[1], item_attributes={})
    CBCF.predict([99], item_ids=[2], item_attributes={})
    CBCF.predict([99], item_ids=[3], item_attributes={})
    CBCF.predict([99], item_ids=[4], item_attributes={})
    CBCF.predict([99], item_ids=[5], item_attributes={})
    CBCF.predict([99], item_ids=[90], item_attributes={})# 平均になるしゃあない（item-basedだから未知のuserによわい)
    CBCF.predict([99], item_ids=[91], item_attributes={})# 平均になるしゃあない（item-basedだから未知のuserによわい)
    CBCF.predict([99], item_ids=[92], item_attributes={})# 平均になるしゃあない（item-basedだから未知のuserによわい)


    CBCF.predict([55], item_ids=[99], item_attributes={99:[0,1,1]})# 未知のユーザーは予測が平均値になる。
    
    
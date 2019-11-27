#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix Factorization which does not learns only UI-matrix but also attributes.
"""


import numpy as np
import sys
from . import util
'''
import numpy as np
import sys
from src.module import util
'''


class MF:
    def __init__(self, n_latent_factor=200, learning_rate=0.005, 
                 regularization_weight=0.02, n_epochs=20, 
                 global_bias=True, id_bias=True,
                 verbose=False, random_seed=None):
        """
        Collabolative Filtering so called Matrix Factorization.
        
        Arguments:
            - n_latent_factor [int]: 
                number of latent dimensions
            - learning_rate [float]: 
                learning rate
            - regularization_weight [float]: 
                regularization parameter
            - global_bias [True/False]:
                set bias of global.
            - id_bias [True/False]:
                set bias of user_id, item_id.
            - n_epochs [int]: 
                number of epoch of train(SGD)
            - random_seed [int]: 
                random seed to set in np.random.seed()
        """
        
        # set random_seed
        if random_seed:
            np.random.seed(random_seed)

        self.n_latent_factor = n_latent_factor        
        self.learning_rate = learning_rate
        self.regularization_weight = regularization_weight
        self.global_bias = global_bias
        self.id_bias = id_bias
        self.n_epochs = n_epochs
        self.verbose = verbose            
        
    def fit(self, user_ids, item_ids, ratings, 
            user_attributes=None, item_attributes=None):            
        """
        Arguments:
            - user_ids [array-like-object]: 
                the array of user id.
            - item_ids [array-like-object]: 
                the array of item id.
            - ratings [array-like-object]: 
                the array of rating.
            - user_attributes [dictinary]:
                dictinary which key is user_id and value is vector of user attributes.
                if None, doesn't train on the attributes.
                ex) {'user00' : [0,1,0], 'user01': [.5,0,.5]]}
            - item_attributes [dictinary]:
                dictinary which key is item_id and value is vector of item attributes.
                if None, doesn't train on the attributes.
                ex) {'item00' : [0,1,0], 'item01': [.5,0,.5]]}
        """
        # Set up before fit
        self._fit_setup(
                user_ids, item_ids, ratings, 
                user_attributes, item_attributes
                )
        
        # Initialize coefficents of attributes.
        if user_attributes:
            self.a_u = np.zeros(self.n_dim_user_attributes, np.double)
        if item_attributes:
            self.a_i = np.zeros(self.n_dim_item_attributes, np.double)

        # Initialize the biases
        if self.global_bias:
            self.b = np.mean([r[2] for r in self.R])

        if self.id_bias:
            self.b_u = np.zeros(self.num_users, np.double)
            self.b_i = np.zeros(self.num_items, np.double)
        
        # Initialize user and item latent feature matrice
        if self.n_latent_factor:
            self.P = np.random.normal(0, scale=.1, size=(self.num_users, self.n_latent_factor))
            self.Q = np.random.normal(0, scale=.1, size=(self.num_items, self.n_latent_factor))

        # Perform stochastic gradient descent for number of iterations
        before_mse = sys.maxsize
        stop_cnt = 0 
        for i in range(self.n_epochs):

            # update parametors
            self.sgd()

            mse = self.mse()            
            if ((i+1) % 10 == 0) and self.verbose:
                print("Iteration: %d ; error(MAE) = %.4f ; learn_rate = %.4f ;" % (i+1, mse, self.learning_rate))
            
            # if error improve rate is not enough, update self.learning_rate lower.
            mse_improve_rate = (before_mse-mse)/before_mse if before_mse>0 else 0
            if  mse_improve_rate < 1e-8 :
                self.learning_rate *= 0.5
                stop_cnt += 1
            # if stop_cnt is more than a threshold, stop training.
            if stop_cnt > 10:
                break
            
            before_mse = mse
        
        return self
    
    def _fit_setup(self, user_ids, item_ids, ratings, 
                   user_attributes, item_attributes):
        """
        transform user_ids and item_ids to the incremental index.
        
        Arguments example:
            user_ids = [1,1,2]
            item_ids = [0,5,0]
            ratings  = [3,3,4]
            user_attributes = {1:[0,1,1], 2:[0,0,1]}
            item_attributes = {0:[0,1], 5:[1,1]}
        """
        # set id transformer
        user_ids_transformer = util.id_transformer()
        item_ids_transformer = util.id_transformer()        
        transformed_user_ids = user_ids_transformer.fit_transform(user_ids)
        transformed_item_ids = item_ids_transformer.fit_transform(item_ids)        
        self.user_ids_transformer = user_ids_transformer
        self.item_ids_transformer = item_ids_transformer
        
        # put parameters pn self
        self.R = [(u,i,r) for u,i,r in zip(transformed_user_ids, transformed_item_ids, ratings)]
        self.num_users, self.num_items = len(set(transformed_user_ids)), len(set(transformed_item_ids))
        
        # change attributes to numpy.array as UserAttribute, ItemAttribute
        if user_attributes:
            self.n_dim_user_attributes = len(list(user_attributes.values())[0])
            self.UserAttr = np.zeros(shape=[self.num_users, self.n_dim_user_attributes])
            for _id,attr in user_attributes.items():
                transformed_id = self.user_ids_transformer.transform([_id], unknown=None)[0]
                if transformed_id is not None:
                    self.UserAttr[transformed_id, :] = attr
            self.fit_user_attributes = True
        else:
            self.fit_user_attributes = False
                        
        if item_attributes:
            self.n_dim_item_attributes = len(list(item_attributes.values())[0])
            self.ItemAttr = np.zeros(shape=[self.num_items, self.n_dim_item_attributes])
            for _id,attr in item_attributes.items():
                transformed_id = self.item_ids_transformer.transform([_id], unknown=None)[0]
                if transformed_id is not None:
                    self.ItemAttr[transformed_id, :] = attr
            self.fit_item_attributes = True
        else:
            self.fit_item_attributes = False
            
        

    def predict(self, user_ids, item_ids, user_attributes=dict(), item_attributes=dict()):
        """
        Arguments:
            user_ids [array-like object]:
                pass
            item_ids [array-like object]:
                pass
            user_attributes [dict]:
                pass
            item_attributes [dict]:
                pass
        """
        # check argument.
        if (self.fit_user_attributes) and (user_attributes==dict()):
            raise 'This instance has be fitted using user_attributes, but no attributes in the arguments.' 
        if (self.fit_item_attributes) and (item_attributes==dict()):
            raise 'This instance has be fitted using item_attributes, but no attributes in the arguments.' 
        
        # predict
        results = []
        for u,i in zip(user_ids, item_ids):
            tf_u = self.user_ids_transformer.transform([u], unknown=None)[0]
            tf_i = self.item_ids_transformer.transform([i], unknown=None)[0]

            user_attr = user_attributes.get(u, None)
            if (user_attr is None) and (self.fit_user_attributes):
                user_attr = self.UserAttr[tf_u]

            item_attr = item_attributes.get(i, None)
            if (item_attr is None) and (self.fit_item_attributes):
                item_attr = self.ItemAttr[tf_i]

            results.append(self._predict(tf_u, tf_i, user_attr, item_attr))

        return np.array(results)
    
    def predict_high_speed_but_no_preprocess(self, user_ids, item_ids, user_attributes=None, item_attributes=None):
        """ Don't Use it. This is for a specific use.
        user_ids = [0,0,1,2,]
        item_ids = [2,4,4,4,]
        """
        
        tf_us = self.user_ids_transformer.transform(user_ids, unknown=None)
        tf_is = self.item_ids_transformer.transform(item_ids, unknown=None)
        
        a_u, a_i, pq = 0, 0, 0

        b = self.b
        b_u = util.numpy_null_indexing(self.b_u, tf_us, {None:0, np.nan:0})
        b_i = util.numpy_null_indexing(self.b_i, tf_is, {None:0, np.nan:0})
        
        if (self.fit_user_attributes) and (user_attributes is not None):
            attributes_array = np.array([user_attributes.get(u, self.UserAttr[tu]) for u,tu in zip(user_ids, tf_us)])
            a_u = (self.a_u * attributes_array).mean(axis=1)
        if (self.fit_item_attributes) and (item_attributes is not None):
            attributes_array = np.array([item_attributes.get(i, self.ItemAttr[ti]) for i,ti in zip(item_ids, tf_is)])
            a_i = (self.a_i * attributes_array).mean(axis=1)
        if self.n_latent_factor:
            pq = (self.P[tf_us, :] * self.Q[tf_is, :]).sum(axis=1)
        
        return b + b_u + b_i + a_u + a_i + pq

    def mse(self):
        """
        A function to compute the total mean square error
        """
        user_ids, item_ids, ratings = [], [], []
        for u,i,r in self.R:
            user_ids.append(u)
            item_ids.append(i)
            ratings.append(r)
        
        error = 0
        for u,i,r in self.R:
            user_attr = self.UserAttr[u] if self.fit_user_attributes else None
            item_attr = self.ItemAttr[i] if self.fit_item_attributes else None
            predicted = self._predict(u, i, user_attr, item_attr)
            error += pow(r - predicted, 2)
        return np.sqrt(error)


    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for u,i,r in self.R:
            # Computer prediction and error
            user_attr = self.UserAttr[u] if self.fit_user_attributes else None
            item_attr = self.ItemAttr[i] if self.fit_item_attributes else None
            e = r - self._predict(u, i, user_attr, item_attr)
            
            # Update attribute coefficient
            if self.fit_user_attributes:
                self.a_u += self.learning_rate * self.UserAttr[u] * self.UserAttr[i].mean() * (e - self.regularization_weight * self.a_u)
            if self.fit_item_attributes:
                self.a_i += self.learning_rate * self.ItemAttr[i] * self.ItemAttr[i].mean() * (e - self.regularization_weight * self.a_i)

            # Update biases
            if self.id_bias:
                self.b_u[u] += self.learning_rate * (e - self.regularization_weight * self.b_u[u])
                self.b_i[i] += self.learning_rate * (e - self.regularization_weight * self.b_i[i])

            # Update user and item latent feature matrices
            if self.n_latent_factor:
                self.P[u, :] += self.learning_rate * (e * self.Q[i, :] - self.regularization_weight * self.P[u,:])
                self.Q[i, :] += self.learning_rate * (e * self.P[u, :] - self.regularization_weight * self.Q[i,:])
        

    def _predict(self, u, i, user_attr=None, item_attr=None):
        """
        Get the predicted rating of user u and item i.
        user_attr [np.array] is vector of user attributes.
        item_attr [np.array] in vector of item attributes.
        """
        prediction = 0

        # global bias
        if self.global_bias:
            prediction += self.b

        # user_id bias, item_id bias
        if self.id_bias:
            if u is not None:
                prediction += self.b_u[u]
            if i is not None:
                prediction += self.b_i[i]

        # attributes 
        if (self.fit_user_attributes) and (user_attr is not None):
            prediction += (self.a_u * user_attr).mean()
        if (self.fit_item_attributes) and (item_attr is not None):
            prediction += (self.a_i * item_attr).mean()
        
        # latent factor
        if self.n_latent_factor:
            if (u is not None) and (i is not None):
                prediction += np.dot(self.P[u, :], self.Q[i, :].T)

        return prediction
    

if __name__ == 'some tests':
    # Usage
    user_ids = [1,1,1,1,5,5,8,8]
    item_ids = [1,2,3,4,2,4,8,9]
    ratings  = [5,5,4,4,3,3,2,2]
    
    out_sample_user_ids = [1,5,8,10,10]
    out_sample_item_ids = [8,1,1,1,10]
    
    #######################################
    # library Suprise と比較して同様の結果を出力できるかの確認。
    from src.module.MF import MF
    n_epochs = 10000
    mf = MF(n_latent_factor=1, learning_rate=0.005, regularization_weight=0.02, n_epochs=n_epochs, verbose=True)
    mf.fit(user_ids, item_ids, ratings, )
    
    # 新しいIDの推定確認
    mf.predict([99,1,99], [1,99,99])
    
    # compair the result with the libraray 'Suprise'
    from surprise import SVD # SVD algorithm
    from src.Suprise_algo_wrapper import algo_wrapper

    svd = algo_wrapper(SVD(n_factors=1, lr_all=0.005, reg_all=0.02, n_epochs=n_epochs))
    svd.fit(user_ids, item_ids, ratings, )
    
    # 目視確認
    for me, sur in zip(mf.predict(user_ids, item_ids), svd.predict(user_ids, item_ids)):
        print(me, sur, '%.5f'%((me-sur)/sur))

    # 目視確認
    for me, sur in zip(mf.predict(out_sample_user_ids, out_sample_item_ids), svd.predict(out_sample_user_ids, out_sample_item_ids)):
        print(me, sur, '%.5f'%((me-sur)/sur))        
    
    #######################################
    # 明らかにattributeの影響を受けているデータに対して、適切に学習ができるか？
    import numpy as np
    n_sample = 1000
    user_ids = np.random.choice(range(10), size=n_sample)
    item_ids = np.random.choice(range(5), size=n_sample)
    user_attribute = {i:np.random.choice([0,1], size=2, replace=True) for i in range(10)}
    item_attribute = {i:np.random.choice([0,1], size=3, replace=True) for i in range(10)}
    
    answer_user_attr_coef = np.array([10, -5])
    answer_item_attr_coef = np.array([-3, 0, 8])
    
    rating_on_only_attribute = lambda u,i: (user_attribute[u]*answer_user_attr_coef).sum() + (item_attribute[i]*answer_item_attr_coef).sum()
    ratings = [rating_on_only_attribute(u,i) for u,i in zip(user_ids, item_ids)]
    
    from src.module.MF import MF
    mf = MF(n_latent_factor=0, learning_rate=0.010, 
            regularization_weight=0.0, n_epochs=100, 
            global_bias=False, id_bias=False, verbose=True)
    mf.fit(user_ids, item_ids, ratings, user_attribute, item_attribute)
    print(mf.a_u, answer_user_attr_coef)
    print(mf.a_i, answer_item_attr_coef)
    for p,a in zip(mf.predict(user_ids, item_ids, user_attribute, item_attribute), ratings):
        print(p,a,(p-a)/(abs(a)+0.00001))
    
    # predict_high_speed_but_no_preprocess のテスト
    from src.module.MF import MF
    mf = MF(n_latent_factor=0, learning_rate=0.010, 
            regularization_weight=0.0, n_epochs=30, 
            global_bias=True, id_bias=True, verbose=True)
    mf.fit(user_ids, item_ids, ratings, user_attribute, item_attribute)
    high_speed = mf.predict_high_speed_but_no_preprocess(user_ids, item_ids, user_attribute, item_attribute)
    normal_speed = mf.predict(user_ids, item_ids, user_attribute, item_attribute)
    assert all(high_speed==normal_speed)

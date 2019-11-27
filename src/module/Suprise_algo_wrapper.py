#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wrapper function for suprise.alogo to be able to use
.fit(user_ids, item_ids, rating) and .predict(user_ids, item_ids)
"""

import pandas as pd
import numpy as np

from surprise import Dataset
from surprise import Reader

class algo_wrapper:

    def __init__(self, algo):
        self.algo = algo

    def fit(self, user_ids, item_ids, rating, **keyargs):
        trainset = convert_to_Surprise_dataset(user_ids, item_ids, rating)
        self.algo.fit(trainset, **keyargs)

    def predict(self, user_ids, item_ids):
        predicted_result = []
        for u,i in zip(user_ids, item_ids):
            predicted_result.append(self.algo.predict(u,i).est)
        return predicted_result




def convert_to_Surprise_dataset(userID, itemID, rating):
    """
    ARGUMENTs:
        userID = [9, 32, 2, 45, 'user_foo']
        itemID = [1, 1, 1, 2, 2]
        rating = [3, 2, 4, 3, 1]
    RETURN:
        surprise.dataset.DatasetAutoFolds
    """
    ratings_dict = {
            'userID': userID,
            'itemID': itemID,
            'rating': rating,
            }
    df = pd.DataFrame(ratings_dict)

    # The columns must correspond to user id, item id and ratings (in that order).
    reader = Reader(rating_scale=(min(rating), max(rating)))
    dataset = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

    return dataset.build_full_trainset()



if __name__ == '__main__':
    from surprise import KNNBasic, SVD
    # INPUTs
    user_ids = np.random.choice(range(100), size=500)
    item_ids = np.random.choice(range(20),  size=500)
    values   = np.random.choice(range(6),   size=500)

    #algo = KNNBasic()
    algo = SVD()

    algo_w = algo_wrapper(algo)

    algo_w.fit(user_ids, item_ids, values)
    algo_w.predict(user_ids, item_ids)

    # set new id, and predict
    user_ids = np.random.choice(range(100,110), size=10)
    item_ids = np.random.choice(range(20,30),  size=10)
    algo_w.predict(user_ids, item_ids)

    # set new user_ids, and predict
    user_ids = np.random.choice(range(100,110), size=10)
    item_ids = np.random.choice(range(20),  size=10)
    algo_w.predict(user_ids, item_ids)

    # set new item_ids, and predict
    user_ids = np.random.choice(range(100), size=10)
    item_ids = np.random.choice(range(20,30),  size=10)
    algo_w.predict(user_ids, item_ids)

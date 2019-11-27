#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from datetime import datetime

def read_netflix_data():
    # set directory path
    DIR_ = './data/netflix/training_set/'    
    files = list(map(lambda x: DIR_+x, os.listdir(DIR_)))    
    frame = []
    for file in files:
        #file = files[0]
        movie_id = int(file.replace(DIR_, '').split('_')[1].replace('.txt', ''))
        data = pd.read_csv(file, names=['UserID', 'Rating', 'Timestamp'], skiprows=1)
        data['MovieID'] = movie_id
        frame.append(data)
    rating = pd.concat(frame, axis=0)
    rating['Timestamp'] = pd.to_datetime(rating['Timestamp'], format='%Y-%m-%d')
    base_dt = datetime(1900,1,1)
    rating['Timestamp'] = (rating['Timestamp'] - base_dt).dt.total_seconds()

    user   = pd.DataFrame()
    movie  = pd.DataFrame()
    
    return rating, user, movie

def read_mllatest_data():
    # set file path
    PATH_rating = './data/ml-latest/ratings.csv'
    PATH_movie  = './data/ml-latest/movies.csv'
    
    # set columns
    COLS_rating = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    COLS_movie  = ['MovieID', 'Title', 'Genres']
    
    
    # read DataSet
    rating = pd.read_csv(PATH_rating, sep=',', names=COLS_rating, skiprows=1)
    user   = pd.DataFrame()
    movie  = pd.read_csv(PATH_movie, sep=',', names=COLS_movie, skiprows=1)
    
    return rating, user, movie

def read_ml20m_data():
    # set file path
    PATH_rating = './data/ml-20m/ratings.csv'
    PATH_movie  = './data/ml-20m/movies.csv'
    
    # set columns
    COLS_rating = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    COLS_movie  = ['MovieID', 'Title', 'Genres']
    
    
    # read DataSet
    rating = pd.read_csv(PATH_rating, sep=',', names=COLS_rating, skiprows=1)
    user   = pd.DataFrame()
    movie  = pd.read_csv(PATH_movie, sep=',', names=COLS_movie, skiprows=1)
    
    return rating, user, movie
    

def read_ml100k_data():
    # set file path
    PATH_rating = './data/ml-100k/u.data'
    PATH_user   = './data/ml-100k/u.user'
    PATH_movie  = './data/ml-100k/u.item'
    
    # set columns
    COLS_rating = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    COLS_user   = ['UserID', 'Age', 'Gender', 'Occupation', 'Zip-code']
    GenreCOLS = ['Genre%02d'%d for d in range(19)]
    COLS_movie  = ['MovieID', 'Title', 'release-date', 'unknown0', 'unknown1'] + GenreCOLS
    
    
    # read DataSet
    rating = pd.read_csv(PATH_rating, sep='\t', names=COLS_rating)
    user   = pd.read_csv(PATH_user, sep='|', names=COLS_user)
    movie  = pd.read_csv(PATH_movie, sep='|', names=COLS_movie, encoding='latin-1')
    
    return rating, user, movie


def read_ml1m_data():
    # set file path
    PATH_rating = './data/ml-1m/ratings.dat'
    PATH_user   = './data/ml-1m/users.dat'
    PATH_movie  = './data/ml-1m/movies.dat'
    
    # set columns
    COLS_rating = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    COLS_user   = 'UserID::Gender::Age::Occupation::Zip-code'.split('::')
    COLS_movie = 'MovieID::Title::Genres'.split('::')
    
    # read DataSet
    rating = pd.read_csv(PATH_rating, sep='::', names=COLS_rating)
    user   = pd.read_csv(PATH_user, sep='::', names=COLS_user)
    movie  = pd.read_csv(PATH_movie, sep='::', names=COLS_movie, encoding='latin-1')
    
    return rating, user, movie



class id_transformer:
    def __init__(self):
        """
        transform ids to the index which start from 0.
        """
        pass
    def fit(self, ids):
        """
        ARGUMETs:
            ids [array-like object]: 
                array of id of user or item.        
        """
        ids_ = sorted(list(set(ids)))
        self.id_convert_dict = {i:index for index,i in enumerate(ids_)}
    
    def transform(self, ids, unknown=None):
        """
        ARGUMETs:
            ids [array-like object]: 
                array of id of user or item.                
        """
        return [self.id_convert_dict.get(i, unknown) for i in ids]

    def fit_transform(self, ids):
        self.fit(ids)
        return self.transform(ids)
        
    def inverse_transform(self, indexes, unknown=None):
        """
        ARGUMETs:
            indexes [array-like object]: 
                array of index which are transformed                
        """
        return [get_key_from_val(self.id_convert_dict, ind) for ind in indexes]
    
    def fit_update(self, ids):
        """
        ARGUMETs:
            ids [array-like object]: 
                array of id of user or item.        
        """
        ids_ = sorted(list(set(ids)))
        ids_ = [id_ for id_ in ids_ if id_ not in self.id_convert_dict.keys()]
        now_max_id = max(self.id_convert_dict.values())
        new_id_convert_dict = {i:now_max_id+1+index for index,i in enumerate(ids_)}
        self.id_convert_dict.update(new_id_convert_dict)
    
    
        
def get_key_from_val(dict_, val, unknown=None):
    """
    dict_ = {'aa':123}
    val = 123
    get_key_from_val(dict_, val)
    > 'aa'    
    """
    list_vals = list(dict_.values())
    if val in list_vals:
        return list(dict_.keys())[list_vals.index(val)]    
    else:
        return unknown

from sklearn.metrics.pairwise import cosine_similarity
def pearson_correlation_from_R(R, **key_args__cosine_similarity):
    R_np = np.array(R)

    no_0_mean = lambda array: array[array>0].mean()
    R_np_mean = np.apply_along_axis(no_0_mean, axis=1, arr=R_np)[:, None]
    
    # R_np - R_np_mean ignoring 0
    diff_R = R_np - R_np_mean
    diff_R[R_np==0] = 0
    
    return cosine_similarity(diff_R, **key_args__cosine_similarity)


def numpy_null_indexing(numpy_array, indexes, null_dict={None:0}):
    """
    return numpy_array[indexes] despite indexes have null
    
    numpy_array = np.array([0,10,20,30,40])
    indexes = [2,2,4,None,'NULL']
    null_dict = {None:-1, 'NULL':-2}
    
    numpy_null_indexing(numpy_array, indexes, null_dict)
     > array([20, 20, 40, -1, -2])
    """
    to_dict = {key:val for key,val in enumerate(numpy_array)}
    to_dict.update(null_dict)
    return np.array([to_dict[i] for i in indexes])
    


if __name__=='__main__':
    dict_ = {'aa':123}
    val = 123
    get_key_from_val(dict_, val)
    
    ids = [1,1,1,5,5,9]
    it = id_transformer()
    it.fit(ids)
    it.transform([1,5,9])
    it.transform([1,5,9,99])
    it.inverse_transform([0,1,2])
    it.inverse_transform([0,1,2,88])
    
        
    

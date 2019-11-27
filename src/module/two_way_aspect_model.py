#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is code which based on:
    @inproceedings{schein2002methods,
      title={Methods and metrics for cold-start recommendations},
      author={Schein, Andrew I and Popescul, Alexandrin and Ungar, Lyle H and Pennock, David M},
      booktitle={Proceedings of the 25th annual international ACM SIGIR conference on Research and development in information retrieval},
      pages={253--260},
      year={2002},
      organization={ACM}
    }    
"""

import copy
import numpy as np
from .PLSA import plsa
from .util import id_transformer

"""プログラミング用
import copy
import numpy as np
from src.module.PLSA import plsa
from src.module.util import id_transformer
"""

class two_way_aspect_model:
    def __init__(self, item_attributes, Z=10):
        """
        # ARGUMENTs
            Z[int]: 
                the number of latent calss
            item_attributes[dict]:
                key of the dict is item_id, the attribute vector is the values of the dict.
            postive_threshold[int,float]:
                two_way_aspect_model learn only positive touple (user_id, item_id).
                
        # Example
            Z = 2            
        """
        self.core_two_way_aspect_model = core_two_way_aspect_model(Z)
        self.item_attributes = item_attributes

    def fit(self ,user_ids, item_ids, values, postive_threshold=5):
        """
        user_ids [array like object]:
            array of user_id which correspond with item_ids.
            the set of (user_id, item_id) means that the user_id like the item_id.
        item_ids [array like object]:
            array of item_id which correspond with user_ids.
            the set of (user_id, item_id) means that the user_id like the item_id.
        values [array like object]:
            result of the preference. (the score how user like item)
            in this model, only preference which larger than postive_threshold are used.
        postive_threshold [number]:
            threshold of values. the larger value than postive_threshold is learned. 
        """
        # filter ids in values > 0
        _user_ids, _item_ids = list(), list()
        for user_id, item_id, value in zip(user_ids, item_ids, values):
            if value >= postive_threshold:
                _user_ids.append(user_id)
                _item_ids.append(item_id)
                
        # fit id_transformer
        set_item_ids = set(list(_item_ids) + list(self.item_attributes))
        set_user_ids = set(list(_user_ids))

        item_id_transformer, user_id_transformer = id_transformer(), id_transformer()
        item_id_transformer.fit(set_item_ids)
        user_id_transformer.fit(set_user_ids)

        self.item_id_transformer = item_id_transformer
        self.user_id_transformer = user_id_transformer

        # transform user_ids and item_ids to sequencial id.        
        transformed_item_ids = item_id_transformer.transform(_item_ids)
        transformed_user_ids = user_id_transformer.transform(_user_ids)
        transforemed_item_attributes = \
            {item_id_transformer.transform([_id])[0]:np.array(val) for _id,val in self.item_attributes.items()}
        
        self.transforemed_item_attributes = transforemed_item_attributes
        
        # make matrix Npa, Nma as self.core_two_way_aspect_model.fit() argument.
        ## Npa(count matrix of person-actor) is equivalent with user-attributes.
        ## Nma(count matrix of movie-actor) is equivalent with item-attributes.
        n_item_ids = len(set_item_ids)
        n_user_ids = len(set_user_ids)
        n_item_attributes = len(list(self.item_attributes.values())[0])
        Nma = np.zeros((n_item_ids, n_item_attributes), dtype=int)
        Npa = np.zeros((n_user_ids, n_item_attributes), dtype=int)
        
        for item_id, val in transforemed_item_attributes.items():
            Nma[item_id] = val
        for user_id, item_id in zip(transformed_user_ids, transformed_item_ids):
            Npa[user_id] += Nma[item_id]
            
        # fit self.core_two_way_aspect_model
        self.core_two_way_aspect_model.fit_person_aspect(Npa)
        # fit Ppa (percent of person-movie which is equivalent with user_item)
        self.core_two_way_aspect_model.fit_movie_aspect(Nma)
        
        self.Pp_m = self.core_two_way_aspect_model.get_Pp_m()

    def predict(self, user_ids, item_ids):
        transformed_item_ids = self.item_id_transformer.transform(item_ids, unknown=None)
        transformed_user_ids = self.user_id_transformer.transform(user_ids, unknown=None)

        # return predicted as np.array()
        predicted = []
        for user_id, item_id in zip(transformed_user_ids, transformed_item_ids):
            if (user_id is not None) and (item_id is not None):
                predicted.append(self.Pp_m[user_id, item_id])
            elif (user_id is None) and (item_id is not None):
                predicted.append(self.Pp_m[:, item_id].mean())
            elif (user_id is not None) and (item_id is None):
                predicted.append(self.Pp_m[user_id, :].mean())
            else:
                predicted.append(self.Pp_m[:, :].mean())
        
        return np.array(predicted)
            
        

class core_two_way_aspect_model:
    def __init__(self, Z):
        """
        # ARGUMENTs
            Z[int]: 
                the number of latent calss
        # Example
            Z = 2            
        """
        self.Z = Z
        
    def fit_person_aspect(self, Npa):    
        """
        # ARGUMENTs
            Npa[np.array]: 
                attributes matrix which dimention is [person, movie-actores]
        # Example
            import numpy as np
            Npa = np.array([
                [2, 0, 1, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 1],
                [1, 0, 2, 3]
            ])        
        """
        # fit person/actor aspect model with PLSA
        pa_model = plsa(self.Z)
        pa_model.fit(Npa)
        self.pa_model = pa_model
        

    def fit_movie_aspect(self, Nma):
        """
        # ARGUMENTs
            Nma[np.array]:
                count matrix which dimention is [movie-actors, movie]
        # Example
            import numpy as np
            Nam = np.array([
                [1, 0, 1, 0],
                [1, 0, 0, 1],
                [1, 1, 1, 0],
            ])
        """
        # the parameters of person/actor aspect model are hold constant.
        Pa_z = self.pa_model.Py_z
        
        # Fold-In  
        ma_model = plsa(self.Z)
        ma_model.fit(Nma, hold_Py_z=Pa_z)
        self.ma_model = ma_model        
    
    def get_Pp_m(self):
        # P(p|z)
        Pp = self.pa_model.Pxy.sum(axis=1)
        Pp_z = (self.pa_model.Pz_x * Pp[None,:]).T
        Pp_z /= Pp_z.sum(axis=0, keepdims=True)
        # P(z|m)
        Pz_m = self.ma_model.Pz_x
        
        # Recommendation are made using P(p|m)
        Pp_m = Pp_z.dot(Pz_m)
        Pp_m /= Pp_m.sum(axis=0, keepdims=True)
        
        return Pp_m
        


if __name__=='__main__':

    user_ids = [5,5,5,6,6,6,6,3,3,3,2,2,2,1,1]
    item_ids = [1,3,6,2,6,3,2,1,2,3,1,2,3,1,2]
    values   = [1,3,3,2,3,1,2,1,3,3,2,1,1,3,2]
    postive_threshold = 3
    item_attributes = {
            1: [1,0,1,0],
            2: [1,1,0,0],
            3: [0,1,0,0],
            6: [1,0,1,1],
            99: [1,1,1,1],
            }

    twam = two_way_aspect_model(Z=3, item_attributes=item_attributes)
    twam.fit(user_ids, item_ids, values, postive_threshold)
    twam.predict(user_ids, item_ids)
    twam.predict([5], [99]) # 学習中にないitemID
    twam.predict([5], [123]) # attributeにすらないitemID
    twam.predict([123, 123, 123, 123, 123], [1, 2, 3, 6, 99]) # 学習中にないuserID
    twam.predict([123], [123]) # itemもuserも完全に新規の場合
    
    twam.Pp_m[:, 0].mean()

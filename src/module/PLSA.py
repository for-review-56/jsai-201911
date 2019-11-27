#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A implementation of Probabilistic Latent Semantic Analysis(PLSA).

This code reffere to:
    https://qiita.com/HZama/items/0957f74f8da1302f7652
"""
import sys
import numpy as np


class plsa(object):
    def __init__(self, Z, random_seed=None):
        """
        PLSA as ( X → Z → Y )
        
        ARGUMENTs:
            Z [int]: the number of latent class.
            random_seed [int]: is set as np.random_seed(random_seed)
        """
        self.Z = Z
        np.random.seed(random_seed)

                
    def fit(self, N, k=200, t=1.0e-7, 
            hold_Pz=None, hold_Pz_x=None, hold_Py_z=None):
        """
        train with EMalgorithm.
        
        ARGUMENTs:
            N [2dim np.array]: 
                the count matrix of (x, y).
            k [int]  : 
                the number of iteration.
            t [float]: 
                the threshold to breake iterations if implovment
                is bellow than t.    
            hold_Pz [None or np.array]:
                if set np.array as P(z), 
                EM-algorithm uses it as constant. 
            hold_Px_z [None or np.array]:
                if set np.array as P(x|z), 
                EM-algorithm uses it as constant. 
            hold_Py_z [None or np.array]:
                if set np.array as P(y|z), 
                EM-algorithm uses it as constant. 
        """
        
        # --- set up ----
        self.N = N
        self.X, self.Y = N.shape[0], N.shape[1]
        self.hold_Pz , self.hold_Pz_x, self.hold_Py_z = hold_Pz, hold_Pz_x, hold_Py_z
        
        # P(z|x,y). the dimention is [z, x, y]
        self.Pz_xy = np.zeros(shape=(self.Z, self.X, self.Y))
        
        # P(z|x). the dimention is [z, x]
        if self.hold_Pz_x is None:
            self.Pz_x = np.random.rand(self.Z, self.X)
            self.Pz_x /= np.sum(self.Pz_x, axis=0, keepdims=True)
        else:
            self.Pz_x = hold_Pz_x
        
        # P(y|z). the dimention is [y, z]
        if self.hold_Py_z is None:
            self.Py_z  = np.random.rand(self.Y, self.Z)
            self.Py_z /= np.sum(self.Py_z, axis=0, keepdims=True)
        else:
            self.Py_z = hold_Py_z
        
        
        # --- train ---
        prev_log_likelihood = sys.maxsize
        for i in range(k):
            self._e_step()
            self._m_step()
            this_log_likelihood = self._log_likelihood()
            
            if abs((this_log_likelihood - prev_log_likelihood) / prev_log_likelihood) < t:
                break
            
            prev_log_likelihood = this_log_likelihood
        
    
    def _e_step(self):
        """
        E-step:
            update P(z|x,y). dimentions of Pz_xy is [z,x,y].
        """
        for z in range(self.Z):
            self.Pz_xy[z,:,:] = self.Pz_x[[z],:].T.dot(self.Py_z[:,[z]].T)
        ## add 1/sys.maxsize to avoid 0-divison error
        self.Pz_xy /= self.Pz_xy.sum(axis=0, keepdims=True) + 1/sys.maxsize
        
        
    def _m_step(self):
        """
        M-step:
            update P(z|x). dimentions of Pz_x is [z, x] 
            update P(y|z). dimentions of Py_z is [y, z]
        """
        NP = self.N[None, :, :] * self.Pz_xy

        if self.hold_Pz_x is None:
            self.Pz_x = np.sum(NP, axis=2)            
            self.Pz_x /= np.sum(self.Pz_x, axis=0, keepdims=True)

        if self.hold_Py_z is None:
            self.Py_z  = np.sum(NP, axis=1).T
            self.Py_z /= np.sum(self.Py_z, axis=0, keepdims=True)
        
    def _log_likelihood(self):
        Pxy = self.Pz_x.T.dot(self.Py_z.T)
        Pxy /= np.sum(Pxy)
        self.Pxy = Pxy
        
        # add 1/sys.maxsize to avoid log(0) error
        return np.sum(self.N * np.log(Pxy + 1/sys.maxsize))
    
        

if __name__=='__main__':
    import numpy as np
    from src.module.PLSA import plsa
    
    N = np.array([
        [20, 23, 1, 4],
        [25, 19, 3, 0],
        [2, 1, 31, 28],
        [0, 1, 22, 17],
        [1, 0, 18, 24]
    ])
    
    plsa = plsa(2)
    plsa.fit(N)

    print('P(z|x)')
    print(plsa.Pz_x)
    print('P(y|z)')
    print(plsa.Py_z)
    print('P(z|x,y)')
    print(plsa.Pz_xy)
    print('P(x,y)')
    print(plsa.Pxy)
    
    
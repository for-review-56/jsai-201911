#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from .. import PLSA

import numpy as np

"""プログラミング用
from src import PLSA

import numpy as np
"""

class test_PLSA(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test01(self):
        # set test data 
        # import pdb;pdb.set_trace()
        N = np.array([
            [20, 23, 1, 4],
            [25, 19, 3, 0],
            [2, 1, 31, 28],
            [0, 1, 22, 17],
            [1, 0, 18, 24],
            [0, 0, 0, 0],
        ])
        p = PLSA.plsa(2)
        p.fit(N)
        
        # calc P(x,y) form fitted p with Bayes theorem.
        Pz_x = p.Px_z * p.Pz[:, None]
        Pz_x /=  Pz_x.sum(axis=0)[None,:] + 1e-10
        Pxy = p.Px[:, None] * Pz_x.T.dot(p.Py_z)
        
        observed_Pxy = N / N.sum()
                
        # assertion
        diff = Pxy - observed_Pxy
        self.assertGreater(0.10, np.max(np.abs(diff)))
                                
    
    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()


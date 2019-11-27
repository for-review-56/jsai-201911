
import unittest

from .. import RandomWalkCF
import imp;imp.reload(RandomWalkCF)


class test_RandomWalkCF(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_insample_prediction(self):
        #INPUTS:
        user_ids = [1,2,3,1,2,3,4,5,6]
        item_ids = [1,1,1,2,2,3,3,3,3]
        values   = [1,2,3,1,2,1,1,1,1]
        
        rwcf = RandomWalkCF.RandomWalkCF()
        rwcf.fit(user_ids, item_ids, values)    
        
        #RETURNS:
        returned = rwcf.predict(user_ids, item_ids)

        #OUTPUTS:
        for ans, ret in zip(values, returned):
            self.assertAlmostEqual(1, abs(ret/ans), delta=0.5)
                
    def tearDown(self):
        pass
    
if __name__ == '__main__':
    unittest.main()
    


    
import unittest

import numpy as np
from numpy.testing import assert_array_less

from minipgm.variables import Value, Normal, Beta, Binomial

class TestVariables(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testNormal(self):
        mu = Value(0.)
        sigma = Value(2.)
        x = Normal(mu, sigma, size=1000, name='x')
        
        self.assertAlmostEqual(
            x.value.mean(), mu.value, delta=3 * sigma.value / np.sqrt(1000))
        
        x.sample()
        self.assertEqual(x.value.shape, (1000, ))
        self.assertEqual(x._last_value.shape, (1000, ))
        
        self.assertAlmostEqual(
            x.value.mean(), mu.value, delta=3 * sigma.value / np.sqrt(1000))
        
        x.logp()  
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
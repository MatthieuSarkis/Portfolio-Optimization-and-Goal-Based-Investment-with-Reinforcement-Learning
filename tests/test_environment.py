# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest
import pandas as pd
from src.environment import Environment

class TestEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        env = Environment(stock_market_history=pd.read_csv('tests/close.csv'),
                          initial_cash_in_bank=10000,
                          buy_cost=0.1,
                          sell_cost=0.1,
                          limit_n_stocks=200)
        
        env.reset()

    def tearDown(self):
        pass
    
    def test_step(self):
        pass
    
    
if __name__ == '__main__':
    unittest.main()

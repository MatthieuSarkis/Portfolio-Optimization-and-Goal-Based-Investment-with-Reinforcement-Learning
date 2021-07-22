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

import pandas as pd
from typing import List

class Logger():
    
    def __init__(self,
                 list_assets: List[str],
                 mode: str,
                 ) -> None:
        
        self.list_assets = list_assets
        self.logs: dict = {}
        self.number_epochs: int = None
        
        self.logs['reward_history'] = []
        
        if mode == 'test':
            self.logs['portfolio_histories'] = []
            
        if mode == 'train':
            self.logs['reward_running_average'] = []
                          
    def set_number_epochs(self,
                           number_epochs: int,
                           ) -> None:
        
        self.number_epochs = number_epochs
        
    def print_status(self) -> None:
        pass
    
    def append_to_portfolio_history(self,
                                    epoch,
                                    ) -> None:
        pass
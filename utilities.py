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

#***********************************************************************************

import os

#***********************************************************************************

def time_to_string(t):
    return t.strftime("%Y.%m.%d/%H.%M.%S")

def make_dirs(odir='saved-files', folder_name=''): 
    saving_directory = os.path.join(odir, folder_name) 
    os.makedirs(saving_directory, exist_ok=True) 
    return saving_directory
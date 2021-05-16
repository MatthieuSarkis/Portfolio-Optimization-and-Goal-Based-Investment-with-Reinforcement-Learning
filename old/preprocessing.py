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
from sklearn.preprocessing import StandardScaler

def __handle_missing_values(X, y):
    temp = pd.concat([X.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1), y['target']], axis=1).dropna()
    y = temp['target']
    X = temp.drop(['target'], axis = 1)
    return X, y

def __standardize_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

def __extract_2d_timeSequence(X):
    return X[['abs_ret{}'.format(i) for i in range(0, 61)] + ['rel_vol{}'.format(i) for i in range(0, 61)]].copy().values.reshape(-1, 61, 2)
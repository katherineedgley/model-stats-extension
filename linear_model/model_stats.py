#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class ModelStats:
    def __init__(self, fitted_model, X, y, colnames = None):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.fitted_model = fitted_model
        if (self.X.shape[0] != self.y.shape[0]):
            raise ValueError('X and y different number of samples')
        else:
            if isinstance(X, pd.DataFrame):
                self.colnames = ['Intercept'] + list(X.columns)
            elif colnames == None:
                self.colnames = ['Intercept'] + ['Variable_' + str(x+1) 
                                                 for x in range(X.shape[1])]
            else:
                self.colnames = ['Intercept'] + colnames


        
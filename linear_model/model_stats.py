#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class ModelStats:
    def __init__(self, fitted_model, X, y, colnames = None):
        
        self.fitted_model = fitted_model
        
        if isinstance(X, pd.DataFrame):
            self.colnames = list(X.columns)
        elif colnames == None:
            self.colnames = ['Intercept'] + ['Variable_' + str(x+1) for x in range(X.shape[1])]
        else:
            self.colnames = colnames
        self.X = np.array(X)
        self.y = np.array(y)


        
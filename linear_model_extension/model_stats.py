#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class ModelStats:
    '''
    General class (not to be called outside of RegressionStats) for classes
    that generate model statistics, like RegressionStats
    Input:
        fitted_model - a scikit-learn LinearRegression fitted model
        X  - feature matrix used to fit the sklearn model, can be 
                numpy array or pandas df
        y - target array used to fit the sklearn model, 
                np array or pandas series
        colnames - default None, only for supplying a list of the column
                    names when inputting X as a numpy array, which
                    does not include the names of the columns
                    If colnames=None and X is numpy array, 
                    will name the variables in order.
    '''
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


        
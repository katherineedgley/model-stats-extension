#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from scipy import stats
from .model import LinearModel

"""
Created on Wed Apr 15 16:38:20 2020

@author: redgley
"""

class OLS_Stats(LinearModel):
    
    def __init__(self, fitted_model, X, y, colnames = None):
        LinearModel.__init__(self, fitted_model, X, y, colnames)
        self.n = len(y)
        
    def get_betas(self):
        coef_intercept = self.fitted_model.intercept_
        coefs = self.fitted_model.coef_
        return np.append(coef_intercept, coefs)
    
    def add_constant(self, X):
        return np.c_[np.ones(self.n), X] 
        
    def compute_s2(self, X):
        y = self.y
        betas = self.get_betas()
        # y'y - beta'X'y
        RSS =  (y.T.dot(y) - betas.T.dot(X.T).dot(y))
        k = len(betas) - 1 # number of dependent variables
        return RSS/(self.n - k - 1)
            
    
    def compute_standard_errors(self):
        X = self.add_constant(self.X)
        k = X.shape[1]
        s2 = self.compute_s2(X)
        # covariance matrix of betas (coefficients)
        sigma = np.linalg.solve(X.T.dot(X), np.eye(k)*s2)
        return np.sqrt(np.diagonal(sigma))
            
    
    def compute_t_stats(self):
        se = self.compute_standard_errors()
        betas = self.get_betas()
        t_stat_ls = betas/se
        return t_stat_ls
    
    
    def compute_pval(self, t_stat):
        # degrees of freedom
        df  = self.n - 2
        return stats.t.sf(np.abs(t_stat), df)*2
    
    def conf_int(self, beta, se):
        t_conf = stats.t.ppf(.975, self.n)
        lower = beta - t_conf*se
        upper = beta + t_conf*se
        return lower, upper
    
    def get_summary(self):
        betas = self.get_betas()
        # standard error list
        errors = list(self.compute_standard_errors())
        t_stat_ls = list(self.compute_t_stats())
        pvals = [self.compute_pval(t_stat) for t_stat in t_stat_ls]
        conf_int_ls = [self.conf_int(beta, se) for beta,se in zip(betas,errors)]
        lower_conf_ls = [lower for lower, upper in conf_int_ls]
        upper_conf_ls = [upper for lower, upper in conf_int_ls]
        
        
        results_df = pd.DataFrame(
            {'coef': betas,
             'std err': errors,
             't': t_stat_ls,
             'P>|t|': pvals,
             '[0.025': lower_conf_ls,
             '0.975]': upper_conf_ls}, index = self.colnames)
        return results_df
    
    
    
        

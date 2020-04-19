#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from scipy import stats
from .model_stats import ModelStats

"""
Created on Wed Apr 15 16:38:20 2020

@author: redgley
"""

class RegressionStats(ModelStats):
    
    def __init__(self, fitted_model, X, y, colnames = None):
        ModelStats.__init__(self, fitted_model, X, y, colnames)
        self.n = len(y)
        self.k = X.shape[1] # num dependent variables
        
    def get_betas(self):
        coef_intercept = self.fitted_model.intercept_
        coefs = self.fitted_model.coef_
        return np.append(coef_intercept, coefs)
    
    def add_constant(self, X):
        X = np.array(X)
        n = X.shape[0]
        return np.c_[np.ones(n), X] 
    
    def compute_standard_errors(self):
        '''
        We compute the standard errors the same way that 
        statsmodels does when cov_type is nonrobust
        
        Method: 
        From the residuals 'resid', compute the estimation of 
        sigma^2 (s^2) = RSS/n-k-1 = (resid^t resid)/ n - k - 1 which 
            scales the X deviation term (X'X)^-1
        Then compute the covariance matrix of coefficients, cov_mat:
            using equation cov(beta_hat) = s^2 * (X'X)^-1
        The standard errors of coefficients are the sqrt of
            diagonal entries in the covariance matrix
        
        '''
        # add constant column to X design matrix
        X_const = self.add_constant(self.X)
        resid = np.array(self.y - self.fitted_model.predict(self.X))
        s2 = resid.T.dot(resid)/(self.n - self.k - 1)
        cov_mat = s2 * np.linalg.inv(X_const.T.dot(X_const))
        se = np.sqrt(np.diag(cov_mat))
        return se
            
    
    def compute_t_stats(self):
        se = self.compute_standard_errors()
        betas = self.get_betas()
        t_stat_ls = betas/se
        return t_stat_ls
    
    
    def compute_pval(self, t_stat):
        # degrees of freedom
        df  = self.n - 2
        return stats.t.sf(np.abs(t_stat), df)*2
    
    def compute_conf_int(self, beta, se):
        t_conf = stats.t.ppf(.975, self.n)
        lower = beta - t_conf*se
        upper = beta + t_conf*se
        return lower, upper
    
    def summary(self):
        betas = self.get_betas()
        # standard error list
        errors = list(self.compute_standard_errors())
        t_stat_ls = list(self.compute_t_stats())
        pvals = [self.compute_pval(t_stat) for t_stat in t_stat_ls]
        conf_int_ls = [self.compute_conf_int(beta, se) 
                       for beta,se in zip(betas,errors)]
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
    
    
    
        

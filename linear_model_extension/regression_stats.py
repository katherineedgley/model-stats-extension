#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from scipy import stats
from .model_stats import ModelStats


class RegressionStats(ModelStats):
    """
    Main class for computing the statistics associated with an sklearn 
    LinearRegression model. 
    """
    
    def __init__(self, fitted_model, X, y, colnames = None):
        ModelStats.__init__(self, fitted_model, X, y, colnames)
        self.n = len(y) # number of samples
        self.k = X.shape[1] # num dependent variables
        
    def get_betas(self):
        '''
        Function to extract the coefficients for intercept and 
        independent variables (features) all into one numpy array
        beginning with the intercept's coefficient
        '''
        coef_intercept = self.fitted_model.intercept_
        coefs = self.fitted_model.coef_
        return np.append(coef_intercept, coefs)
    
    def add_constant(self, X):
        '''
        Like the analagous function from statsmodels, a function
        to input a dataframe or numpy array, X, and add a constant 
        column of 1's into the array (as the first column)
        '''
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
        '''
        Function to compute the test statistic values from the standard
        errors and coefficient values. Test statistic is for the 
        null hypothesis, i.e. beta = 0. 
        t-statistics for linear regression are defined as:
            t = beta_hat/(s/sqrt(dev2(x))) = 
            beta_hat/SE(beta_hat) where the SE(beta_hat) = 
            sqrt(var(beta_hat)) = sqrt(s^2/dev2(x))
            
        where dev2(x) = sum((x_i - x.mean)^2) 
        '''
        se = self.compute_standard_errors()
        betas = self.get_betas()
        t_stat_ls = betas/se
        return t_stat_ls
    
    
    def compute_pval(self, t_stat):
        '''
        The p-value is computed with the test statistic (t_stat)
        by getting the Prob(t > |t_stat|) in the t-distribution
        with n-2 degrees of freedom.
        We then multiply by 2 as we are interested in two-tailed
        test
        '''
        # degrees of freedom
        df  = self.n - 2
        return stats.t.sf(np.abs(t_stat), df)*2
    
    def compute_conf_int(self, beta, se):
        '''
        Function to compute the bounds of 95% confidence 
        interval for a given regression coefficient, beta,
        and the associated standard error, se. 
        '''
        t_conf = stats.t.ppf(.975, self.n)
        lower = beta - t_conf*se
        upper = beta + t_conf*se
        return lower, upper
    
    def summary(self):
        '''
        Main function to call to get all the regression statistics
        output in a simple Pandas dataframe with following columns:
            coef- regression coefficient
            std err - standard error of regression coefficient
            t - test statistic for regression coefficient
            P>|t| - p-value for test statistic
            [0.025 - lower bound of 95% confidence interval
            0.975] - upper bound of 95% confidence interval
        '''
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
    
    
    
        

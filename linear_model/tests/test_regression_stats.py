from linear_model import RegressionStats
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/Boston.csv', index_col = 0)

X = df[['crim']]
y = df['dis']

short_X = np.array([1,4,5,3,6])
short_X_2 = np.array([2])
list_X = [1,3,4]

lr_model = LinearRegression()
lr_model.fit(X,y)


stats = RegressionStats(lr_model, X, y)


def test_get_betas():
    betas = stats.get_betas()
    assert(betas[0] == lr_model.intercept_)
    assert_equal(stats.get_betas()[1:],lr_model.coef_)
    

def test_add_constant():
    # add a constant column to short X matrix
    X_constant_1 = stats.add_constant(short_X)
    # define the correct matrix with constant added
    correct_1 = np.array([[1,1],[1,4],[1,5],[1,3],[1,6]])
    assert_equal(X_constant_1, correct_1)
    
    # test again for edge case
    X_constant_2 = stats.add_constant(short_X_2)
    correct_2 = np.array([[1,2]])
    assert_equal(X_constant_2, correct_2)
    
    # test when X is list
    X_constant_ls = stats.add_constant(list_X)
    correct_3 = np.array([[1,1],[1,3],[1,4]])
    assert_equal(X_constant_ls, correct_3)
    
    
def test_compute_standard_errors():
    # can't test s2 very well as it is used just as a helper function
    # for compute_standard_errors
    computed_se = stats.compute_standard_errors()
    # from statsmodels cov_params:
    verified_se = np.array([0.09403954, 0.01008802])
    assert_allclose(computed_se, verified_se, rtol=1e-5)
    
    

    
    
    
    
    
    
    
    
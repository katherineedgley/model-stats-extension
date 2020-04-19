from linear_model_extension import RegressionStats
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pandas as pd
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
    beta_ls = stats.get_betas()
    assert(beta_ls[0] == lr_model.intercept_)
    assert_equal(beta_ls[1:],lr_model.coef_)
    

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
    
    
def test_compute_t_stats():
    computed_t_stats = stats.compute_t_stats()
    verified_t_stats = np.array([43.927, -9.213]) # from statsmodels
    assert_allclose(computed_t_stats, verified_t_stats, rtol = 1e-3)
    
def test_compute_pval():
    t_stat_ls = stats.compute_t_stats()
    pval_intercept = stats.compute_pval(t_stat_ls[0])
    pval_coef = stats.compute_pval(t_stat_ls[1])
    verified_pval_intercept = 1.896759e-174
    verified_pval_coef = 8.519949e-19
    assert_allclose(pval_intercept, verified_pval_intercept, rtol=1e-5)
    assert_allclose(pval_coef, verified_pval_coef, rtol=1e-5)
    
def test_compute_conf_int():
    beta_ls = stats.get_betas()
    se_ls = stats.compute_standard_errors()
    # get confidence interval for intercept coefficient
    lower, upper = stats.compute_conf_int(beta_ls[0], se_ls[0])
    verified_conf_int = np.array([3.946146, 4.315661])
    assert_allclose(np.array([lower,upper]), verified_conf_int,
                    rtol=1e-5)
    
    # get confidence interval for coefficient
    lower, upper = stats.compute_conf_int(beta_ls[1], se_ls[1])
    verified_conf_int = np.array([-0.112765,	-0.073126])
    assert_allclose(np.array([lower,upper]), verified_conf_int,
                    rtol = 1e-5)
    
    
def test_get_summary():
    # get statsmodels summary:
    verified_summary_df = pd.DataFrame({
        'coef': [lr_model.intercept_,lr_model.coef_[0]],
        'std err':[0.09403954, 0.01008802],
        't': [43.927,-9.213],
        'P>|t|': [1.896759e-174, 8.519949e-19],
        '[0.025': [3.946146, -0.112765],
        '0.975]': [4.315661, -0.073126]})
    summary_df = stats.summary()
    for i,row in verified_summary_df.iterrows():
        assert_allclose(row, summary_df.iloc[i], rtol = 1e-4)
    
    
    
    
    

    
    
    
    
    
    
    
    
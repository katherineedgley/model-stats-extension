from linear_model_extension import RegressionStats
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression



df = pd.read_csv('data/Boston.csv', index_col = 0)

X = df[['crim']]
y = df['dis']
X1 = df[['crim','dis']]
y1 = df['age']


lr_model = LinearRegression().fit(X,y)
lr_model1 = LinearRegression().fit(X1,y1)

rs = RegressionStats(lr_model, X, y)
rs1 = RegressionStats(lr_model1, X1, y1)

def test_init():
    with pytest.raises(ValueError, match='X and y different number of samples'):
        assert(RegressionStats(lr_model, X, y[3:]))
        
def test_colnames():
    assert(rs.colnames == ['Intercept','crim'])
    assert(rs1.colnames == ['Intercept','crim','dis'])
    
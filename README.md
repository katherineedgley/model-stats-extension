# model-stats-extension
model-stats-extension is a Python module built on top of scikit-learn that
provides a more thorough statistical analysis for the linear regression
module from sci-kit learn. 

The module aims to provide quick 
statistical analysis functions for sci-kit learn LinearRegression 
(and others to come) objects without having to fit the model
with the statsmodels module. The functions were tested against
statsmodel output.


## Installation
### Dependencies

The package requires: 

joblib==0.14.1 \
numpy==1.18.2 \
pandas==1.0.3 \
scikit-learn==0.22.2\
scipy==1.4.1 \
six==1.14.0 \
numpy==1.17.4 

(or later)

Works for Python versions 3.5 or newer. 

## User Installation
`pip install model-stats-extension`

or can clone this repository, navigate to source directory, and run
`pip install .`.


## Use
To use the module, first import: \ `from linear_model_extension import RegressionStats`. \
RegressionStats is the working module which can be applied to an scikit-learn LinearRegression() model. \ 
To run, must have a **fitted** scikit-learn LinearRegression() object as well as an X and y object 
(Pandas dataframe or numpy array) e.g.:

```
from sklearn.linear_model import LinearRegression 
from linear_model_extension import RegressionStats 
lr_model = LinearRegression() 
lr_model.fit(X,y) 
stats = RegressionStats() 
stats.summary(lr_model, X, y)
```


|           | coef      | std err  | t         | p-value       | \[0.025   | 0.975\]     |
|-----------|-----------|----------|-----------|---------------|-----------|-------------|
| Intercept | 4.130904  | 0.094040 | 43.927304 | 1.896759e-174 | 3.946148  | 4.315660    |
| crim      | -0.092946 | 0.010088 | -9.213458 | 8.519949e-19  | -0.112765 | -0.073126   |



## Notes on Statistics
Statistics were computed based on well established practices, in particular using the textbook: \
Rencher, A. C., & Schaalje, G. B. (2008). Linear models in statistics. Hoboken, N.J: Wiley-Interscience.
Also consulted Python statsmodels source code (https://www.statsmodels.org/stable/index.html), and 
output from this module was tested against statsmodel.api output.


## Testing
After installing, (with pytest >= 5.4.1 installed), you can launch the
test suite when located outside the source:\
`pytest linear_model_extension`

## Citations
If using the module in publication, please cite :) 





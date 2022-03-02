"""
Adjusted R2 function for sklearn

@author: shadi.m.rachid@gmail.com
"""
'''
Sklearn provides no direct method to calculate adjusted r2
we could rely on statsmodel library or we could use a function
'''
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

# reading data
data = pd.read_csv('real_estate_price_size_year.csv')

# declaring variables
independent_variables = data[['size','year']]
dependent_variable = data['price']

# regression 
reg = LinearRegression()
reg.fit(independent_variables,dependent_variable)

def adjusted_r2(independent_variables, dependent_variable):
    r2 = reg.score(independent_variables, dependent_variable)
    n = independent_variables.shape[0]
    p = independent_variables.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2

print(adjusted_r2(independent_variables, dependent_variable))

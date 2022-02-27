"""
Feature Selection with F-regression  using sklearn

@author: shadi.m.rachid@gmail.com
"""

import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

# reading data
data = pd.read_csv('1.02. Multiple linear regression.csv')

# opening results file
results = open('results.txt', "w+")

results.write("Feature Selection with F-regression Using Sklearn \n\n Data Head:\n")
results.write(tabulate(data.head(),headers='keys', tablefmt='psql')+"\n\n Given data described as:\n")
results.write(tabulate(data.describe(),headers='keys', tablefmt='psql')+"\n\n")

# declaring variables
independent_variables = data[['SAT','Rand 1,2,3']]
dependent_variable = data['GPA']

# The regression
reg = LinearRegression()
reg.fit(independent_variables, dependent_variable)

# The f regression
f_reg = f_regression(independent_variables,dependent_variable)
'''
There are two output arrays
The first one contains the F-statistics for each of the regressions
The second one contains the p-values of these F-statistics
'''

# p-values of regression coefficients of each indeoendent variable
p_values = f_reg[1]
p_values.round(3)

res = pd.DataFrame()
res['Independent Variables'] = ['SAT','Rand 1,2,3']
res['Coefficients'] = reg.coef_
res['p values'] = p_values

results.write('The independent variables and the p-values of their coefficients:\n')
results.write(tabulate(res,headers='keys', tablefmt='psql'))
results.close()

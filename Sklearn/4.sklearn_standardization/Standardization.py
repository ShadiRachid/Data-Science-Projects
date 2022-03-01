"""
Standardization  using sklearn

@author: shadi.m.rachid@gmail.com
"""

import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# reading data
data = pd.read_csv('1.02. Multiple linear regression.csv')

# opening results file
results = open('results.txt', "w+")

results.write("Standardization Using Sklearn \n\n Data Head:\n")
results.write(tabulate(data.head(),headers='keys', tablefmt='psql')+"\n\n Given data described as:\n")
results.write(tabulate(data.describe(),headers='keys', tablefmt='psql')+"\n\n")

# declaring variables
independent_variables = data[['SAT','Rand 1,2,3']]
dependent_variable = data['GPA']

# standardization
scaler = StandardScaler()
scaler.fit(independent_variables)
scaled_independent_variables = scaler.transform(independent_variables)

res = pd.DataFrame()
res['SAT'] = scaled_independent_variables[:,0]
res['Rand 1,2,3'] = scaled_independent_variables[:,1]

results.write('The standardized data looks like:\n')
results.write(tabulate(res.head(),headers='keys', tablefmt='psql')+'\n\n Standradized data described as:\n')
results.write(tabulate(res.describe(),headers='keys', tablefmt='psql')+'\n')

# the linear regression with standardized variables
reg = LinearRegression()
reg.fit(scaled_independent_variables,dependent_variable)


results.close()

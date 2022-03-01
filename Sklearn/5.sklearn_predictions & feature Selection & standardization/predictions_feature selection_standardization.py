"""
Predictions, Feature Selection & Standardization using sklearn

@author: shadi.m.rachid@gmail.com
"""

import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression

# reading data
data = pd.read_csv('1.02. Multiple linear regression.csv')

# opening results file
results = open('results.txt', "w+")

results.write("Predictions, Feature Selection & Standardization using sklearn \n\n Data Head:\n")
results.write(tabulate(data.head(),headers='keys', tablefmt='psql')+"\n\n Given data described as:\n")
results.write(tabulate(data.describe(),headers='keys', tablefmt='psql')+"\n\n")

# declaring variables
independent_variables = data[['SAT','Rand 1,2,3']]
dependent_variable = data['GPA']

# standardization
scaler = StandardScaler()
scaler.fit(independent_variables)
scaled_independent_variables = scaler.transform(independent_variables)

standardized_data = pd.DataFrame()
standardized_data['SAT'] = scaled_independent_variables[:,0]
standardized_data['Rand 1,2,3'] = scaled_independent_variables[:,1]

results.write('The standardized data looks like:\n')
results.write(tabulate(standardized_data.head(),headers='keys', tablefmt='psql')+'\n\n Standradized data described as:\n')
results.write(tabulate(standardized_data.describe(),headers='keys', tablefmt='psql')+'\n\n')

# the linear regression with standardized variables
reg = LinearRegression()
reg.fit(scaled_independent_variables,dependent_variable)

result_dataframe = pd.DataFrame(index = ['y - intercept', 'SAT Coefficient', 'Rand 1,2,3 Coefficient'], 
                                columns=['Weight','p-value'])
result_dataframe['Weight'] = np.round([reg.intercept_, reg.coef_[0], reg.coef_[1]],4)

# Feature Selection
p_values = np.round(f_regression(scaled_independent_variables,dependent_variable)[1],4)
result_dataframe['p-value'] = [ np.nan, p_values[0], p_values[1]]

results.write('The results of the standardized regression are:\n')
results.write(tabulate(result_dataframe,headers='keys', tablefmt='psql')+'\n\n')

# Predictions with standardization 
new_data = pd.DataFrame(data=[[1700,2],[1800,1]], columns = ['SAT', 'Rand 1,2,3'])

# we need to scale the prediction values to fit our scaled model
new_scaled_data = scaler.transform(new_data)

predictions = reg.predict(new_scaled_data)

new_data['Predicted GPA'] = predictions.round(2)
results.write('Predictions of new data according to our model:\n')
results.write(tabulate(new_data,headers='keys', tablefmt='psql'))

results.close()
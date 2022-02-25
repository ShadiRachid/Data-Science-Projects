"""
Predictions from Linear Regression

@author: shadi.m.rachid@gmail.com
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('SAT_GPA_Attendance.csv')

# mapping data
data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'Yes': 1, 'No': 0})


# Regression Model
y = data['GPA']
X = data[['SAT', 'Attendance']]
x = sm.add_constant(X)
LR = sm.OLS(y, x).fit()

# predictions
'''
Predictions here are for
1) Bob: 1700 on SAT, and did not attend
2) Alice: 1670 on SAT, attended

The input data needs to be identical to the data fed into the regression (x)
x has the constant added as first column which is always 1, 
and will be added similarly in our inputs
then follows SAT and Attendance as given
'''

# creating data frame of inouts to be predicted
inputs = pd.DataFrame({'const': 1, 'SAT': [1700, 1670], 'Attendance': [0, 1]})

# dataframes arrange columns by alphebetical order, so reorder accordingly
inputs = inputs[['const', 'SAT', 'Attendance']]

# prediction method
predictions = LR.predict(inputs)
# results in a dataframe with the results of each prediction
predictionsdf = pd.DataFrame({'Predictions': predictions})

predictionsdf = inputs.join(predictionsdf)  # joining inputs with predictions

# changing indeces (not necessary)
predictionsdf.rename(index={0: 'Bob', 1: 'Alice'})

print(predictionsdf)

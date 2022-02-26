"""
Multiple Linear Regression using sklearn

@author: shadi.m.rachid@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
sns.set()

from sklearn.linear_model import LinearRegression

# reading data
data = pd.read_csv('1.02. Multiple linear regression.csv')

# opening results file
results = open('results.txt', "w+")

results.write("Multiple Linear Regression Using Sklearn \n\n Data Head:\n")
results.write(tabulate(data.head(),headers='keys', tablefmt='psql')+"\n\n Given data described as:\n")
results.write(tabulate(data.describe(),headers='keys', tablefmt='psql')+"\n\n")


# declaring variables
independent_variables = data[['SAT','Rand 1,2,3']]
dependent_variable = data['GPA']

# The regression
reg = LinearRegression()
reg.fit(independent_variables, dependent_variable)

# The r-squared values
r_squared = reg.score(independent_variables,dependent_variable)

# The y-intercept
y_intercept = reg.intercept_

# The coefficient of independent variable
coefficient_SAT = reg.coef_[0]
coefficient_Rand = reg.coef_[1]


# Writing Results to txt.file
r_squared = "{:.2e}".format(r_squared) # transforms slope to scientific notation
results.write("The R2 value: "+ str(r_squared)+ "\n")

y_intercept = "{:.2e}".format(y_intercept)
coefficient_SAT= "{:.2e}".format(coefficient_SAT)
coefficient_Rand= "{:.2e}".format(coefficient_Rand)
results.write("The Regression function given as:\n"+ 
              "GPA = " +
              str(coefficient_SAT)+ "(SAT) + " +
              str(coefficient_Rand)+ "(Rand) " +
              "+ "+ str(y_intercept)+"\n\n")

# Prediction of SAT 1650 and Rand 2
values = pd.DataFrame({'SAT':[1650],'Rand 1,2,3':[2]}) # no need to reshape
prediction = round(reg.predict(values)[0],2)
results.write("The expected GPA of a student with 1650 SAT and Rand = 2 according to our model is:\n" + 
              str(prediction))


results.close()

"""
Simple Linear Regression using sklearn

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
data = pd.read_csv('real_estate_price_size.csv')

# opening results file
results = open('results.txt', "w+")

results.write("Simple Linear Regression Using Sklearn \n\n Data Head:\n")
results.write(tabulate(data.head())+"\n\n Given data described as:\n")
results.write(tabulate(data.describe())+"\n\n")


# declaring variables
independent_variable = data['size']
dependent_variable = data['price']

# transforming independent variable input into a matrix (2D object)
# only necessary for regressions with 1 independent variables
independent_variable_matrix = independent_variable.values.reshape(-1,1)

# The regression
reg = LinearRegression()
reg.fit(independent_variable_matrix, dependent_variable)

# The r-squared values
r_squared = reg.score(independent_variable_matrix,dependent_variable)

# The y-intercept
y_intercept = reg.intercept_

# The coefficient of independent variable
coefficient = reg.coef_[0] #returns array for multiple variables

# Values of regression line
regression_values = independent_variable * coefficient + y_intercept

# Writing Results to txt.file
r_squared = "{:.2e}".format(r_squared) # transforms slope to scientific notation
results.write("The R2 value: "+ str(r_squared)+ "\n")

y_intercept = "{:.2e}".format(y_intercept)
coefficient= "{:.2e}".format(coefficient)
results.write("The Regression function given as:\n"+ 
              "Price = " +
              str(coefficient)+ "(size) " +
              "+ "+ str(y_intercept))

# Plotting
plt.scatter(independent_variable, dependent_variable)
fig = plt.plot(independent_variable, regression_values, lw = 3, c = 'orange')
plt.xlabel("Size", fontsize = '15')
plt.ylabel("Price", fontsize = 15)
plt.savefig('Price vs Area.png', bbox_inches='tight')

results.close()



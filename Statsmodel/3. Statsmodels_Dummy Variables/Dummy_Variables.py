"""
Utilizing Dummy Variables for Linear Regression

@author: shadi.m.rachid@gmail.com
"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('Dummies.csv')

'''
The transformation of categorical data into numerical is done by a method "map"
we shall do two mappings and two regressions
1) 'No': 0  and 'Yes':1
2) 'Yes' 0  and ' No': 1
'''

### Regression 1 'No': 0  and 'Yes':1 ###
# Mapping
data_1 = raw_data.copy()
data_1['Attendance'] = data_1['Attendance'].map({'Yes': 1, 'No': 0})

# Regression Model
y_1 = data_1['GPA']
X_1 = data_1[['SAT', 'Attendance']]
x_1 = sm.add_constant(X_1)
LR_1 = sm.OLS(y_1, x_1).fit()

# resulting equation

y_intercept_1 = LR_1.params[0]
SAT_coefficient_1 = LR_1.params[1]
yhat_1 = y_intercept_1 + (SAT_coefficient_1 * data_1['SAT'])

### Regression 2 'No': 1  and 'Yes':0 ###
# Mapping
data_2 = raw_data.copy()
data_2['Attendance'] = data_2['Attendance'].map({'Yes': 0, 'No': 1})

# Regression Model
y_2 = data_2['GPA']
X_2 = data_2[['SAT', 'Attendance']]
x_2 = sm.add_constant(X_2)
LR_2 = sm.OLS(y_2, x_2).fit()

# resulting equation

y_intercept_2 = LR_2.params[0]
SAT_coefficient_2 = LR_2.params[1]
yhat_2 = y_intercept_2 + (SAT_coefficient_2 * data_2['SAT'])


# Plotting
# Scatter plot where points colored depending on attendance
# Use the series 'Attendance' as color, and choose a colour map
plt.scatter(data_1['SAT'], data_1['GPA'],
            c=data_1['Attendance'], cmap='RdYlGn')

plt.xlabel('SAT')
plt.ylabel('GPA')

fig = plt.plot(data_1['SAT'], yhat_1, lw=2, c='lightcoral')
fig = plt.plot(data_2['SAT'], yhat_2, lw=2, c='springgreen')

plt.savefig('Plot', bbox_inches='tight')

'''
Notes on Results:
    the below line shows the regression model of an attendance = 0
    the above line shows the regression model of an attendance = 1
'''

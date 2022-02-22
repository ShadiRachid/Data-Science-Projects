"""
OLS Multiple Linear Regression

@author: shadi.m.rachid@gmail.com
"""
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
sns.set()

'''
    //Multiple Linear Regression\\
       + Two multiple linear regression models
       + Based on the Ordinary Least Squares (OLS) method
'''

# importing the data
data1 = pd.read_csv('GPA.csv')
data2 = pd.read_csv('real_estate_price.csv')

# initializing results.txt file
results = open('results.txt', "w+")

### First Linear Regression ###
###############################
'''
Understainding the causal relationship between SAT scores and college GPAs and 
a randomly assigned variable
'''

y1 = data1['GPA']
X1 = data1[['SAT', 'Rand 1,2,3']]

# creating regression model
x1 = sm.add_constant(X1)  # creates a y-intercept constants and adds to X1
LR1 = sm.OLS(y1, x1).fit()  # OLS linear regression model
summary_1 = LR1.summary()  # summary of regression results

# Extracting Parameters
# extracts the y-intercept value of linear regression model
y_intercept_1 = LR1.params[0]
coefficient1_1 = LR1.params[1]  # extracts the coefficient of SAT score
coefficient2_1 = LR1.params[2]  # extracts the coefficient of "Rand 1,2,3"

R2_1 = LR1.rsquared  # extracts the r2 value
R2_adj_1 = LR1.rsquared_adj  # extracts the adjusted r2 value
# extracting p value of coefficient of SAT score to check significance
P1_1 = LR1.pvalues[1]
# extracting p value of coefficient of "Rand 1,2,3" to check significance
P2_1 = LR1.pvalues[2]

'''note: here we are ignoring the p-value of the y-intercept'''

# the predicted values by regression model
yhat_1 = (coefficient1_1 * x1['SAT']) + \
    (coefficient2_1 * x1['Rand 1,2,3']) + y_intercept_1

# calculating a prediction based on our regression model
prediction1_1 = 1750  # a supposed value for SAT score = 1750
prediction2_1 = 2  # a supposed value for "Rand 1,2,3" = 2

predicted_value_1 = (coefficient1_1 * prediction1_1) + \
    (coefficient2_1 * prediction2_1) + y_intercept_1

'''note: with multiple independent variables, visualization is not important'''

# printing results to .txt file
results.write(
    "/// The Results of the First Regression: GPA vs SAT & Rand 1,2,3 \\\\\\ ")
results.write('\n')
results.write(
    '===========================================================================================')
results.write('\n\n\n\n')

results.write("The Summary of the regression analysis is: \n\n")
results.write(str(summary_1))
results.write('\n\n')
results.write('===========================================')
results.write('\n\n')

# transforms y-intercept to scientific notation
y_intercept_1 = "{:.2e}".format(y_intercept_1)
# transforms coefficient of SAT to scientific notation
coefficient1_1 = "{:.2e}".format(coefficient1_1)
# transforms coefficient of 'Rand 1,2,3' to scientific notation
coefficient2_1 = "{:.2e}".format(coefficient2_1)

results.write(
    "The OLS multiple linear regression model can be expressed as:\n")
results.write("yhat = " + coefficient1_1 + "SAT + " +
              coefficient2_1 + "'Rand 1,2,3' + " + y_intercept_1)
results.write('\n\n')
results.write("The variation in GPA is " +
              str(round(R2_1*100, 2))+"% explained by SAT scores & 'Rand 1,2,3'")
results.write('\n')
results.write('The predicted GPA of an SAT score of '+str(prediction1_1) + " and a 'Rand 1,2,3' of " +
              str(prediction2_1) + ' based on our model is: '+str(round(predicted_value_1, 2)))
results.write('\n\n')

if P1_1 < 0.05 and P2_1 < 0.05:
    # transforms p-value of SAT coeeficient to scientific notation
    P1_1s = "{:.2e}".format(P1_1)
    # transforms p-value of 'Rand 1,2,3' to scientific notation
    P2_1s = "{:.2e}".format(P2_1)
    results.write('the p-value of the SAT coefficient is: ' + P1_1s)
    results.write("the p-value of the 'Rand 1,2,3' coefficient is: " + P2_1s)
    results.write('\n')
    results.write(
        'They are both less than 0.05, meaning our result is significant and model accepted')

else:
    # transforms p-value of SAT coeeficient to scientific notation
    P1_1s = "{:.2e}".format(P1_1)
    # transforms p-value of 'Rand 1,2,3' to scientific notation
    P2_1s = "{:.2e}".format(P2_1)
    results.write('the p-value of the SAT coefficient is: ' + P1_1s)
    results.write('\n')
    results.write("the p-value of the 'Rand 1,2,3' coefficient is: " + P2_1s)
    results.write('\n')

    if P1_1 >= 0.05:
        results.write(
            'Coeficient of SAT is greater than 0.05, meaning our model is not accepted')

    if P2_1 >= 0.05:
        results.write(
            "Coeficcient of 'Rand 1,2,3' is greater than 0.05, meaning our model is not accepted")

    '''
    When the model is not accepted, we may compare the adjusted r-squared values 
    to that of a simple linear regression, indicating if we should omit a variable
    '''
    # creating simple regression model
    X = data1['SAT']
    x = sm.add_constant(X)
    LR = sm.OLS(y1, x).fit()
    summary = LR.summary()  # summary of regression results

    if LR.rsquared_adj > R2_adj_1:
        results.write('\n')
        results.write(
            "R2-adjusted of a simple model is greater than of this model => omit variable 'Rand 1,2,3'")

results.write('\n\n\n\n\n')


### Second Linear Regression ###
###############################
'''
Understainding the causal relationship between SAT scores and college GPAs and 
a year of construction
'''

y2 = data2['price']
X2 = data2[['size', 'year']]

# creating regression model
x2 = sm.add_constant(X2)  # creates a y-intercept constants and adds to X1
LR2 = sm.OLS(y2, x2).fit()  # OLS linear regression model
summary_2 = LR2.summary()  # summary of regression results

# Extracting Parameters
# extracts the y-intercept value of linear regression model
y_intercept_2 = LR2.params[0]
coefficient1_2 = LR2.params[1]  # extracts the coefficient of SAT score
coefficient2_2 = LR2.params[2]  # extracts the coefficient of "Rand 1,2,3"

R2_2 = LR2.rsquared  # extracts the r2 value
R2_adj_2 = LR2.rsquared_adj  # extracts the adjusted r2 value
# extracting p value of coefficient of size to check significance
P1_2 = LR2.pvalues[1]
# extracting p value of coefficient of year to check significance
P2_2 = LR2.pvalues[2]

'''note: here we are ignoring the p-value of the y-intercept'''

# the predicted values by regression model
yhat_2 = (coefficient1_2 * x2['size']) + \
    (coefficient2_2 * x2['year']) + y_intercept_2

# calculating a prediction based on our regression model
prediction1_2 = 580  # a supposed value for size  = 580
prediction2_2 = 2012  # a supposed value for year = 2012

predicted_value_2 = (coefficient1_2 * prediction1_2) + \
    (coefficient2_2 * prediction2_2) + y_intercept_2

'''note: with multiple independent variables, visualization is not important'''

# printing results to .txt file
results.write(
    "/// The Results of the Second Regression: price vs size & year \\\\\\ ")
results.write('\n')
results.write(
    '===========================================================================================')
results.write('\n\n\n\n')

results.write("The Summary of the regression analysis is: \n\n")
results.write(str(summary_2))
results.write('\n\n')
results.write('===========================================')
results.write('\n\n')

# transforms y-intercept to scientific notation
y_intercept_2 = "{:.2e}".format(y_intercept_2)
# transforms coefficient of size to scientific notation
coefficient1_2 = "{:.2e}".format(coefficient1_2)
# transforms coefficient of year to scientific notation
coefficient2_2 = "{:.2e}".format(coefficient2_2)

results.write(
    "The OLS multiple linear regression model can be expressed as:\n")
results.write("yhat = " + coefficient1_2 + "size + " +
              coefficient2_2 + "year + " + y_intercept_2)
results.write('\n\n')
results.write("The variation in price is " +
              str(round(R2_2*100, 2))+"% explained by size & year")
results.write('\n')
results.write('The predicted price of an apartment of size '+str(prediction1_2) + " and a year of " +
              str(prediction2_2) + ' based on our model is: '+str(round(predicted_value_2, 2)))
results.write('\n\n')

if P1_2 < 0.05 and P2_2 < 0.05:
    # transforms p-value of size coeeficient to scientific notation
    P1_2s = "{:.2e}".format(P1_2)
    # transforms p-value of year to scientific notation
    P2_2s = "{:.2e}".format(P2_2)
    results.write('the p-value of the size coefficient is: ' + P1_2s)
    results.write("the p-value of the year coefficient is: " + P2_2s)
    results.write('\n')
    results.write(
        'They are both less than 0.05, meaning our result is significant and model accepted')

else:
    # transforms p-value of size coeeficient to scientific notation
    P1_2s = "{:.2e}".format(P1_2)
    # transforms p-value of year to scientific notation
    P2_2s = "{:.2e}".format(P2_2)
    results.write('the p-value of the size coefficient is: ' + P1_2s)
    results.write('\n')
    results.write("the p-value of the year coefficient is: " + P2_2s)
    results.write('\n')

    if P1_2 >= 0.05:
        results.write(
            'Coeficient of size is greater than 0.05, meaning our model is not accepted')

    if P2_2 >= 0.05:
        results.write(
            "Coeficcient of year is greater than 0.05, meaning our model is not accepted")

    '''
    When the model is not accepted, we may compare the adjusted r-squared values 
    to that of a simple linear regression, indicating if we should omit a variable
    '''
    # creating simple regression model
    X = data2['size']
    x = sm.add_constant(X)
    LR = sm.OLS(y2, x).fit()
    summary = LR.summary()  # summary of regression results

    if LR.rsquared_adj > R2_adj_2:
        results.write('\n')
        results.write(
            "R2-adjusted of a simple model is greater than of this model => omit variable year")

results.close()

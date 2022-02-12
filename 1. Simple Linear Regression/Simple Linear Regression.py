"""
OLS Simple Linear Regression

@author: shadi.m.rachid@gmail.com
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

'''
    //Simple Linear Regression\\
       + Two simple linear regression models
       + Based on the Ordinary Least Squares (OLS) method
'''

# importing the data
data1 = pd.read_csv('SAT_GPA.csv')
data2 = pd.read_csv('real_estate_price_size.csv')

# initializing results.txt file
results = open('results.txt',"w+")

### First Linear Regression ###
###############################
'''
Understainding the causal relationship between SAT scores and college GPAs
in other words: can we create a linear regression model that predicts future college GPAs
based on SAT scores?
'''
X1 = data1['SAT']
y1 = data1['GPA']

### creating regression model
x1 = sm.add_constant(X1) # creates a y-intercept constants and adds to X1
LR1 = sm.OLS(y1,x1).fit() # OLS linear regression model
summary_1 = LR1.summary() # summary of regression results

y_intercept_1 = LR1.params[0] # extracts the y-intercept value of linear regression model
slope_1 = LR1.params[1] # extracts the slope of the linear regression model
R2_1 = LR1.rsquared # extracts the r2 value
P1 = LR1.pvalues[1] # extracting p value of slope to check significance
 
'''note: here we are ignoring the p-value of the y-intercept'''

yhat_1 = (slope_1 * x1)+ y_intercept_1 # the predicted values by regression model

### Plotting Graph
plt.scatter(X1,y1, c = 'royalblue') # scatter plot of original values
fig = plt.plot(x1, yhat_1, lw = 3, c = 'orange', label = 'regression line') # regression line of predicted values

plt.xlabel('SAT Scores', fontsize = 15)
plt.ylabel('GPA', fontsize = 15)

plt.xlim(min(X1) - (0.05*min(X1)), max(X1) + (0.05*min(X1))) #focusing graph
plt.ylim(min(y1) - (0.05*min(y1)), max(y1) + (0.05*min(y1)))
plt.savefig('1_GPA vs SAT.png', bbox_inches='tight')

### printing results to .txt file
results.write("/// The Results of the First Regression: GPA vs SAT \\\\\\ ")
results.write('===========================================================================================')
results.write('\n\n\n\n')

results.write("The Summary of the regression analysis is: \n\n")
results.write(str(summary_1))
results.write('\n\n')
results.write('===========================================')
results.write('\n\n')

y_intercept_1 = "{:.2e}".format(y_intercept_1) # transforms y-intercept to scientific notation
slope_1 = "{:.2e}".format(slope_1) # transforms slope to scientific notation


results.write("The OLS linear regression model can be expressed as:\n")
results.write("yhat = " + slope_1 + "x + "+ y_intercept_1)
results.write('\n\n')
results.write("The variation in GPA is " + str(round(R2_1*100,2))+"% explained by SAT scores")
results.write('\n\n')

if P1 < 0.05:
    P1 = "{:.2e}".format(P1) # transforms p-value to scientific notation
    results.write('the p-value of the slope is: ' + P1)
    results.write('\n')
    results.write('It is less than 0.05, meaning our result is significant and model accepted')
    
else:
    P1 = "{:.2e}".format(P1) # transforms p-value to scientific notation
    results.write('the p-value of the slope is: ' + P1)
    results.write('\n')
    results.write('It is greater than 0.05, meaning our model is not accepted')

results.write('\n\n\n\n\n')





### Second Linear Regression ###
###############################
'''
Understainding the causal relationship between real estate  area and prices
in other words: can we create a linear regression model that predicts price of real estate based on size?
'''
X2 = data2['size']
y2 = data2['price']

### creating regression model
x2 = sm.add_constant(X2) # creates a y-intercept constants and adds to X2
LR2 = sm.OLS(y2,x2).fit() # OLS linear regression model
summary_2 = LR2.summary() # summary of regression results

y_intercept_2 = LR2.params[0] # extracts the y-intercept value of linear regression model
slope_2 = LR2.params[1] # extracts the slope of the linear regression model
R2_2 = LR2.rsquared # extracts the r2 value
P2 = LR2.pvalues[1] # extracting p value of slope to check significance
 
'''note: here we are ignoring the p-value of the y-intercept'''

yhat_2 = (slope_2 * x2)+ y_intercept_2 # the predicted values by regression model

### Plotting Graph
plt.scatter(X2,y2) # scatter plot of original values
fig = plt.plot(x2, yhat_2, lw = 3, c = 'orange', label = 'regression line') # regression line of predicted values

plt.xlabel('Area', fontsize = 15)
plt.ylabel('Price', fontsize = 15)

plt.xlim(min(X2) - (0.1*min(X2)), max(X2) + (0.1*min(X2))) #focusing graph
plt.ylim(min(y2) - (0.1*min(y2)), max(y2) + (0.1*min(y2)))
plt.savefig('2_Price vs Area.png', bbox_inches='tight')

### printing results to .txt file
results.write("/// The Results of the Second Regression: Price vs Size \\\\\\ ")
results.write('===========================================================================================')
results.write('\n\n\n\n')

results.write("The Summary of the regression analysis is: \n\n")
results.write(str(summary_2))
results.write('\n\n')
results.write('===========================================')
results.write('\n\n')

y_intercept_2 = "{:.2e}".format(y_intercept_2) # transforms y-intercept to scientific notation
slope_2 = "{:.2e}".format(slope_2) # transforms slope to scientific notation


results.write("The OLS linear regression model can be expressed as:\n")
results.write("yhat = " + slope_2 + "x + "+ y_intercept_2)
results.write('\n\n')
results.write("The variation in Price is " + str(round(R2_2*100,2))+"% explained by Price")
results.write('\n\n')

if P2 < 0.05:
    P2 = "{:.2e}".format(P2) # transforms p-value to scientific notation
    results.write('the p-value of the slope is: ' + P2)
    results.write('\n')
    results.write('It is less than 0.05, meaning our result is significant and model accepted')

else:
    P2 = "{:.2e}".format(P2) # transforms p-value to scientific notation
    results.write('the p-value of the slope is: ' + P2)
    results.write('\n')
    results.write('It is greater than 0.05, meaning our model is not accepted')

results.close()
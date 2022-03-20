"""
train test splits from sklearn

@author: shadi.m.rachid@gmail.com
"""

import pandas as pd

from sklearn.model_selection import train_test_split

# reading data
data = pd.read_csv('real_estate_price_size_year.csv')

# declaring variables
x = data[['size','year']]
y = data['price']

# Split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=365)

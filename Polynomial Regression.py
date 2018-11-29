"""
Created on Sat Jun  9 18:08:17 2018

@author: Sidneaux
"""
#Importing usual libraries

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#importing a dataset
df = pd.read_csv('Position Salary.csv')
X = df.iloc[:, 1:2].values 
y = dataset.iloc[:, 2].values

from sklearn.Cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Fitting polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#visualizing the linear regression model
plt.scatter(X,y, color = 'red')
plt.plot(X,lin_reg_2.predict(X), color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position level')
plt.show()

#visualizing the polynomial regression result
#playing with the incremental values of X
"""X_grid = np.arange(min(X), max(X), 0.10)
X_grid = X_grid.reshape((len(X_grid),1))
then replace X in visualization by X_grid"""
 
plt.scatter(X,y, color = 'red')
plt.plot(X,lin_reg.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position level')
plt.show()

#predicting with linear regression
lin_reg.predict(6.5)

#predicting with Polynomial regression
lin_reg.predict(poly_reg.fit_transform(6.5))
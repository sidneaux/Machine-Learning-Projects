"""
Created on Sat Jun  9 23:14:14 2018

@author: Sidneaux
Decision Tree Regression
"""

#Importing usual libraries

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#importing a dataset
df = pd.read_csv('Position Salary.csv')
X = df.iloc[:, 1:2].values 
y = dataset.iloc[:, 2].values

"""from sklearn.Cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)"""

#Fitting the regression model 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
#predicting with Decision Tree regression
y_pred = regressor.predict(6.5)
 
#For higher resolutiona and smoother curve
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color = 'red')
plt.plot(X,regressor.predict(X)), color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position level')
plt.show()


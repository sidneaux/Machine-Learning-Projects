"""
Created on Sat Jun  9 19:01:02 2018

@author: Sidneaux
Support Vector Regression (SVR)
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

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Fitting the SVR model 
from sklearn.svm import SVR
#using the gaussian kernel rbf
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#predicting with SVR
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

 
plt.scatter(X,y, color = 'red')
plt.plot(X,regressor.predict(X)), color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position level')
plt.show()

#For higher resolutiona and smoother curve
X_grid = np.arange(min(X), max(X), 0.10)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color = 'red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position level')
plt.show()


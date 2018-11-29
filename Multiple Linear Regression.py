"""
Created on Thu Jun  7 21:53:01 2018
@author: sidneaux
Multiple Linear Regression
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#importing a dataset
df = pd.read_csv('50_startups.csv')
X = df.iloc[:, -1].values 
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3]= labelencoder_X.fit_transform(X[:, 3]) 
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoidind the dummy variable trap
X = X[:, 1:]

#using label encoder (il y a deux categorie de y)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#Splitting into training and test sets
from sklearn.Cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set result
y_pred = regressor.predict(X_test)

#Building optimal model using Backward Elimination Methods
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
#taking the index of X
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#remove the non significant variable
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
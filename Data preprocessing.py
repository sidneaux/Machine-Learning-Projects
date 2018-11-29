### Sidneaux's preprocessing template 
#Importing usual libraries

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#importing a dataset
df = pd.read_csv('Data.csv')
X = df.iloc[:, -1].values #ne marche pas, vous besoin de donnes
y = dataset.iloc[:, 3].values
#Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy='mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3]= imputer.transform((X[:, 1:3]

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0]= labelencoder_X.fit_transform(X[:, 0]) #pour la premiere column de X
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#using label encoder (il y a deux categorie de y)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting into training and test sets
from sklearn.Cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set result
y_pred = regressor.predict(X_test)

#visualizing the training set result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color= 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabe('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualizing the test set result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color= 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabe('Years of Experience')
plt.ylabel('Salary')
plt.show()
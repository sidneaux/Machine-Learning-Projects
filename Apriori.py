"""
Created on Sun Jun 17 20:16:15 2018

@author: Sidneaux
Apriori
"""
#Importing the usual libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Data.csv', header = None)
transactions = []
for i in range(0,7501):
    transaction.append([str(dataset.values[i,j]), for j in range(0, 20)])

#Training Apriori onj the dataset
from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#Visualizing the result
results = list(rules)
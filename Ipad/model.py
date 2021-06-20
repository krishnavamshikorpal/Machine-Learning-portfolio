# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:23:32 2020

@author: Krishna Vamshi
"""

import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

data = pd.read_csv("Data Ipad.csv")
data.head()

x = data.iloc[:,1:]
y = data.iloc[:,0]  

def screen_to_int(word):
    word_dict = {'Mini':1,'Air':2,'Pro':3}
    return word_dict[word]

def capacity_to_int(word):
    word_dict = {'16GB':1,'32GB':2,'64GB':3,'128GB':4}
    return word_dict[word]

def connectivity_to_int(word):
    word_dict = {'Wifi':1,'Cellular':2,'wifi':1}
    return word_dict[word]

def gen_to_int(word):
    word_dict = {'Previous':1,'Current':2,'current':2}
    return word_dict[word]

x['Screen'] = x['Screen'].apply(lambda x:screen_to_int(x))
x['Capacity'] = x['Capacity'].apply(lambda x:capacity_to_int(x))
x['Connectivity'] = x['Connectivity'].apply(lambda x:connectivity_to_int(x))
x['Gen'] = x['Gen'].apply(lambda x:gen_to_int(x))

lr = LinearRegression()
lr.fit(x,y)
#lr_pred = lr.predict(x)
#print(r2_score(y,lr_pred))

pickle.dump(lr, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[1,2,1,2]]))
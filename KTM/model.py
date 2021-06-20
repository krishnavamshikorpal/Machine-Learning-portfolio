# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 21:56:14 2020

@author: Krishna Vamshi
"""
import sklearn

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

df=pd.read_excel("ktm1.xlsx")
df.drop(['ID'], axis=1,inplace=True )
le = LabelEncoder()
df['Response'] = le.fit_transform(df['Response'])

x1 = df.iloc[:,:-1]
y1= df.iloc[:,-1]

def gender_to_int(word):
    word_dict = {'Male':1,'Female':0,'Male ':1}
    return word_dict[word]

def occupation_to_int(word):
    word_dict = {'Student':1,'Unemployed':2,'Professional':3,'Self Employed':4}
    return word_dict[word]

def phonetype_to_int(word):
    word_dict = {'Low End':1,'Average':2,'High End':3}
    return word_dict[word]

def currentbike_to_int(word):
    word_dict = {'No Bike':1,'Below 125':2,'125 to 180':3,'180 to 220':4,'220 and Above':5}
    return word_dict[word]

def relation_to_int(word):
    word_dict = {'Single':1,'Committed':2,'Complicated':3,'Married':4}
    return word_dict[word]

x1['Gender']=x1['Gender'].apply(lambda x1:gender_to_int(x1))
x1['Occupation']=x1['Occupation'].apply(lambda x1:occupation_to_int(x1))
x1['Phone Type']=x1['Phone Type'].apply(lambda x1:phonetype_to_int(x1))
x1['Current Bike']=x1['Current Bike'].apply(lambda x1:currentbike_to_int(x1))
x1['Relationship']=x1['Relationship'].apply(lambda x1:relation_to_int(x1))

lr = LogisticRegression()
lr.fit(x1,y1)


pickle.dump(lr,open('model.pkl','wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict_proba([[76,1,1,3,3,1]]))







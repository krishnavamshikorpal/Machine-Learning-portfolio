# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 22:43:36 2020

@author: Krishna Vamshi
"""

import numpy as np
import pickle
from flask import Flask,render_template,jsonify,request


## Intializing the flask application
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

## Routing the application to root folder
@app.route('/')
def home():
    return render_template('index.html')

## Routing to the prediction outcome
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_data = [np.array(int_features)]
    prediction = model.predict(final_data)
    output = prediction[0]
    
    return render_template('index.html',prediction_text='Price of IPAD is USD{}'.format(round(output,2)))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    final_data = [np.array(list(data.values()))]
    prediction = model.predict(final_data)
    return jsonify(prediction[0])

if __name__=='__main__':
    app.run(debug=True)
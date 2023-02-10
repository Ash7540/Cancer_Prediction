from flask import Flask, make_response, request, render_template, url_for, redirect
import io
from io import StringIO
import csv
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle
from werkzeug.utils import secure_filename
import json
import os
from datetime import datetime
from os.path import join, dirname, realpath

import pandas as pd
import mysql.connector


app = Flask(__name__)
model = pickle.load(open('cvmodel.pkl', 'rb'))

with open('config.json', 'r') as c:
    params = json.load(c)["params"]


@app.route("/")
def home():
    return render_template("index.html", params=params)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    if output == "Low":
        return render_template('index.html', params=params,
                               prediction_text="The Cancer Level is Low")
    elif output =="Medium":
        return render_template('index.html', params=params,
                               prediction_text="The Cancer Level is Medium")
    elif output =="High":
        return render_template('index.html', params=params,
                               prediction_text="The Cancer Level is High")
    else:
        return render_template('index.html', params=params,
                               prediction_text="Incorrect Data")

@app.route("/individual")
def individual():
    return render_template("individual.html", params=params)                              

@app.route("/about")
def about():
    return render_template("about.html", params=params)


@app.route("/dataset")
def dataset():
    return render_template("dataset.html", params=params)


if __name__ == '__main__':
    app.run(debug=True)

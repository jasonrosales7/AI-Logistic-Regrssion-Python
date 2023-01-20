import flask
from flask import Flask, render_template, request
import json
import pickle as pk
import numpy as np

import pandas as pd

app = Flask(__name__)

print("loading pickled model... ")
trained_model = pk.load(open("model.pkl", "rb"))
label_encoder_map = pk.load(open("label_encoder_map.pkl", "rb"))


def init():
    print("initializing... ")

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route("/result", methods=['GET', 'POST'])
def result():
    # break down the values incoming from your form
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    print(to_predict_list)

    # build a data frame out of the incoming values
    to_predict = np.array(to_predict_list).reshape(1, 10)
    print(to_predict)
    df = pd.DataFrame(to_predict,columns=['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type', 'avg_glucose_level', 'bmi' , 'smoking_status'])

    # lowercase & encode the data captured from the form submit with your pickled encoder
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    df.replace(label_encoder_map, inplace=True)
    print(df)

    # make the prediction of your encoded submitted data with your pickled trained model
    y_pred = trained_model.predict_proba(df)
    print("X=%s, Predicted=%s" % (df.to_numpy, y_pred[:, 1]))

    # return the results to your predict screen
    return render_template('result.html', prediction=y_pred[:, 1])

if __name__ == '__main__':
    init()
    app.run(debug=True, port=9090)

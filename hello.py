from flask import Flask, request
from flask_cors import cross_origin
import ml_model_db
import json

app = Flask(__name__)

import numpy as np
import scipy
import psycopg2
import pandas

# @cross_origin(origins="https://webuild-ai.herokuapp.com/")
@app.route('/marco', methods=['GET'])
@cross_origin()
def marco():
    return 'polo'

@app.route('/train', methods=["GET", 'POST'])
@cross_origin()
def train():
    jsonData = request.get_json()
    data = jsonData['data']
    print("received: ", data['participant_id'], data['request_type'])
    response = ml_model_db.run_model(data)
    this_beta, soft_loss, num_correct, n_test = response
    return { "status": "OK", "received": { 'weights': this_beta } }

@app.route('/evaluate', methods=['POST'])
@cross_origin()
def evaluate():
    data = request.form['data']
    try:
        response = ml_model_score.score_instances(data)
    except Exception as e:
        return {"status": "Unscuccessful"}

    return { "status": "OK", "received": data }

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request
from flask_cors import cross_origin

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

@app.route('/train', methods=['POST'])
@cross_origin()
def train():
    data = request.form['data']
    return { "train": "some data", "received": data }

@app.route('/evaluate', methods=['POST'])
@cross_origin()
def evaluate():
    data = request.form['data']
    return { "evaluate": "some data", "received": data }

if __name__ == "__main__":
    app.run()
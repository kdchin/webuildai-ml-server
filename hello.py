from flask import Flask
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

if __name__ == "__main__":
    app.run()
from flask import Flask
app = Flask(__name__)

import numpy as np
import scipy
import psycopg2
import pandas

@app.route('/marco', methods=['GET'])
def marco():
    return 'polo'

if __name__ == "__main__":
    app.run()
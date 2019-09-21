from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import cross_origin
from flask_migrate import Migrate
from flask import jsonify
import json
import os

app = Flask(__name__)
app.config.update(
    TESTING=True,
    DEBUG = True,
    CSRF_ENABLED = True,
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY" , "shrek-donkey-fiona"),
    SQLALCHEMY_DATABASE_URI = os.environ.get('FLASK_DATABASE_URL', os.environ.get('DATABASE_URL')),
)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

from ml_model_db import run_model
from ml_model_score import score_instances
from models import ModelWeights

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
    response = run_model(data, db)
    this_beta, soft_loss, num_correct, n_test = response
    return jsonify({"status": "OK", "weights": list(this_beta)})

@app.route('/evaluate', methods=['POST'])
@cross_origin()
def evaluate():
    jsonData = request.get_json()
    data = jsonData['data']
    scores, ids = score_instances(data)
    a = zip(ids, list(scores))
    sortedIds = list(map(lambda pair: pair[0], sorted(a, key=lambda x: x[1], reverse=True)))
    return { "status": "OK", "order": sortedIds, "scores" : dict(a) }

if __name__ == "__main__":
    app.run(debug=True)
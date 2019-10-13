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
    um_model = response
    return jsonify({"status": "OK", "weights": list(um_model)})

@app.route('/evaluate', methods=['POST'])
@cross_origin()
def evaluate():
    jsonData = request.get_json()
    data = jsonData['data']
    scores, ids = score_instances(data)
    a = zip(ids, scores)
    sortedIds = list(map(lambda pair: pair[0], sorted(a, key=lambda x: x[1], reverse=True)))
    scores = list(map(lambda score: round(score, 2), scores))
    return jsonify({"status": "OK", "order": sortedIds, "scores" : scores})

if __name__ == "__main__":
    app.run(debug=True)
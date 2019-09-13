from flask import Flask, request
from flask_cors import cross_origin
import ml_model_db
import ml_model_score
from flask import jsonify
import json

app = Flask(__name__)


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
    response = ml_model_db.run_model(data)
    this_beta, soft_loss, num_correct, n_test = response

    return jsonify({"status": "OK", "weights": list(this_beta)})

@app.route('/evaluate', methods=['POST'])
@cross_origin()
def evaluate():
    jsonData = request.get_json()
    data = jsonData['data']
    scores, ids = ml_model_score.score_instances(data)
    a = zip(list(scores), ids)
    sortedIds = list(map(lambda pair: pair[1], sorted(a, key=lambda x: x[0], reverse=True)))
    return { "status": "OK", "order": sortedIds }

if __name__ == "__main__":
    app.run(debug=True)
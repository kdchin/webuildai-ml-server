from hello import db
from sqlalchemy.dialects.postgresql import JSON

class ModelWeights(db.Model):
    __tablename__ = 'model_weights'
    
    id = db.Column(db.Integer, primary_key=True)
    participant_id = db.Column(db.Integer)
    feedback_round = db.Column(db.Integer)
    category = db.Column(db.String())
    weights = db.Column(JSON)

    def __repr__(self):
        return '<id:{} part_id:{} fid:{} category:{} weights:{}>'.format(self.id, self.participant_id, self.feedback_round, self.category, self.weights)
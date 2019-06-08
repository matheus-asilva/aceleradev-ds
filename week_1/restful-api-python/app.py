from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import Classifier

app = Flask(__name__)
api = Api(app)

model = Classifier()

clf_path = './models/Sentiment_Classifier.pkl'
with open(clf_path, 'rb') as file:
    model.clf = pickle.load(file)

vec_path = './models/TFIDF_Vectorizer.pkl'
with open(vec_path, 'rb') as file:
    model.vectorizer = pickle.load(file)

# Argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictSentiment(Resource):
    def get(self):
        # Use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        # Vectorize the user's query and make a prediction
        user_query_vectorized = model.vectorizer_transform(np.array([user_query]))
        pred = model.predict(user_query_vectorized)
        pred_proba = model.predict_proba(user_query_vectorized)

        # Output either 'Negative' or 'Positive' along with the score
        if pred == 0:
            pred_text = 'Negative'
        else:
            pred_text = 'Positive'
        
        # Rounds the predicted proba value and set it to a new variable
        confidence = round(pred_proba[0], 3)

        # Creates a JSON object
        output = {'prediction': pred_text, 'confidence': confidence}

        return output

# Setup the Api resource and route the URL to the resource
api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.run(debug=True)
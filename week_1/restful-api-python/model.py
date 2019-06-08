import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from utils import plot_roc

class Classifier(object):
    
    def __init__(self):
        """Simple NLP
            Attributes:
                clf: lightgbm classifier
                vectorizor: TFIDF vectorizer
        """
        self.clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.1, random_state=42)
        self.vectorizer = TfidfVectorizer()
    
    def vectorizer_fit(self, X):
        """Fits a TFIDF to the text
        """
        self.vectorizer.fit(X)
    
    def vectorizer_transform(self, X):
        """Transform the text data to a sparse TFIDF matrix
        """
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def train(self, X, y):
        """Trains the classifier to the associate label with the sparse matrix
        """
        self.clf.fit(X, y)

    def predict_proba(self, X):
        """Returns the probability for the binary class '1' in a numpy array
        """
        y_proba = self.clf.predict_proba(X)
        return y_proba[:,1]
    
    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred
    
    def pickle_vectorizer(self, path='./models/TFIDF_Vectorizer.pkl'):
        """Saves the trained vectorizer
        """
        with open(path, 'wb') as file:
            pickle.dump(self.vectorizer, file)
            print('Dumped vectorizer at:', path)
    
    def pickle_clf(self, path='./models/Sentiment_Classifier.pkl'):
        """Saves the trained classifier
        """
        with open(path, 'wb') as file:
            pickle.dump(self.clf, file)
            print('Dumped classifier at:', path)
    
    def plot_roc(self, X, y, x_size=12, y_size=12):
        """Plot the ROC curve for X_test and y_test
        """
        plot_roc(self.clf, X, y, x_size, y_size)
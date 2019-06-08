from model import Classifier
import pandas as pd
from sklearn.model_selection import train_test_split

def build_model():
    clf = Classifier()

    with open('./data/train.tsv') as file:
        df = pd.read_csv(file, sep='\t')

    pos_neg = df[(df['Sentiment'] == 0) | (df['Sentiment'] == 4)]
    pos_neg['Binary'] = pos_neg.apply(
        lambda x: 0 if x['Sentiment'] == 0 else 1, axis=1
    )

    clf.vectorizer_fit(pos_neg.loc[:, 'Phrase'])
    print('Vectorizer fit complete...')

    X = clf.vectorizer_transform(pos_neg.loc[:, 'Phrase'])
    print('Vectorizer transform complete...')

    y = pos_neg.loc[:,'Binary']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=42)

    clf.train(X_train, y_train)
    print('Model training complete...')

    clf.pickle_clf()
    clf.pickle_vectorizer()

    clf.plot_roc(X_val, y_val)

if __name__ == '__main__':
    build_model()
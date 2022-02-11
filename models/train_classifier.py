import sys
import re
import nltk
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    #New.db is database_filepath, New is table name
    engine = create_engine('sqlite:///' + database_filepath) 
    df = pd.read_sql_table('Data_DisasterResponse',engine)
    X = df.message.values
    y = df.iloc[:, 4:]
    print(X.shape,'\n',y.columns,'\n',y.shape)
    category_names = y.columns
    return X, y, category_names


def tokenize(message):
    tokens = word_tokenize(message)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    
    model = Pipeline([
        ('features',FeatureUnion([
    
    ('text_pipeline',Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
      ])),
        
      ('starting_verb', StartingVerbExtractor())
    ])),
    
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return model


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    print(classification_report(y_test, y_pred, target_names = category_names))


def save_model(model, model_filepath):
    
    #model_filepath = Disaster_Response.pk
    with open (model_filepath, 'wb') as f:
        pickle.dump(model, f)
print('Trained model is saved.')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
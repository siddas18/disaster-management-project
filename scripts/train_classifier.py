import sys
import nltk
nltk.download(['punkt','wordnet'])
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,classification_report
from sklearn.externals import joblib

def load_data(database_filepath):
    full_name = 'sqlite:///' + database_filepath
    engine = create_engine(full_name)
    df = pd.read_sql_table('tweets',engine)
    X = df.message
    Y = df.iloc[:,4:]
    category_names = Y.drop_duplicates()
    return X,Y,category_names

def tokenize(text):
    token_txt = word_tokenize(text)
    lemma=WordNetLemmatizer()
    refined=[]
    for tok in text:
        refined.append(lemma.lemmatize(tok.lower().strip()))
    return refined

def build_model():
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('mclass',MultiOutputClassifier(RandomForestClassifier()))
     ])
    parameters = {
        'mclass__estimator__n_estimators' : [10, 50],
        'vect__ngram_range' : ((1, 1), (1, 2)),
        'vect__max_features' : (10000, None),
        'tfidf__use_idf' : (True, False)
    }
    cv = RandomizedSearchCV(pipeline, parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for pos,label in enumerate(Y_test.columns):
        print(classification_report(Y_test[label].values, Y_pred[:,pos]))

def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import pickle

def load_data(database_filepath):
    
    '''
    Load Data Function: Loads the data
    
    Arguments: database_filepath: Address of the data
               
    Output: Returns the features X, target y and the category names for the data
    '''   
    
    engine = create_engine('sqlite:///'+database_filepath)
    
    df = pd.read_sql_table('df',con=engine)
    
    X = df['message']
    
    y = df.iloc[:,4:]
    
    category_names = y.columns
    
    return X, y, category_names



def tokenize(text):
    
    '''
   tokenize Function: tokenizes the text messages
    
    Arguments: text: data to be tokenized
               
    Output: Clean Tokenized Text
    '''   
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
   build_model Function: Builds the pipeline, adds parameters and performs a grid search
    
    Arguments: None
               
    Output: Returns the model
    '''   
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters =  {'tfidf__use_idf': (True, False),
              'clf__estimator__n_estimators': [5, 25]
              } 

    
    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    '''
    evaluate Function: Evaulates the model performance
    
    Arguments: model: the model to be used on the dataset
               X_test: Test Features
               y_test: Test Target 
               category_names: Category Names
               
    Output: Outputs a Classification report
    '''   
    y_pred = model.predict(X_test)
    for i, col in enumerate(y_test):
        print('------------------------- \n Column_Name: {}\n'.format(col))
        print(classification_report(y_test[col], y_pred[:, i]))
        

def save_model(model, model_filepath):
    '''
    save_model Function: saves the model to a pickle file
    
    Arguments: model: model to be saved
               model_filepath: The path to which model is to be saved
               
    Output: saves the model as a pickle file to the target destination
    '''   
    pickle.dump(model, open(model_filepath, 'wb'))

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
import sys
import re
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


def preprocessing(data):
    # convert reviews to lowercase
    data[['Review']] = data[['Review']].applymap(lambda x: x.lower()) 

    # strip punctuation
    data[['Review']] = data[['Review']].applymap(lambda x: re.sub(r'[^\w\s]','',x))
    # convert new line to space
    data[['Review']] = data[['Review']].applymap(lambda x: re.sub(r'\n',' ',x))

    # remove stopwords
    stopwords = set(['a','ako','ali','bi','bih','bila','bili','bilo','bio','bismo','biste','biti',
    'da','do','duz','ga','hoce','hocemo','hocete','hoces','hocu','i','iako','ih',
    'ili','iz','ja','je','jedna','jedne','jedno','jer','jesam','jesi','jesmo','jeste',
    'jesu','joj','jos','ju','kada','kako','kao','koja','koje','koji','kojima','koju',
    'kroz','li','me','mene','meni','mi','mimo','moj','moja','moje','mu','na','nad',
    'nakon','nam','nama','nas','nas','nasa','nase','naseg','ne','nego','neka','neki',
    'nekog','neku','nema','neko','nece','necemo','necete','neces','necu','nesto','ni',
    'nije','nikoga','nisam','nisi','nismo','niste','nisu','njega','njegov','njegova',
    'njegovo','njemu','njen','njena','njeno','njih','njihov','njihova','njihovo','njim',
    'njima','njoj','nju','no','o','od','odmah','on','ona','oni','ono','ova','pa','pak',
    'po','pod','pored','pre','s','sa','sam','samo','se','sebe','sebi','si','smo','ste',
    'su','sve','svi','svog','svoj','svoja','svoje','svom','ta','tada','taj','tako','te',
    'tebe','tebi','ti','to','toj','tome','tu','tvoj','tvoja','tvoje','u','uz','vam','vama',
    'vas','vas','vasa','vase','vec','vi','vrlo','za','zar','ce','cemo','cete','ces','cu','sta','sto']) 

    data[['Review']] = data[['Review']].applymap(lambda x: ' '.join((stopwords ^ set(x.split(' '))) & set(x.split(' '))))
    
    return data

def bag_of_words(data):
    all_dicts = []
    for row in data.iterrows():
        content = row[1]['Review']
        content = content.split(' ')
        try:
            content.remove('')
        except:
            pass
        word_dict = {el:1 for el in set(content)}
        all_dicts.append(word_dict)   
    return all_dicts

def fit(X, y):
    clf = LinearSVC(C=1)
    clf.fit(X, y)
    return clf

def evaluate(clf, X):
    result = clf.predict(X)
    return result

def load_data(file_path):
    return pd.read_csv(file_path, sep='\t')

def create_validation_set(data):
    validation_data = data[:round(len(data)/5)] # training:validation 80:20 
    data = data[round(len(data)/5):]

    return data, validation_data

def main():
    train_set_path, test_set_path = sys.argv[1], sys.argv[2]

    train_set = load_data(train_set_path)
    test_set = load_data(test_set_path)
    train_set, validation_set = create_validation_set(train_set)

    train_set = preprocessing(train_set)
    validation_set = preprocessing(validation_set)
    test_set = preprocessing(test_set)

    all_dicts = bag_of_words(train_set)
    val_dicts = bag_of_words(validation_set)
    test_dicts = bag_of_words(test_set)

    dict_vectorizer = DictVectorizer(sparse=False)
    train_X = dict_vectorizer.fit_transform(all_dicts)
    validation_X = dict_vectorizer.transform(val_dicts)
    test_X = dict_vectorizer.transform(test_dicts)

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X, train_set['Sentiment'])
    validation_X = scaler.transform(validation_X)
    test_X = scaler.transform(test_X)

    clf = fit(train_X, train_set['Sentiment'])
    y_pred = evaluate(clf, validation_X)
    y_pred_test = evaluate(clf, test_X)

    print('validation ', accuracy_score(validation_set['Sentiment'], y_pred))
    print('test ', accuracy_score(test_set['Sentiment'], y_pred_test))

if __name__ == "__main__":
    main()
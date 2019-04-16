import sys
import re
import pandas as pd


def preprocessing(data):
    # convert reviews to uppercase
    data = data[['Review']].applymap(lambda x: x.lower()) 

    # strip punctuation
    data = data[['Review']].applymap(lambda x: re.sub(r'[^\w\s]','',x))

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
    # NAPOMENA!!! set ne cuva redosled reci iz liste
    # ukoliko je redosled bitan, promeni set u neku strukturu koja cuva redosled(orderedDict?)
    data = data[['Review']].applymap(lambda x: ' '.join((stopwords ^ set(x.split(' '))) & set(x.split(' '))))
    
    return data

def vectorization():
    pass

def fit():
    pass

def evaluate():
    pass

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

    preprocessing(train_set)
    preprocessing(validation_set)


    vectorization()
    fit()
    evaluate()

if __name__ == "__main__":
    main()
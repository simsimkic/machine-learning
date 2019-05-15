import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import f1_score


def calculate_mean(data):
    mean = sum(list(filter(lambda x: not np.isnan(x), data))) / len(list(filter(lambda x: not np.isnan(x), data)))
    return round(mean)

def handle_empty_values(data_set):
    # remove rows with missing gender value
    data_set.dropna(subset=['gender'], inplace=True)

    # replaces missing score values with mean
    data_set['writing score'].replace(np.nan, calculate_mean(data_set['writing score']), inplace=True)
    data_set['reading score'].replace(np.nan, calculate_mean(data_set['reading score']), inplace=True)
    data_set['math score'].replace(np.nan, calculate_mean(data_set['math score']), inplace=True)
    

def create_validation_set(data):
    validation_data = data[:round(len(data)/5)] # training:validation 80:20 
    data = data[round(len(data)/5):]

    return data, validation_data

def main():
    train_set_path, test_set_path = sys.argv[1], sys.argv[2]

    # read data
    train_set = pd.read_csv(train_set_path)
    test_set = pd.read_csv(test_set_path)

    # split data to training and validation set
    train_set, validation_set = create_validation_set(train_set)
    
    # handle empty values
    handle_empty_values(train_set)
    handle_empty_values(validation_set)
    handle_empty_values(test_set)

    # label encoding
    le = preprocessing.LabelEncoder()

    train_set['gender'] = le.fit_transform(train_set['gender'])
    test_set['gender'] = le.transform(test_set['gender'])
    validation_set['gender'] = le.transform(validation_set['gender'])

    train_set['parental level of education'] = le.fit_transform(train_set['parental level of education'])
    test_set['parental level of education'] = le.transform(test_set['parental level of education'])
    validation_set['parental level of education'] = le.transform(validation_set['parental level of education'])

    train_set['lunch'] = le.fit_transform(train_set['lunch'])
    test_set['lunch'] = le.transform(test_set['lunch'])
    validation_set['lunch'] = le.transform(validation_set['lunch'])

    train_set['test preparation course'] = le.fit_transform(train_set['test preparation course'])
    test_set['test preparation course'] = le.transform(test_set['test preparation course'])
    validation_set['test preparation course'] = le.transform(validation_set['test preparation course'])

    label_list = list(train_set.columns)
    label_list.remove('race')
    train_X = train_set[label_list]
    validation_X = validation_set[label_list]
    test_X = test_set[label_list]

    # clf = AdaBoostClassifier()
    # clf = BaggingClassifier()
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
    clf.fit(train_X, train_set['race'])
    y_pred_val = clf.predict(validation_X)
    y_pred = clf.predict(test_X)
    
    print('val:', f1_score(validation_set['race'], y_pred_val, average='micro'))
    print(f1_score(test_set['race'], y_pred, average='micro'))


if __name__ == "__main__":
    main()
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn import utils
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


def create_validation_set(data):
    validation_data = data[:round(len(data)/5)] # training:validation 80:20 
    data = data[round(len(data)/5):]
    return data, validation_data

def calculate_mean(data):
    mean = sum(list(filter(lambda x: not np.isnan(x), data))) / len(list(filter(lambda x: not np.isnan(x), data)))
    return round(mean)

def handle_empty_values(data_set):
    # replaces missing values with mean
    data_set['year'].replace(np.nan, calculate_mean(data_set['year']), inplace=True)
    data_set['age'].replace(np.nan, calculate_mean(data_set['age']), inplace=True)
    data_set['wage'].replace(np.nan, calculate_mean(data_set['wage']), inplace=True)

    # replace missing values with most frequent value
    data_set['maritl'].replace(np.nan, data_set['maritl'].mode().values[0], inplace=True)
    data_set['race'].replace(np.nan, data_set['race'].mode().values[0], inplace=True)
    data_set['education'].replace(np.nan, data_set['education'].mode().values[0], inplace=True)
    data_set['jobclass'].replace(np.nan, data_set['jobclass'].mode().values[0], inplace=True)
    data_set['health_ins'].replace(np.nan, data_set['health_ins'].mode().values[0], inplace=True)

def label_encoding(train_set, test_set, validation_set):
    le = preprocessing.LabelEncoder()

    train_set['maritl'] = le.fit_transform(train_set['maritl'])
    test_set['maritl'] = le.transform(test_set['maritl'])
    validation_set['maritl'] = le.transform(validation_set['maritl'])

    train_set['race'] = le.fit_transform(train_set['race'])
    test_set['race'] = le.transform(test_set['race'])
    validation_set['race'] = le.transform(validation_set['race'])

    train_set['education'] = le.fit_transform(train_set['education'])
    test_set['education'] = le.transform(test_set['education'])
    validation_set['education'] = le.transform(validation_set['education'])

    train_set['jobclass'] = le.fit_transform(train_set['jobclass'])
    test_set['jobclass'] = le.transform(test_set['jobclass'])
    validation_set['jobclass'] = le.transform(validation_set['jobclass'])

    train_set['health_ins'] = le.fit_transform(train_set['health_ins'])
    test_set['health_ins'] = le.transform(test_set['health_ins'])
    validation_set['health_ins'] = le.transform(validation_set['health_ins'])

    return train_set, test_set, validation_set

def main():
    train_set_path, test_set_path = sys.argv[1], sys.argv[2]

    # read data from file
    train_set = pd.read_csv(train_set_path)
    test_set = pd.read_csv(test_set_path)

    # handle empty values in train set
    handle_empty_values(train_set)

    # split data to training and validation set
    train_set, validation_set = create_validation_set(train_set)

    # label encoding for categorical values
    train_set, test_set, validation_set = label_encoding(train_set, test_set, validation_set)

    label_list = list(train_set.columns)
    label_list.remove('wage')
    train_X = train_set[label_list]
    validation_X = validation_set[label_list]
    test_X = test_set[label_list]

    # pca
    pca = PCA(n_components=5)
    train_X = pca.fit_transform(train_X)
    validation_X = pca.transform(validation_X)
    test_X = pca.transform(test_X)

    # classification
    le = preprocessing.LabelEncoder()
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
    clf.fit(train_X, le.fit_transform(train_set['wage']))
    y_pred_val = clf.predict(validation_X)
    y_pred = clf.predict(test_X)

    # score
    print('val:', f1_score(le.fit_transform(validation_set['wage']), y_pred_val, average='micro'))
    print(f1_score(le.fit_transform(test_set['wage']), y_pred, average='micro'))

if __name__ == "__main__":
    main()
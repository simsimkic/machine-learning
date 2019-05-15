import sys
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import v_measure_score
from sklearn.mixture import GaussianMixture
# import matplotlib.pyplot as plt


# def plot_data(x, y):
#     plt.title('Data')
#     plt.xlabel('d1')
#     plt.ylabel('level')
#     plt.scatter(x, y)
#     plt.show()

def create_validation_set(data):
    validation_data = data[:round(len(data)/10)] # training:validation 90:10 
    data = data[round(len(data)/10):]

    return data, validation_data

def find_outliers(outliers, lower_bound, higher_bound, row, index):
    l = lower_bound['d1'][row['level']]
    h = higher_bound['d1'][row['level']]
    if row['d1'] < l or row['d1'] > h:
        outliers.append(index)

    l = lower_bound['d2'][row['level']]
    h = higher_bound['d2'][row['level']]
    if row['d2'] < l or row['d2'] > h:
        outliers.append(index)

    l = lower_bound['d3'][row['level']]
    h = higher_bound['d3'][row['level']]
    if row['d3'] < l or row['d3'] > h:
        outliers.append(index)

    l = lower_bound['d4'][row['level']]
    h = higher_bound['d4'][row['level']]
    if row['d4'] < l or row['d4'] > h:
        outliers.append(index)

    l = lower_bound['d5'][row['level']]
    h = higher_bound['d5'][row['level']]
    if row['d5'] < l or row['d5'] > h:
        outliers.append(index)

    return outliers

def remove_outliers(data):
    outliers = []
    lower_bound = data.groupby(['level']).mean() - 2*data.groupby(['level']).std()
    higher_bound = data.groupby(['level']).mean() + 2*data.groupby(['level']).std()

    for index, row in data.iterrows():
            outliers = find_outliers(outliers, lower_bound, higher_bound, row, index)

    outliers = list(dict.fromkeys(outliers))
    data = data.drop(data.index[outliers])
    return data

def main():
    train_set_path, test_set_path = sys.argv[1], sys.argv[2]

    # read data
    train_set = pd.read_csv(train_set_path)
    test_set = pd.read_csv(test_set_path)

    train_set = remove_outliers(train_set)
    # plot_data(train_set['d1'], train_set['level'])

    # split data to training and validation set
    train_set, validation_set = create_validation_set(train_set)

     # label encoding
    le = preprocessing.LabelEncoder()

    train_set['level'] = le.fit_transform(train_set['level'])
    test_set['level'] = le.transform(test_set['level'])
    validation_set['level'] = le.transform(validation_set['level'])

    label_list = list(train_set.columns)
    label_list.remove('level')
    label_list.remove('d1')
    label_list.remove('d2')
    label_list.remove('d3')
    label_list.remove('d4')
    train_X = train_set[label_list]
    validation_X = validation_set[label_list]
    test_X = test_set[label_list]

    clf = GaussianMixture(n_components=4)
    clf.fit(train_X, train_set['level'])
    y_pred_val = clf.predict(validation_X)
    y_pred = clf.predict(test_X)

    print('val:', v_measure_score(validation_set['level'], y_pred_val))
    print(v_measure_score(test_set['level'], y_pred))

if __name__ == "__main__":
    main()
import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def gradient_descent(x, y, theta, step_size, iterations, n):
    theta_history = []
    cost_history = []

    for i in range(iterations):
        step_size = step_size/(i+1)
        error = np.dot(x, theta) - y
        theta = theta - (step_size * (1/n) * np.dot(x.T, error))
        theta_history.append(theta)
        cost = calculate_cost(theta, x, y, n)
        cost_history.append(cost)
    
    return theta_history, cost_history

def predict(x_data, theta):
    predicted = []
    for x in x_data:
        predicted.append(theta[0] + theta[1]*x + theta[2]*(x**2) + theta[3]*(x**3) + theta[4]*(x**4) + theta[5]*(x**5))
    return predicted


def calculate_rmse(y_true, y_predict):
    sum_error = 0.0
    for i in range(len(y_true)):
        prediction_error = y_predict[i] - y_true[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(y_true))
    return math.sqrt(mean_error)

def calculate_cost(theta, x, y, n):
    prediction = np.dot(x, theta)
    error = prediction - y
    cost = 1/(2*n) * np.dot(error.T, error)
    return cost

def plot_cost_function(cost_history):
    plt.title('Cost Function J')
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.plot(cost_history)
    plt.show()

def plot_data(x, y):
    plt.title('Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(x, y)
    plt.show()

def remove_outliers(data):
    outliers = []
    x_set = set(data['X'])
    data_list = data.values.tolist()

    for x_value in x_set:
        filtered_y = [xy[1] for xy in data_list if xy[0] == x_value]
        y_mean = np.asarray(filtered_y).mean()
        y_std = np.asarray(filtered_y).std()
        for y_value in filtered_y:
            if y_value < (y_mean - 2*y_std) or y_value > (y_mean + 2*y_std):
                outliers.append([x_value, y_value])

    for o in outliers:
        data_list.remove(o)

    return pd.DataFrame(data_list, columns=['X','Y'])['X'], pd.DataFrame(data_list, columns=['X','Y'])['Y'], data_list

def read_file(file_path):
    data = pd.read_csv(file_path)
    return data

def main():
    train_set_path, test_set_path = sys.argv[1], sys.argv[2]

    data = read_file(train_set_path)
    test_data = read_file(test_set_path)
    
    x, y, data_list = remove_outliers(data)

    # x = (x - x.mean()) / x.std() # skaliranje(normalizacija) obelezja - mean normalization
    y = (y - min(y)) / (max(y) - min(y))
    # y = (y - y.mean()) / y.std()

    # plot_data(test_data['X'], test_data['Y'])

    x = np.c_[np.ones(x.shape[0]), x, x ** 2, x ** 3, x ** 4, x ** 5]

    step_size = 1
    iterations = 100
    # step_size = 0.01
    # iterations = 1000
    n = y.size
    np.random.seed(2)
    theta = np.random.rand(6)

    theta_history, cost_history = gradient_descent(x, y, theta, step_size, iterations, n)
    theta = theta_history[-1]
    print("Gradient Descent: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]))

    predicted = predict(test_data['X'], theta)
    
    # plot_cost_function(cost_history) 
   
    # y_test = test_data['Y']
    # y_test = (y_test - min(y_test)) / (max(y_test) - min(y_test))
    
    predicted = (predicted + min(predicted)) * (max(predicted) - min(predicted))
    # print(test_data['Y'])
    # print('----------')
    # print(predicted)

    rmse = calculate_rmse(test_data['Y'], predicted)
    print(rmse)

if __name__ == "__main__":
    main()
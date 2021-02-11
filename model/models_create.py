import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def next_temp_svr(temperature, valve_level, sample_num, file_name):
    np.random.seed(42)

    model = SVR(kernel='rbf')

    dates = temperature.index
    x = []
    y = []
    for i in range(0, len(dates) - 1):
        if i - sample_num + 1 >= 0:
            tup = []
            tup.append(valve_level.iloc[i]["value"])
            for j in range(0, sample_num):
                tup.append(temperature.iloc[i - j]["value"])
            x.append(tup)
            y.append(temperature.iloc[i+1]["value"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
    params = {"gamma": ["scale", "auto"],
              "tol": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
              "C": np.arange(0.2, 3.2, 0.2)
              }
    gs_regressor = GridSearchCV(model, params)
    gs_regressor.fit(x_train, y_train)
    pred = gs_regressor.predict(x_test)
    fig, ax = plt.subplots()
    ax.plot(abs(pred - y_test), 'ro')
    print(f'Średni błąd bezwzględny predykcji temperatury: {mean_absolute_error(y_test, pred)}')
    plt.show()
    regressor_file = open(file_name, 'wb')
    pickle.dump(gs_regressor.best_estimator_, regressor_file)
    regressor_file.close()


def next_valve_svr(temperature, valve_level, sample_num, file_name):
    np.random.seed(42)

    model = SVR(kernel='rbf')

    dates = temperature.index
    x = []
    y = []
    for i in range(0, len(dates) - 1):
        if i - sample_num + 1 >= 0:
            tup = []
            tup.append(temperature.iloc[i]["value"])
            for j in range(0, sample_num):
                tup.append(valve_level.iloc[i - j]["value"])
            x.append(tup)
            y.append(valve_level.iloc[i + 1]["value"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
    params = {"gamma": ["scale", "auto"],
              "tol": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
              "C": np.arange(0.2, 3.2, 0.2)
              }
    gs_regressor = GridSearchCV(model, params)
    gs_regressor.fit(x_train, y_train)
    pred = gs_regressor.predict(x_test)
    fig, ax = plt.subplots()
    ax.plot(abs(pred - y_test), 'ro')
    print(f'Średni błąd bezwzględny predykcji otwarcia zaworu: {mean_absolute_error(y_test, pred)}')
    plt.show()
    regressor_file = open(file_name, 'wb')
    pickle.dump(gs_regressor.best_estimator_, regressor_file)
    regressor_file.close()
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from pathlib import Path


def test_model_next_temp(temperature, valve_level, sample_num, file_name):
    np.random.seed(42)

    with Path(file_name).open('rb') as reg_file:
        reg = pickle.load(reg_file)

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
            y.append(temperature.iloc[i + 1]["value"])
    pred = reg.predict(x)
    fig, ax = plt.subplots()
    ax.plot(abs(pred - y), 'ro')
    print(f'Średni błąd bezwzględny predykcji temperatury: {mean_absolute_error(y, pred)}')
    plt.show()


def test_model_next_valve(temperature, valve_level, sample_num, file_name):
    np.random.seed(42)

    with Path(file_name).open('rb') as reg_file:
        reg = pickle.load(reg_file)

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
    pred = reg.predict(x)
    fig, ax = plt.subplots()
    ax.plot(abs(pred - y), 'ro')
    print(f'Średni błąd bezwzględny predykcji otwarcia zaworu: {mean_absolute_error(y, pred)}')
    plt.show()
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import models_create
import pickle
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    temperature_learn = pd.read_csv("../data/office_1_temperature_supply_points_data_2020-03-05_2020-03-19.csv", index_col=0, parse_dates=True)
    temperature_learn = temperature_learn[temperature_learn["serialNumber"] == "00158D000192D255"].resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')
    temperature_test = pd.read_csv("../data/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv", index_col=0, parse_dates=True)
    temperature_test = temperature_test[temperature_test["serialNumber"] == "00158D000192D255"].resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')
    target_temp_learn = pd.read_csv("../data/office_1_targetTemperature_supply_points_data_2020-03-05_2020-03-19.csv", index_col=0, parse_dates=True)
    target_temp_learn = target_temp_learn[target_temp_learn["serialNumber"] == "00158D000192D255"].resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')
    target_temp_test = pd.read_csv("../data/office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv", index_col=0, parse_dates=True)
    target_temp_test = target_temp_test[target_temp_test["serialNumber"] == "00158D000192D255"].resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')
    valve_level_learn = pd.read_csv("../data/office_1_valveLevel_supply_points_data_2020-03-05_2020-03-19.csv", index_col=0, parse_dates=True)
    valve_level_learn = valve_level_learn[valve_level_learn["serialNumber"] == "00158D000192D255"].resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')
    valve_level_test = pd.read_csv("../data/office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv", index_col=0, parse_dates=True)
    valve_level_test = valve_level_test[valve_level_test["serialNumber"] == "00158D000192D255"].resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')
    print(15*"-")
    print("UCZENIE")
    models_create.next_temp_svr(temperature_learn, valve_level_learn, 3, "reg_temp.p")
    models_create.next_valve_svr(temperature_learn, valve_level_learn, 3, "reg_valve.p")
    # plt.plot(temperature_learn.index, temperature_learn["value"],
    #          target_temp_learn.index, target_temp_learn["value"],
    #          valve_level_learn.index, valve_level_learn["value"])
    # plt.show()
    # Testing on different file
    print(15 * "-")
    print("TESTOWANIE")


if __name__ == '__main__':
    main()

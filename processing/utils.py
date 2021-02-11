from typing import Tuple
import pickle
import pandas as pd
from pathlib import Path


def predict_next_values_method(temperature, valve_level, file_name_temp, file_name_valve):
    # Teaching model additional data
    sample_num = 3
    with Path("model/"+file_name_temp).open('rb') as reg_file_temp:
        reg_temp = pickle.load(reg_file_temp)
    with Path("model/"+file_name_valve).open('rb') as reg_file_valve:
        reg_valve = pickle.load(reg_file_valve)
    dates = temperature.index
    x_temp = []
    x_valve = []
    y_temp = []
    y_valve = []
    for i in range(0, len(dates) - 1):
        if i - sample_num + 1 >= 0:
            tup_temp = []
            tup_valve = []

            tup_temp.append(valve_level.iloc[i]["value"])

            tup_valve.append(temperature.iloc[i]["value"])
            for j in range(0, sample_num):
                tup_valve.append(valve_level.iloc[i - j]["value"])
                tup_temp.append(temperature.iloc[i - j]["value"])
            x_temp.append(tup_temp)
            x_valve.append(tup_valve)

            y_valve.append(valve_level.iloc[i + 1]["value"])
            y_temp.append(temperature.iloc[i + 1]["value"])
    reg_temp.fit(x_temp, y_temp)
    reg_temp.fit(x_valve, y_valve)
    # Making predictions for average
    df_pred = pd.DataFrame(columns=["Temperature", "Valve_level"])
    for i in range(0, sample_num):
        df_pred.loc[pd.Timestamp(temperature.index.values[-sample_num + i])] = [0.0, 0.0]
        df_pred.iloc[-1]["Temperature"] = temperature.iloc[-sample_num + i]["value"]
        df_pred.iloc[-1]["Valve_level"] = valve_level.iloc[-sample_num + i]["value"]

    for i in range(0, 32):
        x_pred_temp = []
        x_pred_valve = []
        x_pred_temp.append(df_pred.iloc[-1]["Valve_level"])
        x_pred_valve.append(df_pred.iloc[-1]["Temperature"])
        for j in range(0, sample_num):
            x_pred_temp.append(df_pred.iloc[-(1+j)]["Temperature"])
            x_pred_valve.append(df_pred.iloc[-(j+1)]["Valve_level"])
        df_pred.loc[pd.Timestamp(df_pred.index.values[-1]) + pd.Timedelta(minutes=15)] = [0.0, 0.0]
        df_pred.iloc[-1]["Temperature"] = reg_temp.predict([x_pred_temp])[0]
        df_pred.iloc[-1]["Valve_level"] = reg_valve.predict([x_pred_valve])[0]

    avg_temp = df_pred["Temperature"].values[3:].sum() / 32
    avg_valve = df_pred["Valve_level"].values[3:].sum() / 32

    return avg_temp, avg_valve


def perform_processing(
        temperature: pd.DataFrame,
        target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,
        serial_number_for_prediction: str
) -> Tuple[float, float]:
    # NOTE(MF): sample how this can be done
    # data = preprocess_data(temperature, target_temperature, valve_level)
    # model_temp, model_valve = load_models()
    # or load model once at the beginning and pass it to this function
    # predicted_temp = model_temp.predict(data)
    # predicted_valve_level = model_valve.predict(data)
    # return predicted_temp, predicted_valve_level
    temperature_resampled = temperature[temperature["serialNumber"] == serial_number_for_prediction].resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')
    target_temperature_resampled = target_temperature[target_temperature["serialNumber"] == serial_number_for_prediction].resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')
    valve_level_resampled = valve_level[valve_level["serialNumber"] == serial_number_for_prediction].resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')

    return predict_next_values_method(temperature_resampled, valve_level_resampled, "reg_temp.p", "reg_valve.p")

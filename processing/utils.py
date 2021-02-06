from typing import Tuple

import pandas as pd


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

    return 20.99, 75
